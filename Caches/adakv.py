from __future__ import annotations

"""Ada-KV head-adaptive cache compression.

The reducer implements the adaptive head-budget allocation from Ada-KV on the
mutable historical cache only.  It keeps a dense padded representation for
compatibility with Hugging Face caches, together with a per-KV-head validity
mask.  The Ada-KV-specific SDPA attention adapter consumes that mask, making
padded entries mathematically invisible.
"""

from dataclasses import dataclass
import math
import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AdaKVConfig:
    enabled: bool = False
    total_budget: Optional[int] = None
    importance_source: str = "attention"
    observation_window: int = 32
    kernel_size: int = 7
    pooling: str = "maxpool"
    floor_alpha: float = 0.2
    min_budget_per_kv_head: int = 1
    gqa_reduction: str = "mean"
    attention_backend: str = "sdpa"
    compatibility_mode: str = "exact"
    log_every_reductions: int = 100

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "AdaKVConfig":
        raw = dict(raw or {})
        return cls(
            enabled=bool(raw.get("enabled", False)),
            total_budget=(
                None if raw.get("total_budget", None) is None
                else int(raw["total_budget"])
            ),
            importance_source=str(raw.get("importance_source", "attention")).lower(),
            observation_window=int(raw.get("observation_window", 32)),
            kernel_size=int(raw.get("kernel_size", 7)),
            pooling=str(raw.get("pooling", "maxpool")).lower(),
            floor_alpha=float(raw.get("floor_alpha", 0.2)),
            min_budget_per_kv_head=int(raw.get("min_budget_per_kv_head", 1)),
            gqa_reduction=str(raw.get("gqa_reduction", "mean")).lower(),
            attention_backend=str(raw.get("attention_backend", "sdpa")).lower(),
            compatibility_mode=str(raw.get("compatibility_mode", "exact")).lower(),
            log_every_reductions=int(raw.get("log_every_reductions", 100)),
        )

    def validate(self, mutable_token_size: int) -> None:
        if not self.enabled:
            return
        if self.compatibility_mode != "exact":
            raise ValueError(
                "This integration implements only compatibility_mode='exact'. "
                "No silent union/FIFO approximation is available."
            )
        if self.attention_backend != "sdpa":
            raise ValueError(
                "Exact padded Ada-KV requires attention_backend='sdpa' so the "
                "per-head validity mask is enforced."
            )
        if self.importance_source != "attention":
            raise ValueError("Ada-KV currently requires importance_source='attention'")
        if self.observation_window <= 0:
            raise ValueError("adakv.observation_window must be > 0")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("adakv.kernel_size must be a positive odd integer")
        if self.pooling not in {"maxpool", "avgpool"}:
            raise ValueError("adakv.pooling must be 'maxpool' or 'avgpool'")
        if not 0.0 <= self.floor_alpha <= 1.0:
            raise ValueError("adakv.floor_alpha must be in [0, 1]")
        if self.min_budget_per_kv_head < 0:
            raise ValueError("adakv.min_budget_per_kv_head must be >= 0")
        if self.gqa_reduction not in {"mean", "max"}:
            raise ValueError("adakv.gqa_reduction must be 'mean' or 'max'")
        budget = self.total_budget if self.total_budget is not None else mutable_token_size
        if int(budget) <= 0:
            raise ValueError("Ada-KV total_budget must be > 0")

    def average_budget(self, mutable_token_size: int) -> int:
        return int(self.total_budget if self.total_budget is not None else mutable_token_size)


def aggregate_query_heads_to_kv_heads(
    query_head_scores: torch.Tensor,
    num_kv_heads: int,
    reduction: str = "mean",
) -> torch.Tensor:
    """Map [B, Hq, L] importance to [B, Hkv, L] for GQA.

    Qwen-style GQA assigns contiguous groups of query heads to one KV head.
    """
    if query_head_scores.dim() != 3:
        raise ValueError(
            f"Expected query-head scores [B,Hq,L], got {tuple(query_head_scores.shape)}"
        )
    num_query_heads = int(query_head_scores.shape[1])
    num_kv_heads = int(num_kv_heads)
    if num_kv_heads <= 0 or num_query_heads % num_kv_heads != 0:
        raise ValueError(
            f"Inconsistent GQA mapping: num_query_heads={num_query_heads}, "
            f"num_kv_heads={num_kv_heads}"
        )
    groups = num_query_heads // num_kv_heads
    grouped = query_head_scores.view(
        query_head_scores.shape[0], num_kv_heads, groups, query_head_scores.shape[-1]
    )
    reduction = str(reduction).lower()
    if reduction == "mean":
        return grouped.mean(dim=2)
    if reduction == "max":
        return grouped.max(dim=2).values
    raise ValueError(f"Unsupported GQA reduction: {reduction}")


def _deterministic_integer_projection(
    desired: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    target_sum: int,
) -> torch.Tensor:
    """Round desired budgets under box constraints and an exact global sum."""
    if desired.dim() != 1:
        raise ValueError("Budget projection expects one-dimensional tensors")
    target_sum = int(target_sum)
    lower = lower.to(dtype=torch.long, device=desired.device)
    upper = upper.to(dtype=torch.long, device=desired.device)
    if torch.any(lower < 0) or torch.any(upper < lower):
        raise ValueError("Invalid lower/upper budget bounds")
    min_sum = int(lower.sum().item())
    max_sum = int(upper.sum().item())
    if target_sum < min_sum or target_sum > max_sum:
        raise ValueError(
            f"Requested total budget {target_sum} is outside feasible range "
            f"[{min_sum}, {max_sum}]"
        )

    clipped = torch.minimum(torch.maximum(desired, lower.to(desired.dtype)), upper.to(desired.dtype))
    out = torch.floor(clipped).to(torch.long)
    out = torch.minimum(torch.maximum(out, lower), upper)
    delta = target_sum - int(out.sum().item())

    # Normally |delta| < H. Clamping can make it larger, so apply deterministic
    # chunked water-filling rather than a token-by-token Python loop.
    while delta != 0:
        if delta > 0:
            capacity = upper - out
            active = capacity > 0
            if not bool(active.any()):
                raise RuntimeError("Unable to satisfy Ada-KV total budget (no upward capacity)")
            residual = (clipped - out.to(clipped.dtype)).masked_fill(~active, float("-inf"))
            # First pass: one slot by largest remainder / stable head index.
            take = min(delta, int(active.sum().item()))
            order = torch.argsort(residual, descending=True, stable=True)
            chosen = order[:take]
            out[chosen] += 1
            delta -= take
            if delta > 0:
                capacity = upper - out
                active = capacity > 0
                # Bulk-fill evenly; this branch is reached only after strong clipping.
                n_active = int(active.sum().item())
                if n_active == 0:
                    continue
                bulk = max(1, delta // n_active)
                add = torch.minimum(capacity, torch.full_like(capacity, bulk))
                add = add * active.to(add.dtype)
                added = min(delta, int(add.sum().item()))
                if added > 0:
                    # If the final bulk vector overshoots, trim deterministically.
                    csum = torch.cumsum(add, dim=0)
                    prev = csum - add
                    limited = (torch.clamp(torch.tensor(added, device=out.device) - prev, min=0))
                    limited = torch.minimum(limited.to(add.dtype), add)
                    out += limited
                    delta -= int(limited.sum().item())
        else:
            removable = out - lower
            active = removable > 0
            if not bool(active.any()):
                raise RuntimeError("Unable to satisfy Ada-KV total budget (no downward capacity)")
            excess = (out.to(clipped.dtype) - clipped).masked_fill(~active, float("-inf"))
            take = min(-delta, int(active.sum().item()))
            order = torch.argsort(excess, descending=True, stable=True)
            chosen = order[:take]
            out[chosen] -= 1
            delta += take
            if delta < 0:
                removable = out - lower
                active = removable > 0
                n_active = int(active.sum().item())
                if n_active == 0:
                    continue
                bulk = max(1, (-delta) // n_active)
                sub = torch.minimum(removable, torch.full_like(removable, bulk))
                sub = sub * active.to(sub.dtype)
                removed = min(-delta, int(sub.sum().item()))
                if removed > 0:
                    csum = torch.cumsum(sub, dim=0)
                    prev = csum - sub
                    limited = torch.clamp(torch.tensor(removed, device=out.device) - prev, min=0)
                    limited = torch.minimum(limited.to(sub.dtype), sub)
                    out -= limited
                    delta += int(limited.sum().item())

    if int(out.sum().item()) != target_sum:
        raise RuntimeError("Ada-KV budget correction failed to reach exact total")
    return out


def allocate_adakv_budgets(
    importance: torch.Tensor,
    valid_mask: torch.Tensor,
    total_budget: int,
    floor_alpha: float,
    min_budget_per_head: int,
) -> torch.Tensor:
    """Ada-KV Algorithm 1 allocation plus official floor safeguard.

    Args:
        importance: [B,Hkv,L] token importance (higher is better).
        valid_mask: same shape; invalid/padded entries never receive budget.
        total_budget: fixed total number of mutable KV entries across heads.
        floor_alpha: uniform floor ratio.  The official implementation computes
            raw_count * (1-floor_alpha) + floor_alpha * (B/H).
        min_budget_per_head: explicit lower bound requested by this integration.
    """
    if importance.shape != valid_mask.shape or importance.dim() != 3:
        raise ValueError("importance and valid_mask must share shape [B,H,L]")
    if importance.shape[0] != 1:
        raise ValueError("Current EgoStream Ada-KV integration supports batch size 1")

    _, num_heads, seq_len = importance.shape
    valid_counts = valid_mask.sum(dim=-1).to(torch.long)[0]
    feasible_total = int(valid_counts.sum().item())
    total_budget = int(total_budget)
    if total_budget > feasible_total:
        raise ValueError(
            f"Ada-KV requested {total_budget} mutable entries but only "
            f"{feasible_total} valid entries are available"
        )

    lower = torch.minimum(
        valid_counts,
        torch.full_like(valid_counts, int(min_budget_per_head)),
    )
    if total_budget < int(lower.sum().item()):
        raise ValueError(
            f"Ada-KV total budget {total_budget} is smaller than protected/minimum "
            f"allocation {int(lower.sum().item())}"
        )

    masked = importance[0].masked_fill(~valid_mask[0], float("-inf"))
    flat = masked.reshape(-1)
    top = torch.topk(flat, k=total_budget, largest=True, sorted=False).indices
    head_ids = torch.div(top, seq_len, rounding_mode="floor")
    raw_counts = torch.bincount(head_ids, minlength=num_heads).to(importance.dtype)

    uniform = float(total_budget) / float(num_heads)
    desired = raw_counts * (1.0 - float(floor_alpha)) + uniform * float(floor_alpha)
    budgets = _deterministic_integer_projection(
        desired=desired,
        lower=lower,
        upper=valid_counts,
        target_sum=total_budget,
    )
    return budgets.unsqueeze(0)


def select_kv_per_head(
    mutable_k: torch.Tensor,
    mutable_v: torch.Tensor,
    importance: torch.Tensor,
    valid_mask: torch.Tensor,
    budgets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select independent chronological top-k entries for every KV head."""
    if mutable_k.shape != mutable_v.shape:
        raise ValueError("K and V shapes must match")
    if mutable_k.dim() != 4 or importance.shape != valid_mask.shape:
        raise ValueError("Expected K/V [B,H,L,D] and scores/mask [B,H,L]")
    if mutable_k.shape[:3] != importance.shape:
        raise ValueError("K/V and importance dimensions are inconsistent")
    if budgets.shape != importance.shape[:2]:
        raise ValueError("budgets must have shape [B,H]")

    max_budget = int(budgets.max().item()) if budgets.numel() else 0
    if max_budget <= 0:
        shape = (*mutable_k.shape[:2], 0, mutable_k.shape[-1])
        empty_k = mutable_k.new_empty(shape)
        empty_v = mutable_v.new_empty(shape)
        empty_mask = torch.empty((*mutable_k.shape[:2], 0), device=mutable_k.device, dtype=torch.bool)
        empty_idx = torch.empty((*mutable_k.shape[:2], 0), device=mutable_k.device, dtype=torch.long)
        return empty_k, empty_v, empty_mask, empty_idx

    seq_len = mutable_k.shape[-2]
    masked = importance.masked_fill(~valid_mask, float("-inf"))
    ranked = torch.argsort(masked, dim=-1, descending=True, stable=True)[..., :max_budget]
    rank_ids = torch.arange(max_budget, device=mutable_k.device).view(1, 1, -1)
    selected_valid = rank_ids < budgets.unsqueeze(-1)

    # Chronological order, with padded positions sorted after valid indices.
    sentinel = torch.full_like(ranked, seq_len)
    chronological = torch.sort(torch.where(selected_valid, ranked, sentinel), dim=-1, stable=True).values
    selected_valid = chronological < seq_len
    safe_idx = chronological.clamp_max(max(seq_len - 1, 0))
    gather_idx = safe_idx.unsqueeze(-1).expand(-1, -1, -1, mutable_k.shape[-1])
    out_k = mutable_k.gather(dim=2, index=gather_idx)
    out_v = mutable_v.gather(dim=2, index=gather_idx)
    out_k = out_k.masked_fill(~selected_valid.unsqueeze(-1), 0)
    out_v = out_v.masked_fill(~selected_valid.unsqueeze(-1), 0)
    return out_k, out_v, selected_valid, chronological


def compute_adakv_importance(
    query_states: torch.Tensor,
    full_keys: torch.Tensor,
    full_valid_mask: torch.Tensor,
    mutable_start: int,
    mutable_end: int,
    pooling: str,
    kernel_size: int,
    gqa_reduction: str,
    observation_window: int,
) -> torch.Tensor:
    """Compute causal observation-window attention importance for mutable KV.

    Query and key states are expected after RoPE.  Softmax normalization is over
    every causally visible, valid cache entry; only the mutable slice is returned.
    """
    if query_states.dim() != 4 or full_keys.dim() != 4:
        raise ValueError("query_states/full_keys must be [B,H,L,D]")
    if full_valid_mask.shape != full_keys.shape[:3]:
        raise ValueError("full_valid_mask must match full_keys [B,Hkv,L]")
    if query_states.shape[0] != full_keys.shape[0] or query_states.shape[-1] != full_keys.shape[-1]:
        raise ValueError("Query/key batch or head dimension mismatch")

    batch, num_q_heads, q_len, head_dim = query_states.shape
    num_kv_heads = full_keys.shape[1]
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"GQA mapping inconsistent: Hq={num_q_heads}, Hkv={num_kv_heads}"
        )
    groups = num_q_heads // num_kv_heads
    q_keep = min(int(observation_window), int(q_len))
    q = query_states[..., -q_keep:, :]
    k = full_keys[:, :, None, :, :].expand(
        batch, num_kv_heads, groups, full_keys.shape[-2], head_dim
    ).reshape(batch, num_q_heads, full_keys.shape[-2], head_dim)
    valid = full_valid_mask[:, :, None, :].expand(
        batch, num_kv_heads, groups, full_keys.shape[-2]
    ).reshape(batch, num_q_heads, full_keys.shape[-2])

    logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(head_dim))
    logits = logits.masked_fill(~valid.unsqueeze(-2), float("-inf"))

    # Bottom-right causal alignment: the observation queries correspond to the
    # final q_keep physical key positions after visual-only finalization.
    kv_len = full_keys.shape[-2]
    q_abs = torch.arange(kv_len - q_keep, kv_len, device=full_keys.device)
    k_abs = torch.arange(kv_len, device=full_keys.device)
    causal = k_abs.view(1, -1) <= q_abs.view(-1, 1)
    logits = logits.masked_fill(~causal.view(1, 1, q_keep, kv_len), float("-inf"))

    attn = torch.softmax(logits, dim=-1, dtype=torch.float32).to(query_states.dtype)
    mutable_attn = attn[..., int(mutable_start):int(mutable_end)]
    query_head_scores = mutable_attn.mean(dim=-2)
    kv_scores = aggregate_query_heads_to_kv_heads(
        query_head_scores, num_kv_heads=num_kv_heads, reduction=gqa_reduction
    )

    if pooling == "maxpool":
        kv_scores = F.max_pool1d(
            kv_scores, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )
    elif pooling == "avgpool":
        kv_scores = F.avg_pool1d(
            kv_scores, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )
    else:
        raise ValueError(f"Unsupported Ada-KV pooling: {pooling}")

    mutable_valid = full_valid_mask[..., int(mutable_start):int(mutable_end)]
    return kv_scores.masked_fill(~mutable_valid, float("-inf"))


class AdaKVReducer:
    """Per-layer runtime state for exact padded Ada-KV."""

    def __init__(self, config: AdaKVConfig, mutable_token_size: int, layer_idx: int):
        self.config = config
        self.mutable_token_size = int(mutable_token_size)
        self.layer_idx = int(layer_idx)
        self.config.validate(self.mutable_token_size)

        self.valid_mask: Optional[torch.Tensor] = None
        self.last_query_states: Optional[torch.Tensor] = None
        self.last_selected_indices: Optional[torch.Tensor] = None
        self.last_budgets: Optional[torch.Tensor] = None
        self.num_kv_heads: Optional[int] = None
        self.num_query_heads: Optional[int] = None
        self.reduction_count = 0
        self.reduction_time_total_sec = 0.0
        self.retained_mutable_entries = 0
        self.last_physical_mutable_length = 0

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def append(
        self,
        key_states: torch.Tensor,
        query_states: Optional[torch.Tensor],
    ) -> None:
        if not self.enabled:
            return
        if key_states.dim() != 4:
            raise ValueError(f"Ada-KV expects K [B,Hkv,L,D], got {tuple(key_states.shape)}")
        bsz, hkv, new_len, _ = key_states.shape
        self.num_kv_heads = int(hkv)
        if query_states is None:
            raise RuntimeError(
                "Ada-KV attention statistics are unavailable. Ensure "
                "configure_adakv_attention(model, cache_cfg) is called after model loading."
            )
        if query_states.dim() != 4 or query_states.shape[0] != bsz or query_states.shape[-2] != new_len:
            raise ValueError(
                f"Ada-KV query shape {tuple(query_states.shape)} is inconsistent with "
                f"new K shape {tuple(key_states.shape)}"
            )
        self.num_query_heads = int(query_states.shape[1])
        if self.num_query_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"Invalid GQA mapping Hq={self.num_query_heads}, Hkv={self.num_kv_heads}"
            )
        new_valid = torch.ones((bsz, hkv, new_len), device=key_states.device, dtype=torch.bool)
        if self.valid_mask is None:
            self.valid_mask = new_valid
        else:
            if self.valid_mask.shape[:2] != new_valid.shape[:2]:
                raise ValueError("Ada-KV cache head shape changed across updates")
            self.valid_mask = torch.cat([self.valid_mask.to(key_states.device), new_valid], dim=-1)
        self.last_query_states = query_states.detach()

    def finalize_visual_append(
        self,
        prefix_len: int,
        keep_indices: torch.Tensor,
        appended_seq_len: int,
    ) -> None:
        if not self.enabled or self.valid_mask is None:
            return
        prefix_len = int(prefix_len)
        appended_seq_len = int(appended_seq_len)
        keep_indices = keep_indices.to(self.valid_mask.device, dtype=torch.long)
        suffix = self.valid_mask[..., prefix_len:prefix_len + appended_seq_len]
        kept_suffix = suffix.index_select(-1, keep_indices)
        self.valid_mask = torch.cat([self.valid_mask[..., :prefix_len], kept_suffix], dim=-1)

        if self.last_query_states is None:
            raise RuntimeError("Ada-KV lost the current observation query states")
        if self.last_query_states.shape[-2] != appended_seq_len:
            raise ValueError(
                f"Ada-KV query length {self.last_query_states.shape[-2]} does not match "
                f"appended sequence length {appended_seq_len}"
            )
        self.last_query_states = self.last_query_states.index_select(-2, keep_indices)

    def needs_reduction(self, sink_size: int, hat_size: int) -> bool:
        if not self.enabled or self.valid_mask is None or self.num_kv_heads is None:
            return False
        start = int(sink_size)
        end = self.valid_mask.shape[-1] - int(hat_size) if int(hat_size) > 0 else self.valid_mask.shape[-1]
        mutable_valid = int(self.valid_mask[..., start:end].sum().item())
        target = self.config.average_budget(self.mutable_token_size) * self.num_kv_heads
        return mutable_valid > target

    def reduce(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        sink_size: int,
        hat_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.needs_reduction(sink_size=sink_size, hat_size=hat_size):
            return keys, values
        if self.last_query_states is None or self.valid_mask is None:
            raise RuntimeError("Ada-KV reduction requires current causal query states and a validity mask")
        if self.valid_mask.shape != keys.shape[:3]:
            raise ValueError(
                f"Ada-KV validity mask {tuple(self.valid_mask.shape)} does not match "
                f"cache {tuple(keys.shape[:3])}"
            )

        start_time = time.perf_counter()
        sink_size = int(sink_size)
        hat_size = int(hat_size)
        mutable_end = keys.shape[-2] - hat_size if hat_size > 0 else keys.shape[-2]
        mutable_start = sink_size

        importance = compute_adakv_importance(
            query_states=self.last_query_states,
            full_keys=keys,
            full_valid_mask=self.valid_mask,
            mutable_start=mutable_start,
            mutable_end=mutable_end,
            pooling=self.config.pooling,
            kernel_size=self.config.kernel_size,
            gqa_reduction=self.config.gqa_reduction,
            observation_window=self.config.observation_window,
        )
        mutable_k = keys[..., mutable_start:mutable_end, :]
        mutable_v = values[..., mutable_start:mutable_end, :]
        mutable_valid = self.valid_mask[..., mutable_start:mutable_end]
        target_total = self.config.average_budget(self.mutable_token_size) * keys.shape[1]
        feasible_total = int(mutable_valid.sum().item())
        target_total = min(int(target_total), feasible_total)

        budgets = allocate_adakv_budgets(
            importance=importance,
            valid_mask=mutable_valid,
            total_budget=target_total,
            floor_alpha=self.config.floor_alpha,
            min_budget_per_head=self.config.min_budget_per_kv_head,
        )
        selected_k, selected_v, selected_valid, selected_indices = select_kv_per_head(
            mutable_k=mutable_k,
            mutable_v=mutable_v,
            importance=importance,
            valid_mask=mutable_valid,
            budgets=budgets,
        )

        sink_k = keys[..., :sink_size, :] if sink_size > 0 else keys[..., :0, :]
        sink_v = values[..., :sink_size, :] if sink_size > 0 else values[..., :0, :]
        sink_valid = self.valid_mask[..., :sink_size] if sink_size > 0 else self.valid_mask[..., :0]
        hat_k = keys[..., mutable_end:, :] if hat_size > 0 else keys[..., :0, :]
        hat_v = values[..., mutable_end:, :] if hat_size > 0 else values[..., :0, :]
        hat_valid = self.valid_mask[..., mutable_end:] if hat_size > 0 else self.valid_mask[..., :0]

        keys = torch.cat([sink_k, selected_k, hat_k], dim=-2)
        values = torch.cat([sink_v, selected_v, hat_v], dim=-2)
        self.valid_mask = torch.cat([sink_valid, selected_valid, hat_valid], dim=-1)
        self.last_selected_indices = selected_indices.detach()
        self.last_budgets = budgets.detach()
        self.reduction_count += 1
        self.reduction_time_total_sec += time.perf_counter() - start_time
        self.retained_mutable_entries = int(selected_valid.sum().item())
        self.last_physical_mutable_length = int(selected_valid.shape[-1])
        return keys, values

    def get_stats(self) -> Dict[str, Any]:
        budgets = self.last_budgets
        if budgets is not None and budgets.numel() > 0:
            b = budgets.to(torch.float32)
            budget_summary = {
                "mean": float(b.mean().item()),
                "min": int(b.min().item()),
                "max": int(b.max().item()),
                "sum": int(b.sum().item()),
            }
        else:
            budget_summary = {"mean": 0.0, "min": 0, "max": 0, "sum": 0}
        avg_time = (
            self.reduction_time_total_sec / self.reduction_count
            if self.reduction_count > 0 else 0.0
        )
        return {
            "enabled": self.enabled,
            "implementation_mode": self.config.compatibility_mode,
            "attention_backend": self.config.attention_backend,
            "average_mutable_budget_per_kv_head": self.config.average_budget(self.mutable_token_size),
            "num_kv_heads": int(self.num_kv_heads or 0),
            "num_query_heads": int(self.num_query_heads or 0),
            "gqa_reduction": self.config.gqa_reduction,
            "head_budget": budget_summary,
            "retained_mutable_entries": int(self.retained_mutable_entries),
            "physical_mutable_length": int(self.last_physical_mutable_length),
            "reduction_count": int(self.reduction_count),
            "average_reduction_time_sec": float(avg_time),
            "fallback": None,
        }
