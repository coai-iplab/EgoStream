from __future__ import annotations

"""Paper-faithful STC-Pruner for causal streaming visual-token compression.

STC-Pruner acts after vision encoding/projection and before LLM prefill.  It is
therefore intentionally independent from the KV-cache reducers in ``Caches/``.

Paper algorithm (arXiv:2512.00891v1, Algorithm 2):
  * SCA: mean token vector of the current frame;
  * TCA: mean of SCA vectors from strictly previous frames;
  * score: alpha * cosine_distance(token, TCA)
           + (1-alpha) * cosine_distance(token, SCA);
  * retain the highest-scoring floor(N * (1-pruning_ratio)) tokens;
  * append the current SCA to history only after selection.
"""

from collections import deque
from dataclasses import dataclass
import math
import time
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class STCPrunerConfig:
    enabled: bool = False
    implementation: str = "dual_anchor_paper"
    pruning_ratio: float = 0.75
    alpha: float = 0.5
    history_window: int = 0
    min_tokens_per_frame: int = 1
    tokens_per_frame: Optional[int] = None
    first_frame_temporal_anchor: str = "spatial"
    score_dtype: str = "float32"
    log_every_batches: int = 100

    @classmethod
    def from_cache_config(cls, cache_cfg: Optional[Dict[str, Any]]) -> "STCPrunerConfig":
        cache_cfg = cache_cfg or {}
        raw = cache_cfg.get("stc_pruner", {}) or {}

        cfg = cls(
            enabled=bool(raw.get("enabled", False)),
            implementation=str(raw.get("implementation", "dual_anchor_paper")),
            pruning_ratio=float(raw.get("pruning_ratio", 0.75)),
            alpha=float(raw.get("alpha", 0.5)),
            history_window=int(raw.get("history_window", 0)),
            min_tokens_per_frame=int(raw.get("min_tokens_per_frame", 1)),
            tokens_per_frame=(
                None
                if raw.get("tokens_per_frame", None) is None
                else int(raw["tokens_per_frame"])
            ),
            first_frame_temporal_anchor=str(
                raw.get("first_frame_temporal_anchor", "spatial")
            ),
            score_dtype=str(raw.get("score_dtype", "float32")),
            log_every_batches=int(raw.get("log_every_batches", 100)),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.implementation != "dual_anchor_paper":
            raise ValueError(
                "STC-Pruner currently supports only "
                "implementation='dual_anchor_paper'."
            )
        if not 0.0 <= self.pruning_ratio < 1.0:
            raise ValueError("stc_pruner.pruning_ratio must be in [0, 1).")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("stc_pruner.alpha must be in [0, 1].")
        if self.history_window < 0:
            raise ValueError("stc_pruner.history_window must be >= 0.")
        if self.min_tokens_per_frame < 1:
            raise ValueError("stc_pruner.min_tokens_per_frame must be >= 1.")
        if self.tokens_per_frame is not None and self.tokens_per_frame < 1:
            raise ValueError("stc_pruner.tokens_per_frame must be >= 1 when set.")
        if self.first_frame_temporal_anchor not in {"spatial", "zeros"}:
            raise ValueError(
                "stc_pruner.first_frame_temporal_anchor must be 'spatial' or 'zeros'."
            )
        if self.score_dtype not in {"float32", "input"}:
            raise ValueError("stc_pruner.score_dtype must be 'float32' or 'input'.")
        if self.log_every_batches < 0:
            raise ValueError("stc_pruner.log_every_batches must be >= 0.")


@dataclass
class STCPruneResult:
    projected_embeddings: torch.Tensor
    deepstack_visual_embeds: Any
    frame_token_counts: List[int]
    kept_indices: torch.Tensor
    input_tokens: int
    retained_tokens: int
    pruning_time_sec: float


def _visual_token_axis(tensor: torch.Tensor, expected_tokens: int) -> int:
    """Resolve the token axis for Qwen projected/deep-stack visual tensors."""
    if tensor.ndim == 2:
        if tensor.shape[0] != expected_tokens:
            raise ValueError(
                f"Expected {expected_tokens} visual tokens on axis 0, got {tuple(tensor.shape)}"
            )
        return 0

    if tensor.ndim == 3:
        candidates = [axis for axis in (0, 1) if tensor.shape[axis] == expected_tokens]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) == 0:
            raise ValueError(
                f"Cannot find visual-token axis of length {expected_tokens} in {tuple(tensor.shape)}"
            )
        # Qwen projected tensors are conventionally [B, T, D].
        return 1

    raise ValueError(
        "STC-Pruner supports projected visual tensors shaped [T,D], [1,T,D], "
        f"or [T,1,D]; got {tuple(tensor.shape)}"
    )


def _flatten_projected_tokens(tensor: torch.Tensor, expected_tokens: int) -> torch.Tensor:
    axis = _visual_token_axis(tensor, expected_tokens)
    if tensor.ndim == 2:
        return tensor
    if axis == 1:
        if tensor.shape[0] != 1:
            raise ValueError(
                "Batched STC pruning currently supports batch size 1; "
                f"got shape {tuple(tensor.shape)}"
            )
        return tensor[0]
    if tensor.shape[1] != 1:
        raise ValueError(
            "Token-first 3D visual tensor must have singleton batch axis; "
            f"got shape {tuple(tensor.shape)}"
        )
    return tensor[:, 0, :]


def _index_visual_tensor(
    tensor: Optional[torch.Tensor],
    indices: torch.Tensor,
    expected_tokens: int,
) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    axis = _visual_token_axis(tensor, expected_tokens)
    return tensor.index_select(axis, indices.to(tensor.device))


def _index_deepstack(
    deepstack_visual_embeds: Any,
    indices: torch.Tensor,
    expected_tokens: int,
) -> Any:
    if deepstack_visual_embeds is None:
        return None
    if torch.is_tensor(deepstack_visual_embeds):
        return _index_visual_tensor(deepstack_visual_embeds, indices, expected_tokens)
    if isinstance(deepstack_visual_embeds, tuple):
        return tuple(
            _index_visual_tensor(x, indices, expected_tokens)
            if torch.is_tensor(x)
            else x
            for x in deepstack_visual_embeds
        )
    if isinstance(deepstack_visual_embeds, list):
        return [
            _index_visual_tensor(x, indices, expected_tokens)
            if torch.is_tensor(x)
            else x
            for x in deepstack_visual_embeds
        ]
    raise TypeError(
        "Unsupported deepstack_visual_embeds type for STC-Pruner: "
        f"{type(deepstack_visual_embeds)}"
    )


class STCDualAnchorPruner:
    """Stateful, causal implementation of Algorithm 2 from the STC paper."""

    def __init__(self, config: STCPrunerConfig):
        config.validate()
        self.config = config
        maxlen = None if config.history_window == 0 else config.history_window
        self._history: Deque[torch.Tensor] = deque(maxlen=maxlen)

        self.total_batches = 0
        self.total_frames = 0
        self.total_input_tokens = 0
        self.total_retained_tokens = 0
        self.total_pruning_time_sec = 0.0
        self.last_kept_indices: Optional[torch.Tensor] = None
        self.last_frame_token_counts: List[int] = []

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def reset(self) -> None:
        self._history.clear()
        self.total_batches = 0
        self.total_frames = 0
        self.total_input_tokens = 0
        self.total_retained_tokens = 0
        self.total_pruning_time_sec = 0.0
        self.last_kept_indices = None
        self.last_frame_token_counts = []

    def _retained_count(self, frame_tokens: int) -> int:
        if frame_tokens <= 0:
            raise ValueError(f"frame_tokens must be positive, got {frame_tokens}")
        if self.config.tokens_per_frame is not None:
            k = int(self.config.tokens_per_frame)
        else:
            k = int(math.floor(frame_tokens * (1.0 - self.config.pruning_ratio)))
        k = max(int(self.config.min_tokens_per_frame), k)
        return min(frame_tokens, k)

    def _temporal_anchor(self, spatial_anchor: torch.Tensor) -> torch.Tensor:
        if self._history:
            return torch.stack(list(self._history), dim=0).mean(dim=0)
        if self.config.first_frame_temporal_anchor == "spatial":
            return spatial_anchor
        return torch.zeros_like(spatial_anchor)

    def score_frame(self, frame_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return paper novelty scores and the current frame SCA.

        The history is deliberately not updated here.  The caller appends the SCA
        only after top-k selection, preserving strict causality.
        """
        if frame_tokens.ndim != 2:
            raise ValueError(
                f"frame_tokens must be [tokens, channels], got {tuple(frame_tokens.shape)}"
            )
        if frame_tokens.shape[0] == 0:
            raise ValueError("Cannot score an empty visual frame.")

        score_tokens = (
            frame_tokens.float()
            if self.config.score_dtype == "float32"
            else frame_tokens
        )
        spatial_anchor = score_tokens.mean(dim=0)
        temporal_anchor = self._temporal_anchor(spatial_anchor)

        norm_tokens = F.normalize(score_tokens, dim=-1, eps=1e-12)
        norm_spatial = F.normalize(spatial_anchor, dim=-1, eps=1e-12)
        norm_temporal = F.normalize(temporal_anchor, dim=-1, eps=1e-12)

        temporal_distance = 1.0 - torch.matmul(norm_tokens, norm_temporal)
        spatial_distance = 1.0 - torch.matmul(norm_tokens, norm_spatial)
        scores = (
            self.config.alpha * temporal_distance
            + (1.0 - self.config.alpha) * spatial_distance
        )
        return scores, spatial_anchor

    def select_frame_indices(self, frame_tokens: torch.Tensor) -> torch.Tensor:
        scores, spatial_anchor = self.score_frame(frame_tokens)
        k = self._retained_count(frame_tokens.shape[0])
        local_indices = torch.topk(scores, k=k, largest=True, sorted=False).indices
        local_indices = local_indices.sort().values

        # Paper Algorithm 2: update history only after selection.
        self._history.append(spatial_anchor.detach())
        return local_indices

    def prune(
        self,
        projected_embeddings: torch.Tensor,
        frame_token_counts: Sequence[int],
        deepstack_visual_embeds: Any = None,
    ) -> STCPruneResult:
        frame_token_counts = [int(x) for x in frame_token_counts]
        if any(x <= 0 for x in frame_token_counts):
            raise ValueError(
                f"frame_token_counts must contain only positive values: {frame_token_counts}"
            )
        expected_tokens = int(sum(frame_token_counts))
        flat = _flatten_projected_tokens(projected_embeddings, expected_tokens)
        if flat.shape[0] != expected_tokens:
            raise ValueError(
                f"Projected visual tokens ({flat.shape[0]}) do not match frame metadata "
                f"({expected_tokens})."
            )

        if not self.enabled:
            identity = torch.arange(expected_tokens, device=flat.device, dtype=torch.long)
            return STCPruneResult(
                projected_embeddings=projected_embeddings,
                deepstack_visual_embeds=deepstack_visual_embeds,
                frame_token_counts=list(frame_token_counts),
                kept_indices=identity,
                input_tokens=expected_tokens,
                retained_tokens=expected_tokens,
                pruning_time_sec=0.0,
            )

        t0 = time.perf_counter()
        kept_global: List[torch.Tensor] = []
        retained_counts: List[int] = []
        cursor = 0
        for frame_len in frame_token_counts:
            frame = flat[cursor : cursor + frame_len]
            local = self.select_frame_indices(frame)
            kept_global.append(local + cursor)
            retained_counts.append(int(local.numel()))
            cursor += frame_len

        kept_indices = torch.cat(kept_global, dim=0)
        pruned_projected = _index_visual_tensor(
            projected_embeddings, kept_indices, expected_tokens
        )
        pruned_deepstack = _index_deepstack(
            deepstack_visual_embeds, kept_indices, expected_tokens
        )
        elapsed = time.perf_counter() - t0

        retained_tokens = int(kept_indices.numel())
        self.total_batches += 1
        self.total_frames += len(frame_token_counts)
        self.total_input_tokens += expected_tokens
        self.total_retained_tokens += retained_tokens
        self.total_pruning_time_sec += elapsed
        self.last_kept_indices = kept_indices.detach().cpu()
        self.last_frame_token_counts = list(retained_counts)

        return STCPruneResult(
            projected_embeddings=pruned_projected,
            deepstack_visual_embeds=pruned_deepstack,
            frame_token_counts=retained_counts,
            kept_indices=kept_indices,
            input_tokens=expected_tokens,
            retained_tokens=retained_tokens,
            pruning_time_sec=elapsed,
        )

    def should_log(self) -> bool:
        every = int(self.config.log_every_batches)
        return every > 0 and (
            self.total_batches <= 3 or self.total_batches % every == 0
        )

    def metrics(self) -> Dict[str, Any]:
        input_tokens = int(self.total_input_tokens)
        retained_tokens = int(self.total_retained_tokens)
        retained_ratio = (
            retained_tokens / input_tokens if input_tokens > 0 else 1.0
        )
        return {
            "enabled": bool(self.enabled),
            "implementation": self.config.implementation,
            "pruning_ratio_configured": float(self.config.pruning_ratio),
            "alpha": float(self.config.alpha),
            "history_window": int(self.config.history_window),
            "history_size_current": int(len(self._history)),
            "tokens_per_frame_override": self.config.tokens_per_frame,
            "min_tokens_per_frame": int(self.config.min_tokens_per_frame),
            "first_frame_temporal_anchor": self.config.first_frame_temporal_anchor,
            "score_dtype": self.config.score_dtype,
            "total_batches": int(self.total_batches),
            "total_frames": int(self.total_frames),
            "input_visual_tokens": input_tokens,
            "retained_visual_tokens": retained_tokens,
            "pruned_visual_tokens": int(input_tokens - retained_tokens),
            "retained_ratio_actual": float(retained_ratio),
            "pruned_ratio_actual": float(1.0 - retained_ratio),
            "total_pruning_time_sec": float(self.total_pruning_time_sec),
            "average_pruning_time_per_batch_sec": (
                float(self.total_pruning_time_sec / self.total_batches)
                if self.total_batches > 0
                else 0.0
            ),
            "last_frame_token_counts": list(self.last_frame_token_counts),
        }


def build_stc_pruner(cache_cfg: Optional[Dict[str, Any]]) -> Optional[STCDualAnchorPruner]:
    config = STCPrunerConfig.from_cache_config(cache_cfg)
    if not config.enabled:
        return None
    return STCDualAnchorPruner(config)


def get_stc_metrics(cache_instance: Any) -> Optional[Dict[str, Any]]:
    pruner = getattr(cache_instance, "stc_pruner", None)
    if pruner is None or not getattr(pruner, "enabled", False):
        return None
    return pruner.metrics()
