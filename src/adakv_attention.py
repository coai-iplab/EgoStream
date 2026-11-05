from __future__ import annotations

"""Ada-KV-only Qwen3-VL text-attention adapter.

This module patches text attention instances only when cache.adakv.enabled=true.
Existing configurations never import or execute the patched forward path.
"""

import sys
import types
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    return (
        hidden_states[:, :, None, :, :]
        .expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
        .reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)
    )


def _additive_mask_from_input(
    attention_mask: Optional[torch.Tensor],
    query: torch.Tensor,
    kv_len: int,
) -> Optional[torch.Tensor]:
    if attention_mask is None:
        return None
    mask = attention_mask.to(query.device)
    q_len = query.shape[-2]
    if mask.dim() == 2:
        if mask.shape[-1] > kv_len:
            mask = mask[..., -kv_len:]
        elif mask.shape[-1] < kv_len:
            pad = torch.ones(
                (*mask.shape[:-1], kv_len - mask.shape[-1]),
                device=mask.device,
                dtype=mask.dtype,
            )
            mask = torch.cat([pad, mask], dim=-1)
        elif mask.shape[-1] != kv_len:
            raise ValueError(
                f"Ada-KV attention mask length {mask.shape[-1]} != KV length {kv_len}"
            )
        valid = mask.to(torch.bool)
        additive = torch.zeros(
            (mask.shape[0], 1, 1, kv_len), device=query.device, dtype=query.dtype
        )
        return additive.masked_fill(~valid[:, None, None, :], float("-inf"))
    if mask.dim() == 4:
        if mask.shape[-1] > kv_len:
            mask = mask[..., -kv_len:]
        elif mask.shape[-1] < kv_len:
            fill_value = True if mask.dtype == torch.bool else 0
            pad = torch.full(
                (*mask.shape[:-1], kv_len - mask.shape[-1]),
                fill_value=fill_value,
                device=mask.device,
                dtype=mask.dtype,
            )
            mask = torch.cat([pad, mask], dim=-1)
        elif mask.shape[-1] != kv_len:
            raise ValueError(
                f"Ada-KV 4D attention mask length {mask.shape[-1]} != KV length {kv_len}"
            )
        if mask.shape[-2] not in (1, q_len):
            if mask.shape[-2] > q_len:
                mask = mask[..., -q_len:, :]
            else:
                raise ValueError(
                    f"Ada-KV 4D attention query mask length {mask.shape[-2]} != Q length {q_len}"
                )
        if mask.dtype == torch.bool:
            additive = torch.zeros_like(mask, dtype=query.dtype)
            return additive.masked_fill(~mask, float("-inf"))
        return mask.to(dtype=query.dtype)
    raise ValueError(f"Unsupported Ada-KV attention mask rank: {mask.dim()}")


def _build_bottom_right_causal_mask(query: torch.Tensor, kv_len: int) -> torch.Tensor:
    q_len = query.shape[-2]
    q_abs = torch.arange(kv_len - q_len, kv_len, device=query.device)
    k_abs = torch.arange(kv_len, device=query.device)
    allowed = k_abs.view(1, -1) <= q_abs.view(-1, 1)
    additive = torch.zeros((1, 1, q_len, kv_len), device=query.device, dtype=query.dtype)
    return additive.masked_fill(~allowed.view(1, 1, q_len, kv_len), float("-inf"))


def _adakv_text_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings=None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values=None,
    past_key_value=None,
    **kwargs,
):
    """Generic Qwen3-VL-like text attention with exact padded-head masking."""
    cache = past_key_values if past_key_values is not None else past_key_value
    input_shape = hidden_states.shape[:-1]
    head_dim = int(getattr(self, "head_dim"))
    hidden_shape = (*input_shape, -1, head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    if hasattr(self, "q_norm"):
        query_states = self.q_norm(query_states)
    if hasattr(self, "k_norm"):
        key_states = self.k_norm(key_states)

    if position_embeddings is None:
        raise RuntimeError("Ada-KV Qwen3-VL adapter requires position_embeddings=(cos, sin)")
    cos, sin = position_embeddings
    modeling_module = sys.modules.get(self.__class__.__module__)
    apply_rope = getattr(modeling_module, "apply_rotary_pos_emb", None)
    if apply_rope is None:
        raise RuntimeError(
            f"Cannot locate apply_rotary_pos_emb in {self.__class__.__module__}"
        )
    query_states, key_states = apply_rope(query_states, key_states, cos, sin)

    if cache is not None:
        cache_kwargs = {
            "query_states": query_states,
            "cache_position": kwargs.get("cache_position", None),
            "cos": cos,
            "sin": sin,
        }
        key_states, value_states = cache.update(
            key_states, value_states, int(self.layer_idx), cache_kwargs
        )

    num_groups = int(getattr(self, "num_key_value_groups", query_states.shape[1] // key_states.shape[1]))
    if query_states.shape[1] != key_states.shape[1] * num_groups:
        raise ValueError(
            f"Ada-KV GQA mismatch Hq={query_states.shape[1]}, "
            f"Hkv={key_states.shape[1]}, groups={num_groups}"
        )
    repeated_k = _repeat_kv(key_states, num_groups)
    repeated_v = _repeat_kv(value_states, num_groups)
    kv_len = repeated_k.shape[-2]

    additive_mask = _additive_mask_from_input(attention_mask, query_states, kv_len)
    causal_mask = _build_bottom_right_causal_mask(query_states, kv_len)
    additive_mask = causal_mask if additive_mask is None else additive_mask + causal_mask

    if cache is not None and hasattr(cache, "get_adakv_valid_mask"):
        kv_valid = cache.get_adakv_valid_mask(int(self.layer_idx))
        if kv_valid is None:
            raise RuntimeError("Ada-KV is enabled but the per-head validity mask is unavailable")
        if kv_valid.shape[-1] != kv_len or kv_valid.shape[1] != key_states.shape[1]:
            raise ValueError(
                f"Ada-KV validity mask {tuple(kv_valid.shape)} incompatible with "
                f"K {tuple(key_states.shape)}"
            )
        q_valid = kv_valid[:, :, None, :].expand(
            kv_valid.shape[0], kv_valid.shape[1], num_groups, kv_len
        ).reshape(kv_valid.shape[0], query_states.shape[1], kv_len)
        head_mask = torch.zeros(
            (q_valid.shape[0], q_valid.shape[1], 1, kv_len),
            device=query_states.device,
            dtype=query_states.dtype,
        ).masked_fill(~q_valid[:, :, None, :], float("-inf"))
        additive_mask = additive_mask + head_mask

    dropout_p = 0.0 if not self.training else float(getattr(self, "attention_dropout", 0.0))
    scaling = float(getattr(self, "scaling", head_dim ** -0.5))
    try:
        attn_output = F.scaled_dot_product_attention(
            query_states,
            repeated_k,
            repeated_v,
            attn_mask=additive_mask,
            dropout_p=dropout_p,
            is_causal=False,
            scale=scaling,
        )
    except TypeError:
        # PyTorch versions before the explicit scale kwarg.
        attn_output = F.scaled_dot_product_attention(
            query_states * (scaling * (head_dim ** 0.5)),
            repeated_k,
            repeated_v,
            attn_mask=additive_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )

    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def configure_adakv_attention(model, cache_cfg: Dict[str, Any]) -> int:
    """Patch Qwen3-VL text-attention modules only for explicit exact Ada-KV."""
    adakv_cfg = dict((cache_cfg or {}).get("adakv", {}) or {})
    if not bool(adakv_cfg.get("enabled", False)):
        return 0
    if str(adakv_cfg.get("compatibility_mode", "exact")).lower() != "exact":
        raise ValueError("Only exact Ada-KV compatibility_mode is implemented")
    if str(adakv_cfg.get("attention_backend", "sdpa")).lower() != "sdpa":
        raise ValueError("Exact Ada-KV requires the local SDPA attention backend")
    if not hasattr(F, "scaled_dot_product_attention"):
        raise RuntimeError("The installed PyTorch lacks scaled_dot_product_attention")

    patched = 0
    required = ("q_proj", "k_proj", "v_proj", "o_proj", "layer_idx", "head_dim")
    for module in model.modules():
        if not all(hasattr(module, name) for name in required):
            continue
        if hasattr(module, "_adakv_original_forward"):
            patched += 1
            continue
        module._adakv_original_forward = module.forward
        module.forward = types.MethodType(_adakv_text_attention_forward, module)
        patched += 1

    if patched <= 0:
        raise RuntimeError(
            "No Qwen3-VL-compatible text attention modules were found; "
            "exact Ada-KV cannot be activated."
        )
    print(
        f"[AdaKV] enabled exact padded-head retention | backend=sdpa | "
        f"patched_text_attention_layers={patched}",
        flush=True,
    )
    return patched
