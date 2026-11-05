"""Backbone selection shared by run_benchmark.py and run_single_video.py.

The backbone is inferred from ``model.model_id`` (or set explicitly with the
optional ``model.backbone`` key) so that one command line works for every
supported model family.
"""
from __future__ import annotations

QWEN3VL = "qwen3vl"
INTERNVL = "internvl"
MOLMO2 = "molmo2"
BACKBONES = (QWEN3VL, INTERNVL, MOLMO2)


def backbone_from_config(model_cfg: dict) -> str:
    """Return one of BACKBONES for the ``model`` section of a config."""
    model_cfg = model_cfg or {}
    explicit = str(model_cfg.get("backbone", "") or "").strip().lower()
    if explicit:
        if explicit not in BACKBONES:
            raise ValueError(f"model.backbone must be one of {BACKBONES}, got {explicit!r}")
        return explicit

    name = str(model_cfg.get("model_id", "") or "").lower()
    if "internvl" in name:
        return INTERNVL
    if "molmo" in name:
        return MOLMO2
    # Qwen3-VL is the default family (also for local checkpoint paths).
    return QWEN3VL


def check_policy_support(backbone: str, cache_cfg: dict) -> None:
    """Fail early when a Qwen3-VL-only policy is requested on another backbone."""
    cache_cfg = cache_cfg or {}
    adakv = bool((cache_cfg.get("adakv") or {}).get("enabled", False))
    stc = bool((cache_cfg.get("stc_pruner") or {}).get("enabled", False))
    if backbone == QWEN3VL or not (adakv or stc):
        return
    policy = "Ada-KV (cache.adakv.enabled)" if adakv else "STC (cache.stc_pruner.enabled)"
    raise ValueError(
        f"{policy} is implemented for the Qwen3-VL backbone only, but the config selects "
        f"'{backbone}'. Disable the policy or use a Qwen3-VL model_id."
    )
