"""
Cache Factory - Creates cache instances from configuration

Supports:
- legacy flexible-cache flags
- frame-aware pruning flags
- optional explicit pruning pipeline
- optional disk offloading
- optional int8 quantized disk offloading
"""

from typing import Any, Optional, Dict, List
from .FlexibleCache import FlexibleCache
from .pruning_pipeline import PruningStep


def create_cache_from_config(
    config: Optional[Any] = None,
    cache_cfg: Optional[Dict[str, Any]] = None,
    K: Optional[int] = None,
    **kwargs
) -> FlexibleCache:
    """
    Create a FlexibleCache instance from configuration.
    """
    if cache_cfg is None:
        cache_cfg = {}

    cache_config = _parse_cache_config(cache_cfg, K)
    return FlexibleCache(config=config, **cache_config, **kwargs)


def _parse_cache_config(
    cache_cfg: Dict[str, Any],
    K: Optional[int] = None
) -> Dict[str, Any]:
    """
    Parse the cache configuration into FlexibleCacheLayer-compatible kwargs.
    """
    config: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Core capacity
    # ------------------------------------------------------------------
    if "mutable_token_size" in cache_cfg:
        config["mutable_token_size"] = int(cache_cfg["mutable_token_size"])
    elif K is not None:
        config["mutable_token_size"] = int(K)

    # ------------------------------------------------------------------
    # Sliding window control
    # ------------------------------------------------------------------
    if "use_sliding_window" in cache_cfg:
        config["use_sliding_window"] = bool(cache_cfg["use_sliding_window"])
    if "frame_aware_ingestion" in cache_cfg:
        config["frame_aware_ingestion"] = bool(cache_cfg["frame_aware_ingestion"])

    # ------------------------------------------------------------------
    # Sink tokens
    # ------------------------------------------------------------------
    if "use_sink_tokens" in cache_cfg:
        config["use_sink_tokens"] = bool(cache_cfg["use_sink_tokens"])
    if "sink_sys_prompt" in cache_cfg:
        config["sink_sys_prompt"] = bool(cache_cfg["sink_sys_prompt"])
    if "size_sink_tokens" in cache_cfg:
        config["size_sink_tokens"] = int(cache_cfg["size_sink_tokens"])

    # ------------------------------------------------------------------
    # Hat tokens
    # ------------------------------------------------------------------
    if "use_hat_tokens" in cache_cfg:
        config["use_hat_tokens"] = bool(cache_cfg["use_hat_tokens"])
    if "hat_category" in cache_cfg:
        config["hat_category"] = str(cache_cfg["hat_category"])
    if "hat_num_frames" in cache_cfg:
        config["hat_num_frames"] = int(cache_cfg["hat_num_frames"])

    # support both aliases:
    # - old config name: size_hat_tokens
    # - layer constructor name: HAT_SIZE
    if "size_hat_tokens" in cache_cfg:
        config["HAT_SIZE"] = int(cache_cfg["size_hat_tokens"])
    if "HAT_SIZE" in cache_cfg:
        config["HAT_SIZE"] = int(cache_cfg["HAT_SIZE"])

    # ------------------------------------------------------------------
    # Similarity-based reduction
    # ------------------------------------------------------------------
    if "use_similarity" in cache_cfg:
        config["use_similarity"] = bool(cache_cfg["use_similarity"])
    if "similarity_merge" in cache_cfg:
        config["similarity_merge"] = bool(cache_cfg["similarity_merge"])
    if "similarity_keep_old" in cache_cfg:
        config["similarity_keep_old"] = bool(cache_cfg["similarity_keep_old"])

    # ------------------------------------------------------------------
    # Frame-aware flags
    # ------------------------------------------------------------------
    if "intra_frame_pruning" in cache_cfg:
        config["intra_frame_pruning"] = bool(cache_cfg["intra_frame_pruning"])
    if "inter_frame_pruning" in cache_cfg:
        config["inter_frame_pruning"] = bool(cache_cfg["inter_frame_pruning"])

    # default shared strategy
    if "index_strategy" in cache_cfg:
        config["index_strategy"] = str(cache_cfg["index_strategy"])

    # ------------------------------------------------------------------
    # Pruning controls
    # ------------------------------------------------------------------
    if "intra_chunk_size" in cache_cfg:
        config["intra_chunk_size"] = int(cache_cfg["intra_chunk_size"])

    if "inter_chunk_size" in cache_cfg:
        config["inter_chunk_size"] = int(cache_cfg["inter_chunk_size"])

    if "cleanup_min_frame_len" in cache_cfg:
        config["cleanup_min_frame_len"] = int(cache_cfg["cleanup_min_frame_len"])

    if "intra_remove_pct" in cache_cfg:
        config["intra_remove_pct"] = float(cache_cfg["intra_remove_pct"])

    if "intra_min_keep_per_frame" in cache_cfg:
        config["intra_min_keep_per_frame"] = int(cache_cfg["intra_min_keep_per_frame"])

    if "sink_guidance_max_tokens" in cache_cfg:
        config["sink_guidance_max_tokens"] = int(cache_cfg["sink_guidance_max_tokens"])

    if "sink_guidance_q_chunk_size" in cache_cfg:
        config["sink_guidance_q_chunk_size"] = int(cache_cfg["sink_guidance_q_chunk_size"])

    if "sink_cross_attn_reduction" in cache_cfg:
        config["sink_cross_attn_reduction"] = str(cache_cfg["sink_cross_attn_reduction"])

    # ------------------------------------------------------------------
    # Ada-KV (strictly opt-in; absent/disabled leaves legacy behavior intact)
    # ------------------------------------------------------------------
    if "adakv" in cache_cfg and cache_cfg["adakv"] is not None:
        if not isinstance(cache_cfg["adakv"], dict):
            raise ValueError("cache.adakv must be a dictionary")
        adakv_cfg = dict(cache_cfg["adakv"])
        adakv_cfg.setdefault("enabled", False)
        config["adakv"] = adakv_cfg

        if bool(adakv_cfg.get("enabled", False)):
            if bool(cache_cfg.get("use_similarity", False)):
                raise ValueError("Ada-KV cannot be combined with legacy similarity pruning/merging")
            if bool(cache_cfg.get("intra_frame_pruning", False)) or bool(cache_cfg.get("inter_frame_pruning", False)):
                raise ValueError("Ada-KV cannot be combined with the existing frame pruning pipeline")
            if bool(cache_cfg.get("enable_disk_offload", False)):
                raise ValueError("Exact Ada-KV does not support disk offloading")
            retrieve_cfg = cache_cfg.get("retrieve", {}) or {}
            if bool(retrieve_cfg.get("enabled", False)):
                raise ValueError("Exact Ada-KV cannot be combined with the current aligned-head retriever")

    # ------------------------------------------------------------------
    # Debug flags
    # ------------------------------------------------------------------
    if "debug_cache_layout" in cache_cfg:
        config["debug_cache_layout"] = bool(cache_cfg["debug_cache_layout"])
    if "debug_tracking" in cache_cfg:
        config["debug_tracking"] = bool(cache_cfg["debug_tracking"])

    # ------------------------------------------------------------------
    # Optional explicit pruning pipeline
    # ------------------------------------------------------------------
    if "pruning_pipeline" in cache_cfg and cache_cfg["pruning_pipeline"] is not None:
        config["pruning_pipeline"] = _parse_pruning_pipeline(
            cache_cfg["pruning_pipeline"]
        )

    # ------------------------------------------------------------------
    # Disk offloading
    # ------------------------------------------------------------------
    if "enable_disk_offload" in cache_cfg:
        config["enable_disk_offload"] = bool(cache_cfg["enable_disk_offload"])

    if "offload_dir" in cache_cfg:
        config["offload_dir"] = str(cache_cfg["offload_dir"])

    if "offload_min_tokens" in cache_cfg:
        config["offload_min_tokens"] = int(cache_cfg["offload_min_tokens"])

    if "offload_empty_cache" in cache_cfg:
        config["offload_empty_cache"] = bool(cache_cfg["offload_empty_cache"])

    # ------------------------------------------------------------------
    # Quantized disk offloading
    # ------------------------------------------------------------------
    if "offload_quantize" in cache_cfg:
        config["offload_quantize"] = bool(cache_cfg["offload_quantize"])

    if "offload_quant_bits" in cache_cfg:
        config["offload_quant_bits"] = int(cache_cfg["offload_quant_bits"])

    if "offload_quant_scheme" in cache_cfg:
        config["offload_quant_scheme"] = str(cache_cfg["offload_quant_scheme"])

    return config


def _parse_pruning_pipeline(raw_pipeline: List[Dict[str, Any]]) -> List[PruningStep]:
    """
    Convert the raw pruning_pipeline config into PruningStep objects.
    """
    if not isinstance(raw_pipeline, list):
        raise ValueError("pruning_pipeline must be a list of step dictionaries")

    parsed_steps: List[PruningStep] = []

    for idx, step_cfg in enumerate(raw_pipeline):
        if not isinstance(step_cfg, dict):
            raise ValueError(
                f"pruning_pipeline[{idx}] must be a dict, got {type(step_cfg)}"
            )

        name = str(step_cfg.get("name", f"step_{idx}"))
        domain = str(step_cfg.get("domain", "")).strip().lower()
        scorer = str(step_cfg.get("scorer", "adiacent_cosine"))
        reducer = str(step_cfg.get("reducer", "drop")).strip().lower()
        enabled = bool(step_cfg.get("enabled", True))
        params = step_cfg.get("params", {})

        if domain not in {"cleanup", "intra", "inter", "storage"}:
            raise ValueError(
                f"Invalid pruning_pipeline[{idx}].domain='{domain}'. "
                f"Expected one of: cleanup, intra, inter, storage"
            )

        if not isinstance(params, dict):
            raise ValueError(
                f"pruning_pipeline[{idx}].params must be a dict, got {type(params)}"
            )

        parsed_steps.append(
            PruningStep(
                name=name,
                domain=domain,
                scorer=scorer,
                reducer=reducer,
                enabled=enabled,
                params=params,
            )
        )

    return parsed_steps
