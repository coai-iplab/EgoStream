#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def build_adakv_config(base: dict) -> dict:
    cfg = copy.deepcopy(base)
    cache = cfg.setdefault("cache", {})
    if "mutable_token_size" not in cache:
        raise KeyError("Baseline config must define cache.mutable_token_size")

    # Disable only mutually exclusive cache-management mechanisms. Sink, hat,
    # budget, FPS, resolution, stride, prompt, model, paths, generation and seed
    # are inherited unchanged from the baseline.
    cache["use_similarity"] = False
    cache["similarity_merge"] = False
    cache["intra_frame_pruning"] = False
    cache["inter_frame_pruning"] = False
    cache.pop("pruning_pipeline", None)

    cache["enable_disk_offload"] = False
    cache["offload_quantize"] = False
    retrieve = dict(cache.get("retrieve", {}) or {})
    retrieve["enabled"] = False
    cache["retrieve"] = retrieve

    cache["adakv"] = {
        "enabled": True,
        # In this integration total_budget is the average mutable sequence
        # budget per KV head. The exact global slot budget is Hkv * total_budget,
        # matching a dense baseline with mutable_token_size positions per head.
        "total_budget": int(cache["mutable_token_size"]),
        "importance_source": "attention",
        "observation_window": 32,
        "kernel_size": 7,
        "pooling": "maxpool",
        "floor_alpha": 0.2,
        "min_budget_per_kv_head": 1,
        "gqa_reduction": "mean",
        "attention_backend": "sdpa",
        "compatibility_mode": "exact",
        "log_every_reductions": 100,
    }
    cfg["adakv_generated_from_baseline"] = True
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_config", type=Path)
    parser.add_argument("output_config", type=Path)
    args = parser.parse_args()

    with args.base_config.open("r", encoding="utf-8") as f:
        base = json.load(f)
    out = build_adakv_config(base)
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    with args.output_config.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(args.output_config)


if __name__ == "__main__":
    main()
