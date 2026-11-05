#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def build_stc_config(baseline: dict) -> dict:
    """Clone an EgoStream config and replace only its compression policy."""
    out = copy.deepcopy(baseline)
    cache = out.setdefault("cache", {})

    # Disable pre-existing cache-reduction methods for a controlled STC-Pruner
    # experiment. Core budget, sink, hat, model and video protocol are inherited.
    cache["use_similarity"] = False
    cache["similarity_merge"] = False
    cache["similarity_keep_old"] = False
    cache["intra_frame_pruning"] = False
    cache["inter_frame_pruning"] = False
    cache["pruning_pipeline"] = []
    cache["enable_disk_offload"] = False
    cache["offload_quantize"] = False
    cache["frame_aware_ingestion"] = True

    retrieve = cache.setdefault("retrieve", {})
    retrieve["enabled"] = False

    # If the patch is applied on top of the Ada-KV integration, keep methods
    # mutually exclusive for the principal controlled experiment.
    if isinstance(cache.get("adakv"), dict):
        cache["adakv"]["enabled"] = False

    cache["stc_pruner"] = {
        "enabled": True,
        "implementation": "dual_anchor_paper",
        "pruning_ratio": 0.75,
        "alpha": 0.5,
        "history_window": 0,
        "min_tokens_per_frame": 1,
        "tokens_per_frame": None,
        "first_frame_temporal_anchor": "spatial",
        "score_dtype": "float32",
        "log_every_batches": 100,
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_config", type=Path)
    parser.add_argument("output_config", type=Path)
    args = parser.parse_args()

    with args.baseline_config.open("r", encoding="utf-8") as f:
        baseline = json.load(f)

    output = build_stc_config(baseline)
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    with args.output_config.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
