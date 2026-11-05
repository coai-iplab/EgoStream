#!/usr/bin/env python3
"""Run the EgoStream benchmark with the backbone chosen by the config.

Reads ``model.model_id`` (or ``model.backbone``) from the JSON config, checks that
the requested memory policy is supported by that backbone and delegates to the
matching runner, forwarding every other command-line argument unchanged:

    python run_benchmark.py --config configs/qwen3vl_8b/t4_r11_prune_intra_cos.json
    python run_benchmark.py --config configs/molmo2_8b/t6_r09_prune_intra_cos.json --rank 0 --world_size 4

Runner-specific arguments (for example ``--annotation_id`` and ``--max_tasks`` of the
Qwen3-VL runner) are passed through as they are.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backbones import INTERNVL, MOLMO2, QWEN3VL, backbone_from_config, check_policy_support

RUNNERS = {
    QWEN3VL: "run_benchmark_qwen3vl",
    INTERNVL: "run_benchmark_internvl35",
    MOLMO2: "run_benchmark_molmo2",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Common runner arguments: --rank R --world_size N (shard the videos over N "
            "processes, one GPU each). See the runner modules for backbone-specific flags."
        ),
    )
    parser.add_argument("--config", required=True, type=Path, help="Run configuration (JSON).")
    args, _passthrough = parser.parse_known_args()

    if not args.config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")
    with args.config.open("r", encoding="utf-8") as f:
        config = json.load(f)

    backbone = backbone_from_config(config.get("model", {}))
    check_policy_support(backbone, config.get("cache", {}))

    runner_name = RUNNERS[backbone]
    print(f"[run_benchmark] backbone={backbone} -> {runner_name}.py", flush=True)
    runner = importlib.import_module(runner_name)
    sys.argv[0] = str(PROJECT_ROOT / f"{runner_name}.py")
    runner.main()


if __name__ == "__main__":
    main()
