#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.stc_pruner import STCDualAnchorPruner, STCPrunerConfig


def main() -> None:
    torch.manual_seed(42)
    config = STCPrunerConfig(
        enabled=True,
        implementation="dual_anchor_paper",
        pruning_ratio=0.75,
        alpha=0.5,
        history_window=0,
        min_tokens_per_frame=1,
        tokens_per_frame=None,
        first_frame_temporal_anchor="spatial",
        score_dtype="float32",
        log_every_batches=0,
    )
    pruner = STCDualAnchorPruner(config)

    batch1 = torch.randn(1, 16, 32)
    deep1 = [torch.randn(1, 16, 32), torch.randn(16, 32)]
    out1 = pruner.prune(batch1, [8, 8], deep1)

    batch2 = torch.randn(1, 12, 32)
    deep2 = [torch.randn(1, 12, 32), torch.randn(12, 32)]
    out2 = pruner.prune(batch2, [6, 6], deep2)

    result = {
        "status": "ok",
        "batch1_shape": list(out1.projected_embeddings.shape),
        "batch1_counts": out1.frame_token_counts,
        "batch2_shape": list(out2.projected_embeddings.shape),
        "batch2_counts": out2.frame_token_counts,
        "history_size": len(pruner._history),
        "metrics": pruner.metrics(),
    }
    output_path = PROJECT_ROOT / "tests" / "smoke_stc_pruner_result.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
