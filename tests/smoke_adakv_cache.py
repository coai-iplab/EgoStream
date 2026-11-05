#!/usr/bin/env python3
"""Small CPU smoke test for the exact padded Ada-KV cache core."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Caches.adakv import AdaKVConfig, AdaKVReducer


def main() -> None:
    torch.manual_seed(42)
    hkv, hq, seq, dim = 4, 8, 20, 8
    keys = torch.randn(1, hkv, seq, dim)
    values = torch.randn_like(keys)
    queries = torch.randn(1, hq, seq, dim)
    reducer = AdaKVReducer(
        AdaKVConfig(
            enabled=True,
            total_budget=6,
            observation_window=4,
            kernel_size=3,
            floor_alpha=0.2,
            attention_backend="sdpa",
            compatibility_mode="exact",
        ),
        mutable_token_size=6,
        layer_idx=0,
    )
    reducer.append(keys, queries)
    out_k, out_v = reducer.reduce(keys, values, sink_size=2, hat_size=2)
    record = {
        "status": "ok",
        "key_shape": list(out_k.shape),
        "value_shape": list(out_v.shape),
        "valid_mask_shape": list(reducer.valid_mask.shape),
        "adakv": reducer.get_stats(),
    }
    output = ROOT / "tests" / "smoke_adakv_cache_result.json"
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
