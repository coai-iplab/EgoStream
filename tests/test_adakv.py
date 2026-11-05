from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Caches.adakv import (  # noqa: E402
    AdaKVConfig,
    AdaKVReducer,
    aggregate_query_heads_to_kv_heads,
    allocate_adakv_budgets,
    select_kv_per_head,
)


class AdaKVAlgorithmTests(unittest.TestCase):
    def test_budget_allocation_exact_and_monotonic(self):
        # Head 0 is clearly strongest, followed by heads 1, 2, 3.
        importance = torch.tensor(
            [[[10., 9., 8., 7., 6.],
              [5., 4., 3., 2., 1.],
              [0.9, 0.8, 0.7, 0.6, 0.5],
              [0.4, 0.3, 0.2, 0.1, 0.0]]]
        )
        valid = torch.ones_like(importance, dtype=torch.bool)
        budgets_a = allocate_adakv_budgets(
            importance, valid, total_budget=12, floor_alpha=0.2, min_budget_per_head=1
        )
        budgets_b = allocate_adakv_budgets(
            importance, valid, total_budget=12, floor_alpha=0.2, min_budget_per_head=1
        )
        self.assertTrue(torch.equal(budgets_a, budgets_b))
        self.assertEqual(int(budgets_a.sum()), 12)
        self.assertTrue(bool((budgets_a >= 1).all()))
        b = budgets_a[0].tolist()
        self.assertGreaterEqual(b[0], b[1])
        self.assertGreaterEqual(b[1], b[2])
        self.assertGreaterEqual(b[2], b[3])

    def test_budget_smaller_than_minimum_fails(self):
        importance = torch.ones((1, 4, 5))
        valid = torch.ones_like(importance, dtype=torch.bool)
        with self.assertRaises(ValueError):
            allocate_adakv_budgets(
                importance, valid, total_budget=3, floor_alpha=0.2, min_budget_per_head=1
            )

    def test_gqa_mapping(self):
        scores = torch.zeros((1, 8, 3))
        scores[:, 0:4, :] = 2.0
        scores[:, 4:8, :] = 6.0
        mapped = aggregate_query_heads_to_kv_heads(scores, num_kv_heads=2, reduction="mean")
        self.assertEqual(tuple(mapped.shape), (1, 2, 3))
        self.assertTrue(torch.allclose(mapped[:, 0], torch.full((1, 3), 2.0)))
        self.assertTrue(torch.allclose(mapped[:, 1], torch.full((1, 3), 6.0)))
        with self.assertRaises(ValueError):
            aggregate_query_heads_to_kv_heads(scores[:, :7], num_kv_heads=2)

    def test_kv_selection_is_head_specific_and_chronological(self):
        b, h, l, d = 1, 2, 6, 1
        base = torch.arange(l, dtype=torch.float32).view(1, 1, l, 1)
        k = torch.cat([base, base + 100], dim=1)
        v = k + 1000
        importance = torch.tensor([[[0., 9., 1., 8., 2., 7.], [9., 0., 8., 1., 7., 2.]]])
        valid = torch.ones((b, h, l), dtype=torch.bool)
        budgets = torch.tensor([[3, 2]])
        out_k, out_v, out_valid, indices = select_kv_per_head(
            k, v, importance, valid, budgets
        )
        self.assertEqual(indices[0, 0, :3].tolist(), [1, 3, 5])
        self.assertEqual(indices[0, 1, :2].tolist(), [0, 2])
        self.assertTrue(bool((indices[0, 0, :3][1:] >= indices[0, 0, :3][:-1]).all()))
        self.assertTrue(bool((indices[0, 1, :2][1:] >= indices[0, 1, :2][:-1]).all()))
        self.assertTrue(torch.equal(out_v[out_valid], out_k[out_valid] + 1000))
        self.assertNotEqual(indices[0, 0, 0].item(), indices[0, 1, 0].item())

    def test_reducer_preserves_sink_hat_and_total_mutable_budget(self):
        torch.manual_seed(7)
        b, hkv, hq, seq, dim = 1, 2, 4, 12, 4
        keys = torch.randn(b, hkv, seq, dim)
        values = torch.randn_like(keys)
        queries = torch.randn(b, hq, seq, dim)
        cfg = AdaKVConfig(
            enabled=True,
            total_budget=3,
            observation_window=4,
            kernel_size=3,
            pooling="maxpool",
            floor_alpha=0.2,
            min_budget_per_kv_head=1,
            gqa_reduction="mean",
            attention_backend="sdpa",
            compatibility_mode="exact",
        )
        reducer = AdaKVReducer(cfg, mutable_token_size=3, layer_idx=0)
        reducer.append(keys, queries)
        old_sink_k = keys[..., :2, :].clone()
        old_sink_v = values[..., :2, :].clone()
        old_hat_k = keys[..., -2:, :].clone()
        old_hat_v = values[..., -2:, :].clone()
        new_k, new_v = reducer.reduce(keys, values, sink_size=2, hat_size=2)
        self.assertTrue(torch.equal(new_k[..., :2, :], old_sink_k))
        self.assertTrue(torch.equal(new_v[..., :2, :], old_sink_v))
        self.assertTrue(torch.equal(new_k[..., -2:, :], old_hat_k))
        self.assertTrue(torch.equal(new_v[..., -2:, :], old_hat_v))
        mutable_mask = reducer.valid_mask[..., 2:-2]
        self.assertEqual(int(mutable_mask.sum()), hkv * 3)
        self.assertEqual(int(reducer.last_budgets.sum()), hkv * 3)
        for head in range(hkv):
            n = int(reducer.last_budgets[0, head])
            idx = reducer.last_selected_indices[0, head, :n]
            self.assertTrue(bool((idx[1:] >= idx[:-1]).all()))


class AdaKVAttentionAdapterTests(unittest.TestCase):
    def test_per_head_padding_is_masked_in_sdpa(self):
        import torch.nn as nn
        from src.adakv_attention import _adakv_text_attention_forward

        fake_mod = types.ModuleType("fake_qwen_adakv")
        fake_mod.apply_rotary_pos_emb = lambda q, k, cos, sin: (q, k)
        sys.modules[fake_mod.__name__] = fake_mod

        class FakeAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_idx = 0
                self.head_dim = 1
                self.num_key_value_groups = 2
                self.scaling = 1.0
                self.attention_dropout = 0.0
                self.q_proj = nn.Linear(2, 2, bias=False)
                self.k_proj = nn.Linear(2, 1, bias=False)
                self.v_proj = nn.Linear(2, 1, bias=False)
                self.o_proj = nn.Linear(2, 2, bias=False)
                with torch.no_grad():
                    self.q_proj.weight.copy_(torch.eye(2))
                    self.k_proj.weight.copy_(torch.tensor([[1.0, 0.0]]))
                    self.v_proj.weight.copy_(torch.tensor([[1.0, 0.0]]))
                    self.o_proj.weight.copy_(torch.eye(2))

        FakeAttention.__module__ = fake_mod.__name__

        class FakeCache:
            def __init__(self, invalid_value):
                self.k = torch.tensor([[[[1.0], [invalid_value], [3.0]]]])
                self.v = torch.tensor([[[[2.0], [invalid_value], [4.0]]]])
                self.mask = torch.tensor([[[True, False, True]]])

            def update(self, key, value, layer_idx, cache_kwargs):
                self.query = cache_kwargs["query_states"]
                return self.k, self.v

            def get_adakv_valid_mask(self, layer_idx):
                return self.mask

        attn = FakeAttention().eval()
        hidden = torch.tensor([[[1.0, 1.0]]])
        pos = (torch.ones((1, 1, 1)), torch.zeros((1, 1, 1)))
        out_a, _ = _adakv_text_attention_forward(
            attn, hidden, position_embeddings=pos, past_key_values=FakeCache(100.0)
        )
        out_b, _ = _adakv_text_attention_forward(
            attn, hidden, position_embeddings=pos, past_key_values=FakeCache(1.0e9)
        )
        self.assertTrue(torch.isfinite(out_a).all())
        self.assertTrue(torch.allclose(out_a, out_b, atol=1e-6, rtol=1e-6))

    def test_overlong_qwen_mask_is_right_aligned_to_physical_cache(self):
        from src.adakv_attention import _additive_mask_from_input

        query = torch.zeros((1, 2, 3, 4))
        overlong_2d = torch.tensor([[0, 1, 1, 0, 1]], dtype=torch.long)
        out_2d = _additive_mask_from_input(overlong_2d, query, kv_len=3)
        self.assertEqual(tuple(out_2d.shape), (1, 1, 1, 3))
        self.assertEqual(float(out_2d[..., 0].max()), 0.0)
        self.assertTrue(torch.isneginf(out_2d[..., 1]).all())
        self.assertEqual(float(out_2d[..., 2].max()), 0.0)

        overlong_4d = torch.zeros((1, 1, 5, 5), dtype=torch.float32)
        overlong_4d[..., 0] = float("-inf")
        out_4d = _additive_mask_from_input(overlong_4d, query, kv_len=3)
        self.assertEqual(tuple(out_4d.shape), (1, 1, 3, 3))
        self.assertTrue(torch.isfinite(out_4d).all())

    def test_short_qwen_mask_is_padded_to_physical_cache(self):
        from src.adakv_attention import _additive_mask_from_input

        query = torch.zeros((1, 2, 3, 4))
        short_2d = torch.tensor([[1, 0]], dtype=torch.long)
        out_2d = _additive_mask_from_input(short_2d, query, kv_len=4)
        self.assertEqual(tuple(out_2d.shape), (1, 1, 1, 4))
        self.assertEqual(float(out_2d[..., 0].max()), 0.0)
        self.assertEqual(float(out_2d[..., 1].max()), 0.0)
        self.assertTrue(torch.isneginf(out_2d[..., 3]).all())

        short_4d = torch.zeros((1, 1, 3, 2), dtype=torch.float32)
        short_4d[..., -1] = float("-inf")
        out_4d = _additive_mask_from_input(short_4d, query, kv_len=4)
        self.assertEqual(tuple(out_4d.shape), (1, 1, 3, 4))
        self.assertTrue(torch.isfinite(out_4d[..., 0]).all())
        self.assertTrue(torch.isneginf(out_4d[..., -1]).all())

    def test_disabled_config_does_not_patch_model(self):
        import torch.nn as nn
        from src.adakv_attention import configure_adakv_attention

        model = nn.Sequential(nn.Linear(2, 2))
        original = model[0].forward
        count = configure_adakv_attention(model, {"adakv": {"enabled": False}})
        self.assertEqual(count, 0)
        self.assertEqual(model[0].forward, original)


class BackwardCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The CI image used for these focused tests need not install transformers.
        if "transformers.cache_utils" not in sys.modules:
            transformers = types.ModuleType("transformers")
            cache_utils = types.ModuleType("transformers.cache_utils")

            class CacheLayerMixin:
                pass

            class Cache:
                def __init__(self, layers=None, **kwargs):
                    self.layers = list(layers or [])

            cache_utils.CacheLayerMixin = CacheLayerMixin
            cache_utils.Cache = Cache
            transformers.cache_utils = cache_utils
            sys.modules["transformers"] = transformers
            sys.modules["transformers.cache_utils"] = cache_utils

    def test_existing_config_parse_has_no_adakv_key(self):
        from Caches.CacheFactory import _parse_cache_config

        legacy = {
            "mutable_token_size": 8,
            "use_sliding_window": True,
            "use_sink_tokens": True,
            "sink_sys_prompt": True,
            "size_sink_tokens": 2,
            "use_hat_tokens": True,
            "hat_category": "tokens",
            "size_hat_tokens": 2,
            "use_similarity": False,
            "similarity_merge": False,
        }
        parsed = _parse_cache_config(legacy)
        self.assertNotIn("adakv", parsed)
        self.assertEqual(parsed["mutable_token_size"], 8)
        self.assertEqual(parsed["HAT_SIZE"], 2)
        self.assertFalse(parsed["use_similarity"])

    def test_disabled_path_matches_with_absent_or_explicit_false(self):
        from Caches.FlexibleCacheLayer import FlexibleCacheLayer

        common = dict(
            mutable_token_size=3,
            use_sliding_window=True,
            use_sink_tokens=True,
            size_sink_tokens=1,
            use_hat_tokens=True,
            HAT_SIZE=1,
            use_similarity=False,
        )
        layer_absent = FlexibleCacheLayer(**common)
        layer_false = FlexibleCacheLayer(**common, adakv={"enabled": False})
        k = torch.arange(6, dtype=torch.float32).view(1, 1, 6, 1)
        v = k + 50
        out_a = layer_absent.update(k, v)
        out_b = layer_false.update(k, v)
        self.assertFalse(layer_absent.adakv_enabled)
        self.assertFalse(layer_false.adakv_enabled)
        self.assertTrue(torch.equal(out_a[0], out_b[0]))
        self.assertTrue(torch.equal(out_a[1], out_b[1]))
        self.assertEqual(out_a[0].flatten().tolist(), [0.0, 2.0, 3.0, 4.0, 5.0])

    def test_conflicting_policy_fails_loudly(self):
        from Caches.CacheFactory import _parse_cache_config

        with self.assertRaises(ValueError):
            _parse_cache_config({
                "mutable_token_size": 8,
                "use_similarity": True,
                "adakv": {"enabled": True},
            })

    def test_config_generator_preserves_non_cache_protocol(self):
        from scripts.create_adakv_config_from_baseline import build_adakv_config

        baseline = {
            "model": {"model_id": "Qwen/Qwen3-VL-8B-Instruct"},
            "paths": {"annotations_file": "a.json", "prompt_path": "p.txt", "output_dir": "out"},
            "video_processing": {"fps_to_process": 1, "target_resolution": [896, 512], "stride": 16},
            "generation": {"temperature": 0.01},
            "cache": {
                "mutable_token_size": 8192,
                "use_sink_tokens": True,
                "hat_num_frames": 2,
                "use_similarity": True,
                "intra_frame_pruning": True,
            },
        }
        generated = build_adakv_config(baseline)
        for key in ("model", "paths", "video_processing", "generation"):
            self.assertEqual(generated[key], baseline[key])
        self.assertTrue(generated["cache"]["adakv"]["enabled"])
        self.assertEqual(generated["cache"]["adakv"]["total_budget"], 8192)
        self.assertFalse(generated["cache"]["use_similarity"])
        self.assertFalse(generated["cache"]["intra_frame_pruning"])


if __name__ == "__main__":
    unittest.main()
