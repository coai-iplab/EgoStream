from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.stc_pruner import (  # noqa: E402
    STCDualAnchorPruner,
    STCPrunerConfig,
    build_stc_pruner,
)


def make_config(**overrides) -> STCPrunerConfig:
    raw = dict(
        enabled=True,
        implementation="dual_anchor_paper",
        pruning_ratio=0.5,
        alpha=0.5,
        history_window=0,
        min_tokens_per_frame=1,
        tokens_per_frame=None,
        first_frame_temporal_anchor="spatial",
        score_dtype="float32",
        log_every_batches=0,
    )
    raw.update(overrides)
    return STCPrunerConfig(**raw)


def test_default_config_disables_stc() -> None:
    assert STCPrunerConfig.from_cache_config({}).enabled is False
    assert build_stc_pruner({}) is None
    assert build_stc_pruner({"stc_pruner": {"enabled": False}}) is None


def test_disabled_path_is_exact_noop() -> None:
    cfg = STCPrunerConfig(enabled=False)
    pruner = STCDualAnchorPruner(cfg)
    x = torch.randn(8, 5)
    deep = [torch.randn(8, 5), torch.randn(1, 8, 5)]
    result = pruner.prune(x, [4, 4], deep)

    assert result.projected_embeddings is x
    assert result.deepstack_visual_embeds is deep
    assert result.frame_token_counts == [4, 4]
    assert torch.equal(result.kept_indices, torch.arange(8))
    assert pruner.total_batches == 0


def test_dual_anchor_score_matches_paper_equation() -> None:
    cfg = make_config(alpha=0.25, first_frame_temporal_anchor="zeros")
    pruner = STCDualAnchorPruner(cfg)
    pruner._history.append(torch.tensor([1.0, 0.0]))
    frame = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

    scores, spatial = pruner.score_frame(frame)
    temporal = torch.tensor([1.0, 0.0])
    expected = 0.25 * (
        1.0 - F.cosine_similarity(frame, temporal.expand_as(frame), dim=-1)
    ) + 0.75 * (
        1.0 - F.cosine_similarity(frame, spatial.expand_as(frame), dim=-1)
    )
    assert torch.allclose(scores, expected, atol=1e-6)


def test_current_frame_not_in_temporal_anchor_until_after_selection() -> None:
    cfg = make_config(tokens_per_frame=1, alpha=1.0)
    pruner = STCDualAnchorPruner(cfg)

    first = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    pruner.select_frame_indices(first)
    assert len(pruner._history) == 1
    assert torch.allclose(pruner._history[0], torch.tensor([1.0, 0.0]))

    second = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    scores, _ = pruner.score_frame(second)
    # With strict-past TCA=[1,0], the opposite token is more novel.
    assert scores[1] > scores[0]
    assert len(pruner._history) == 1

    selected = pruner.select_frame_indices(second)
    assert selected.tolist() == [1]
    assert len(pruner._history) == 2


def test_per_frame_budget_and_chronological_order() -> None:
    cfg = make_config(pruning_ratio=0.75)
    pruner = STCDualAnchorPruner(cfg)
    x = torch.randn(16, 7)
    result = pruner.prune(x, [8, 8])

    assert result.frame_token_counts == [2, 2]
    assert result.retained_tokens == 4
    assert result.kept_indices[:2].max() < 8
    assert result.kept_indices[2:].min() >= 8
    assert torch.all(result.kept_indices[1:] > result.kept_indices[:-1])
    assert torch.equal(result.projected_embeddings, x[result.kept_indices])


def test_same_indices_apply_to_deepstack_variants() -> None:
    cfg = make_config(tokens_per_frame=2)
    pruner = STCDualAnchorPruner(cfg)
    x = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    deep_2d = x + 1000
    deep_3d = (x + 2000).unsqueeze(0)

    result = pruner.prune(x, [3, 3], [deep_2d, deep_3d])
    idx = result.kept_indices
    assert torch.equal(result.deepstack_visual_embeds[0], deep_2d[idx])
    assert torch.equal(result.deepstack_visual_embeds[1], deep_3d[:, idx, :])


def test_variable_frame_lengths_supported() -> None:
    cfg = make_config(pruning_ratio=0.5)
    pruner = STCDualAnchorPruner(cfg)
    x = torch.randn(12, 4)
    result = pruner.prune(x, [3, 4, 5])
    assert result.frame_token_counts == [1, 2, 2]
    assert result.retained_tokens == 5


def test_history_window_is_bounded() -> None:
    cfg = make_config(tokens_per_frame=1, history_window=2)
    pruner = STCDualAnchorPruner(cfg)
    for i in range(5):
        frame = torch.full((2, 3), float(i + 1))
        pruner.select_frame_indices(frame)
    assert len(pruner._history) == 2
    assert torch.allclose(pruner._history[0], torch.full((3,), 4.0))
    assert torch.allclose(pruner._history[1], torch.full((3,), 5.0))


def test_first_frame_spatial_fallback_is_deterministic() -> None:
    cfg = make_config(tokens_per_frame=2, first_frame_temporal_anchor="spatial")
    frame = torch.tensor(
        [[1.0, 0.0], [0.8, 0.2], [-1.0, 0.0], [0.0, 1.0]]
    )
    a = STCDualAnchorPruner(cfg).select_frame_indices(frame)
    b = STCDualAnchorPruner(cfg).select_frame_indices(frame)
    assert torch.equal(a, b)


def test_invalid_config_fails_loudly() -> None:
    with pytest.raises(ValueError):
        make_config(pruning_ratio=1.0).validate()
    with pytest.raises(ValueError):
        make_config(alpha=-0.1).validate()
    with pytest.raises(ValueError):
        make_config(history_window=-1).validate()


def _load_generator_module():
    path = PROJECT_ROOT / "scripts" / "create_stc_config_from_baseline.py"
    spec = importlib.util.spec_from_file_location("create_stc_config", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_generator_preserves_protocol_and_replaces_policy() -> None:
    module = _load_generator_module()
    baseline = {
        "model": {"model_id": "Qwen/Qwen3-VL-8B-Instruct"},
        "paths": {"annotations_file": "a.json", "output_dir": "Inference"},
        "video_processing": {
            "fps_to_process": 1,
            "target_resolution": [896, 512],
            "dimention_sw": 16,
            "stride": 16,
        },
        "generation": {"max_new_tokens": 16},
        "cache": {
            "mutable_token_size": 8192,
            "use_sink_tokens": True,
            "use_hat_tokens": True,
            "hat_num_frames": 2,
            "use_similarity": True,
            "intra_frame_pruning": True,
            "pruning_pipeline": [{"name": "old"}],
        },
    }
    out = module.build_stc_config(baseline)

    for key in ("model", "paths", "video_processing", "generation"):
        assert out[key] == baseline[key]
    for key in ("mutable_token_size", "use_sink_tokens", "use_hat_tokens", "hat_num_frames"):
        assert out["cache"][key] == baseline["cache"][key]

    assert out["cache"]["use_similarity"] is False
    assert out["cache"]["intra_frame_pruning"] is False
    assert out["cache"]["pruning_pipeline"] == []
    assert out["cache"]["frame_aware_ingestion"] is True
    assert out["cache"]["stc_pruner"]["enabled"] is True
    assert out["cache"]["stc_pruner"]["pruning_ratio"] == 0.75


def test_metrics_account_exact_tokens() -> None:
    cfg = make_config(pruning_ratio=0.75)
    pruner = STCDualAnchorPruner(cfg)
    pruner.prune(torch.randn(16, 3), [8, 8])
    m = pruner.metrics()
    assert m["input_visual_tokens"] == 16
    assert m["retained_visual_tokens"] == 4
    assert m["pruned_visual_tokens"] == 12
    assert m["retained_ratio_actual"] == pytest.approx(0.25)


def test_cache_factory_new_flag_is_default_off_and_opt_in() -> None:
    import types

    package_name = "_stc_test_caches"
    package = types.ModuleType(package_name)
    package.__path__ = []
    fake_flexible = types.ModuleType(f"{package_name}.FlexibleCache")
    fake_pipeline = types.ModuleType(f"{package_name}.pruning_pipeline")

    class FakeFlexibleCache:
        pass

    class FakePruningStep:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    fake_flexible.FlexibleCache = FakeFlexibleCache
    fake_pipeline.PruningStep = FakePruningStep
    sys.modules[package_name] = package
    sys.modules[f"{package_name}.FlexibleCache"] = fake_flexible
    sys.modules[f"{package_name}.pruning_pipeline"] = fake_pipeline

    path = PROJECT_ROOT / "Caches" / "CacheFactory.py"
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.CacheFactory", path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    disabled = module._parse_cache_config({"mutable_token_size": 8192})
    enabled = module._parse_cache_config(
        {"mutable_token_size": 8192, "frame_aware_ingestion": True}
    )
    assert "frame_aware_ingestion" not in disabled
    assert enabled["frame_aware_ingestion"] is True
