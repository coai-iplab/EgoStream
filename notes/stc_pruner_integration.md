# STC-Pruner integration for EgoStream

This patch integrates the **STC-Pruner** component from *Accelerating Streaming
Video Large Language Models via Hierarchical Token Compression*
(arXiv:2512.00891v1) into the existing Qwen3-VL EgoStream streaming pipeline.
It does **not** implement STC-Cacher.

## Architectural placement

STC-Pruner is not a KV-cache eviction method. Its correct placement is:

```text
frames
  -> Qwen3-VL vision encoder
  -> multimodal projector
  -> STC dual-anchor visual-token selection
  -> placeholder expansion / LLM prefill
  -> FlexibleCache KV update
  -> existing FIFO / pruning / offload lifecycle
```

The pruner therefore lives in `src/stc_pruner.py` and is invoked from
`src/model_operations.py`. Existing reducers under `Caches/` are not reused for
STC scoring and are unchanged except for a new disabled-by-default
`frame_aware_ingestion` flag.

## Implemented algorithm

For each incoming frame with projected visual tokens `Z_t`:

1. Compute the Spatial Context Anchor (SCA) as the mean of all current-frame
   token vectors.
2. Compute the Temporal Context Anchor (TCA) as the mean of SCA vectors stored
   from strictly previous frames.
3. Score each token with the paper equation:

   ```text
   score(z) = alpha * cosine_distance(z, TCA)
              + (1-alpha) * cosine_distance(z, SCA)
   ```

4. Retain the highest-scoring
   `floor(N * (1 - pruning_ratio))` tokens independently for every frame.
5. Sort selected indices to preserve their original spatial/token order.
6. Apply the same global indices to projected embeddings and every Qwen3-VL
   deep-stack visual embedding tensor.
7. Append the current SCA to history only after selection. No future frame or
   question is visible to the method.

The main experiment uses:

```json
{
  "pruning_ratio": 0.75,
  "alpha": 0.5,
  "history_window": 0
}
```

Thus, a normal Qwen3-VL frame with 448 projected tokens retains 112. A
`history_window` of `0` means all strictly previous frame anchors are used. The
paper introduces a finite `W` but does not report a numeric value; this option is
explicit so a finite ablation can be run without changing code.

### First-frame initialization

The paper pseudocode assumes a temporal history but does not specify the empty
buffer case. The implementation uses the current SCA as the first-frame TCA.
This makes the first frame use spatial novelty only, remains causal, and keeps a
fixed token budget. The behavior is configurable as
`first_frame_temporal_anchor: spatial|zeros`.

### Paper wording discrepancy

The prose calls the combination “multiplicative”, but Equation 8 and Algorithm 2
use a weighted sum. This patch follows the equation and pseudocode.

## Backward compatibility

All new options default to disabled:

```json
"stc_pruner": {"enabled": false}
```

When disabled:

- no pruner object is constructed;
- projected embeddings and deep-stack tensors are untouched;
- the original `get_rope_index` call is executed exactly as before;
- no STC fields are added to result metrics;
- `frame_aware_ingestion` remains false;
- every existing pruning, merging, FIFO, sink, hat, offload, retrieval, resume,
  result naming, and scheduling path remains selected by its original flags.

`frame_aware_ingestion` is a new metadata-only cache flag. STC enables it so the
existing visual-only finalization path is used and frame-based hat size is
computed from the reduced per-frame lengths. It does not enable an intra/inter
scorer or merge reducer.

## Files modified

- `src/model_operations.py`
  - constructs a per-video STC pruner only when enabled;
  - prunes projected and deep-stack visual embeddings before placeholder
    expansion;
  - updates visual-token counts and frame metadata;
  - skips only the unused dense-grid RoPE helper call in the STC path.
- `Caches/CacheFactory.py`
  - parses the disabled-by-default `frame_aware_ingestion` flag.
- `Caches/FlexibleCacheLayer.py`
  - includes `frame_aware_ingestion` in the existing frame bookkeeping gate.
- `run_benchmark_qwen3vl.py`
  - records STC metrics only when enabled;
  - adds optional `--annotation_id` and `--max_tasks` smoke-test filters.

## Files created

- `src/stc_pruner.py`
- `EgoStream/Configs/12_Q3VL8B_stc_pruner.json`
- `scripts/create_stc_config_from_baseline.py`
- `scripts/Egostream_cineca_stc_pruner.sh`
- `tests/test_stc_pruner.py`
- `tests/smoke_stc_pruner.py`
- `STC_PRUNER_INTEGRATION.md`

## Validation performed

```text
12 unit tests passed
Python compilation passed
Slurm/bash syntax passed
JSON validation passed
Synthetic 448-token Qwen frame: 448 -> 112 tokens passed
Patch apply check passed
```

The tests cover:

- default-disabled and exact no-op behavior;
- Equation 8 score equivalence;
- strict-past causality of TCA;
- exact per-frame integer budget;
- independent frame selection;
- chronological/spatial ordering;
- identical indices for projected and deep-stack embeddings;
- variable frame lengths;
- bounded and unbounded history;
- deterministic first-frame initialization;
- invalid configuration errors;
- baseline-config preservation by the generator;
- token and timing metrics.

## Generate the runtime config

Use the actual comparable pseudo-attention configuration as the source of truth:

```bash
python scripts/create_stc_config_from_baseline.py \
  EgoStream/Configs/12_Q3VL8B_sim_prune_intra_pseudo_att_inter_pseudo_att.json \
  EgoStream/Configs/12_Q3VL8B_stc_pruner.json
```

The generator preserves model, paths, FPS, resolution, stride, prompt,
generation settings, mutable budget, sink and hat settings. It disables the old
compression policy and enables only STC-Pruner plus frame-aware ingestion.

## Unit and synthetic smoke tests

```bash
pytest -q tests/test_stc_pruner.py
python tests/smoke_stc_pruner.py
```

## One-example real-model smoke test

```bash
python run_benchmark_qwen3vl.py \
  --config EgoStream/Configs/12_Q3VL8B_stc_pruner.json \
  --rank 0 \
  --world_size 1 \
  --annotation_id 1 \
  --max_tasks 1
```

Expected output directory:

```text
Inference/12_Q3VL8B_stc_pruner/
```

Verify that the result contains `metrics.stc_pruner`, that the answer is valid,
and that no placeholder, attention-mask, cache-position, or frame-boundary error
is raised.

## Full CINECA inference

```bash
sbatch \
  --export=ALL,BASE_CONFIG="$PWD/EgoStream/Configs/12_Q3VL8B_sim_prune_intra_pseudo_att_inter_pseudo_att.json" \
  scripts/Egostream_cineca_stc_pruner.sh
```

## Logged metrics

When STC is active, `get_cache_metrics()` adds:

- configured pruning ratio and alpha;
- history window and current history size;
- number of batches and frames;
- input, retained, and pruned visual tokens;
- actual retained/pruned ratios;
- total and average pruner host-dispatch time;
- retained token counts for the latest frame batch.

## Compatibility and limitations

1. **Only STC-Pruner is implemented.** STC-Cacher requires invasive changes
   inside Qwen3-VL’s vision transformer and is outside this patch.
2. Scoring is performed on Qwen3-VL projected visual embeddings, immediately
   before LLM prefill. This is the model-appropriate analogue of the official
   ReKV integration.
3. Qwen3-VL deep-stack tensors must expose the same visual-token axis length as
   the projected embeddings. Unsupported shapes fail loudly.
4. The original `image_grid_thw` describes dense vision tokens and cannot be
   passed to Qwen’s dense-grid RoPE helper after pruning. In this ingestion path
   that helper’s output was already unused; the STC path continues to provide
   explicit monotonic `position_ids` to the language model.
5. STC can technically precede another KV policy, but the principal config and
   generator intentionally disable other compression methods to make the
   reviewer-facing comparison controlled and interpretable.
6. Pruning time is measured with host wall time and does not force a CUDA
   synchronization inside the per-frame loop. End-to-end ingestion timing remains
   the authoritative runtime metric.

## Real-model validation still required on CINECA

The local validation environment does not contain the Qwen3-VL checkpoint,
EgoStream videos, `transformers`, or a GPU. The one-example command above must be
run before launching the full array. Do not claim an end-to-end model smoke test
until its JSON output and log have been inspected.
