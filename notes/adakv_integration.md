# Ada-KV integration for EgoStream

## Execution flow mapped

`Slurm script` → JSON config → `run_benchmark_qwen3vl.py` →
`load_model` + Ada-KV-only attention patch → `model.model.language_model(...)` →
`FlexibleCache.update(...)` → per-layer `FlexibleCacheLayer.update(...)` →
visual-only frame finalization → Ada-KV overflow reduction → result serialization.

The existing cache is a multi-layer dense cache. Each layer stores K/V as
`[batch, kv_heads, physical_sequence, head_dim]`, splits the physical sequence
into sink / mutable / hat regions, and delegates legacy content reduction to a
pruning pipeline. FrameState tracks visual-token counts. Layer 0 normally
produces a reduction trace that follower layers replay. Offloading and retrieval
assume token positions remain aligned across heads.

## Implemented Ada-KV algorithm

The new path is enabled only by `cache.adakv.enabled=true`.

1. The latest causal visual queries are captured after Q/K normalization and
   RoPE inside the Ada-KV-only Qwen3-VL attention adapter.
2. During visual-only finalization, the same visual indices filter K, V, query
   states, and the per-head validity mask.
3. At overflow, attention from the last `observation_window` visual queries is
   computed over the currently visible cache. Invalid padded positions and
   future positions are masked. Scores are averaged over observation queries,
   GQA query-head groups are reduced to KV heads, and the configured 1D pooling
   is applied.
4. Mutable token scores from all KV heads are flattened. A single global top-B
   operation produces raw per-head counts (Ada-KV Algorithm 1).
5. The official floor safeguard is applied:
   `desired_h = raw_h * (1-floor_alpha) + floor_alpha * (B/H)`.
   Deterministic constrained largest-remainder correction enforces integer
   budgets, explicit per-head minima, per-head capacity limits, and an exact
   final sum.
6. Each KV head independently keeps its highest-scoring mutable entries. The
   same indices gather K and V; selected indices are sorted chronologically.
7. Sink and hat K/V blocks are concatenated back unchanged.

`adakv.total_budget` is the average mutable sequence budget per KV head. The
fixed global mutable slot budget is therefore
`num_kv_heads * adakv.total_budget`, matching a dense baseline with
`mutable_token_size` positions in every KV head.

## Variable head lengths

Exact head-specific retention is represented as a dense tensor padded to the
largest current head budget plus a `[B, Hkv, L]` validity mask. A local
Ada-KV-only SDPA attention forward maps this mask from KV heads to their GQA
query-head groups and adds it to the causal mask. Padded entries therefore
receive zero attention probability. Existing model configurations are not
patched.

This is exact in retained-set and attention semantics, but padding means the
physical GPU allocation is governed by the largest head budget rather than the
sum of ragged entries. Metrics report both logical retained entries and physical
mutable length.

## Deliberate compatibility constraints

Exact Ada-KV fails loudly when combined with:

- existing similarity pruning or merging;
- existing intra/inter pruning pipelines;
- disk or quantized offloading;
- the current ReKV retriever.

Those components assume a single shared token timeline across heads. Existing
configs remain unchanged and continue to use their original dense cache and
attention backend.

## Files modified

- `Caches/CacheFactory.py`: parse and validate the optional Ada-KV block.
- `Caches/FlexibleCacheLayer.py`: isolated Ada-KV state, query/mask lifecycle,
  overflow dispatch, metrics, and incompatible-retrieval guard.
- `Caches/FlexibleCache.py`: per-layer Ada-KV dispatch, validity-mask access,
  aggregate Ada-KV metrics.
- `src/model_operations.py`: include Ada-KV in the existing visual-only
  prepare/finalize lifecycle.
- `run_benchmark_qwen3vl.py`: opt-in attention patch, conditional
  Ada-KV result metrics, and optional single-task smoke selectors.

## Files created

- `Caches/adakv.py`
- `src/adakv_attention.py`
- `EgoStream/Configs/11_Q3VL8B_adakv.json`
- `scripts/create_adakv_config_from_baseline.py`
- `scripts/Egostream_cineca_adakv.sh`
- `tests/test_adakv.py`
- `tests/smoke_adakv_cache.py`

## Validation performed locally

- Python compilation of every modified/new Python file: passed.
- 11 unit/regression tests: passed.
- Actual original-vs-modified config parse with an existing-policy dictionary:
  identical.
- Actual original-vs-modified disabled FIFO cache update: K/V and shapes
  identical.
- CPU cache smoke: passed and wrote a valid metrics JSON.

A real Qwen3-VL single-video inference was not executable in the supplied
runtime because it has no `transformers` installation, checkpoint, GPU, or
EgoStream video files. The command below is the intended cluster smoke test and
must succeed before the full job is submitted.

## Generate the exact experiment config from the comparable baseline

```bash
python scripts/create_adakv_config_from_baseline.py \
  EgoStream/Configs/10_Q3VL8B_sim_prune_intra_pseudo_attention.json \
  EgoStream/Configs/11_Q3VL8B_adakv.json
```

This copies every non-cache protocol field from the real baseline and changes
only mutually exclusive cache-management fields.

## Single-example Qwen3-VL smoke command

```bash
python run_benchmark_qwen3vl.py \
  --config EgoStream/Configs/11_Q3VL8B_adakv.json \
  --rank 0 \
  --world_size 1 \
  --annotation_id 1 \
  --max_tasks 1
```

Check that `Inference/11_Q3VL8B_adakv/part_0.json` contains one complete MCQA
and open-ended record with `metrics.adakv`.

## Full inference

```bash
sbatch --export=ALL,BASE_CONFIG="$PWD/EgoStream/Configs/10_Q3VL8B_sim_prune_intra_pseudo_attention.json" \
  scripts/Egostream_cineca_adakv.sh
```

or directly:

```bash
python run_benchmark_qwen3vl.py \
  --config EgoStream/Configs/11_Q3VL8B_adakv.json \
  --rank 0 --world_size 1
```

## Difference from the official implementation

The adaptive global allocation, floor safeguard, head-wise top-k selection, and
GQA score aggregation follow the official formulation. The official repository
stores ragged heads in a flattened layout and invokes variable-length
FlashAttention. This integration instead uses exact masked dense padding and an
Ada-KV-local SDPA path because EgoStream's cache, frame bookkeeping, retrieval,
and offload code are dense and aligned-head abstractions. The streaming trigger
is also adapted from one-shot prompt compression to EgoStream overflow events;
importance uses only the most recent causal visual observation window.
