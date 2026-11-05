# Configuration guide

A run is fully described by one JSON file: which model, how the video is streamed, how
much memory the model may keep and which memory policy manages it. This page explains the
file section by section and gives one recipe per policy of the paper. The exhaustive
field-by-field list is in [`REFERENCE.md`](REFERENCE.md).

Two kinds of configs live here:

| Folder | Used by | Paths inside the file are relative to |
|---|---|---|
| `qwen3vl_8b/`, `internvl3_5_8b/`, `molmo2_8b/` | `run_benchmark.py` | the repository root (run from there) |
| `single_video/` | `run_single_video.py` | the config file itself |

## 1. Anatomy of a benchmark config

```jsonc
{
  "paths": {
    "annotations_file": "data/annotations/egostream_annotations.json",  // benchmark questions
    "clip_base_path": "data/videos",     // root of the (trimmed) videos, see data/README.md
    "output_dir": "outputs",             // predictions go to outputs/<config name>/part_<rank>.json
    "prompt_path": "prompts/ingestion_prompt.txt"   // system prompt used while streaming
  },
  "model": {
    "model_id": "Qwen/Qwen3-VL-8B-Instruct",   // picks the backbone, see section 2
    "min_pixels": 65536,                       // processor limits; frames are resized anyway
    "max_pixels": 1638400
  },
  "video_processing": {
    "fps_to_process": 1,                 // frames sampled per second of video
    "target_resolution": [896, 512],     // every frame is resized to this before encoding
    "use_sliding_window": true,          // always true: frames enter the cache in windows
    "dimention_sw": 16,                  // window size in frames (historical key name)
    "stride": 16,                        // 16/16 = non-overlapping windows
    "overlapping_sw": false
  },
  "generation": {
    "max_new_tokens": 4,                 // multiple choice: the model answers with the option number
    "temperature": 0.01,
    "top_p": 0.9,
    "open_ended_max_new_tokens": 48,     // second, free-form answer generated for every question
    "open_ended_temperature": 0.5,
    "open_ended_top_p": 0.95,
    "open_ended_top_k": 0,
    "open_ended_repetition_penalty": 1.02,
    "open_ended_ban_eos_steps": 2
  },
  "cache": { ... },                      // memory budget and policy, sections 3 and 4
  "verbose": false
}
```

The paper protocol (Appendix C.1) is the `video_processing` and `generation` block above:
1 FPS, 896x512, windows of 16 frames, temperature 0.01, top-p 0.9. Keep them fixed when
comparing memory policies.

## 2. Choosing the backbone

The backbone is inferred from `model.model_id`; `run_benchmark.py` and
`run_single_video.py` load the right implementation. Add `"backbone": "qwen3vl" |
"internvl" | "molmo2"` to the `model` section to override the inference (for example for a
local checkpoint whose folder name does not contain the family name).

| Backbone | `model_id` | Notes |
|---|---|---|
| Qwen3-VL (main experiments) | `Qwen/Qwen3-VL-8B-Instruct` (`-2B-Instruct` for small GPUs) | supports every policy |
| InternVL 3.5 | `OpenGVLab/InternVL3_5-8B-HF` | |
| Molmo 2 | `allenai/Molmo2-8B` | needs `cache.molmo2_image_seq_length` (576 for 896x512); loaded with `trust_remote_code` |

**Ada-KV and STC are implemented for Qwen3-VL only.** Enabling them with another
backbone stops the run with an explicit error.

## 3. Memory budget: sink, mutable, hat

The active cache is split into three regions (paper Appendix B.1 and B.2):

```jsonc
"cache": {
  "type": "FlexibleCache",
  "MAX_FRAMES_TO_PROCESS": 1000000,
  "mutable_token_size": 8192,        // B_mut: budget for historical visual tokens (the control variable)

  "use_sink_tokens": true,           // SINK: protect the priming prompt
  "sink_sys_prompt": true,           // ... the whole prompt (size_sink_tokens is ignored)
  "size_sink_tokens": 400,           // ... or a fixed number of tokens when sink_sys_prompt is false

  "use_hat_tokens": true,            // HAT: protect the most recent frames
  "hat_category": "frames",          // "frames" (hat_num_frames) or "tokens" (size_hat_tokens)
  "hat_num_frames": 2,
  "size_hat_tokens": 400
}
```

Only the mutable region is ever pruned, merged or offloaded. With the ingestion prompt and
two hat frames the sink and hat regions add about 1,612 tokens, so `mutable_token_size =
8192` is the 9,804-token budget of Table 4; `qwen3vl_8b/budgets/` contains the Table 5
sweep (1,024 to 49,152 mutable tokens).

## 4. Recipes

All snippets below are the `cache` keys that change with respect to the
FIFO + SINK + HAT base (`qwen3vl_8b/base_fifo_sink_hat.json`); everything else stays as in
section 3. Paper label on the left, config on the right.

Scorer names used by the policies:

| Paper | `index_strategy` / step `scorer` | Where it applies |
|---|---|---|
| INTRA-COS, INTER-COS | `adiacent_cosine` (alias `adjacent_cosine`) | intra and inter |
| INTRA-PA, INTER-PA | `pseudo_attn` (alias `pseudo_attention`) | intra and inter |
| INTRA-VN | `value_norm` (alias `van_l2`) | intra only, prune only |
| INTRA-CS | `sink_cross_attention` (alias `streammem_attention`) | intra only, prune only |

### 4.1 FIFO, SINK, HAT (Table 4, rows 1-2)

```jsonc
"use_sink_tokens": false, "use_hat_tokens": false, "use_similarity": false   // row 1: plain sliding window
"use_sink_tokens": true,  "use_hat_tokens": false, "use_similarity": false   // row 2: + SINK (StreamingVLM)
"use_sink_tokens": true,  "use_hat_tokens": true,  "use_similarity": false   // base of rows 6-21
```

Overflowing mutable tokens are discarded oldest-first.

### 4.2 Prune inside each frame (INTRA-*, rows 6, 7, 11, 12)

```jsonc
"use_similarity": true,
"similarity_merge": false,
"intra_frame_pruning": true,
"inter_frame_pruning": false,
"index_strategy": "adiacent_cosine",   // or pseudo_attn, value_norm, sink_cross_attention
"intra_chunk_size": 64                 // frames scored per batch (memory vs speed)
```

`sink_cross_attention` (StreamMem-style) also reads `sink_guidance_max_tokens`,
`sink_guidance_q_chunk_size` and `sink_cross_attn_reduction` (`"mean"` or `"max"`), see
`t4_r06_prune_intra_cs_streammem.json`.

### 4.3 Prune across neighbouring frames (INTER-*, rows 13-14)

```jsonc
"use_similarity": true,
"similarity_merge": false,
"intra_frame_pruning": false,
"inter_frame_pruning": true,
"index_strategy": "adiacent_cosine",   // or pseudo_attn
"inter_chunk_size": 64
```

### 4.4 Merge instead of prune (MERGE, rows 3-4)

```jsonc
"use_similarity": true,
"similarity_merge": true,              // the selected token is averaged into the kept one
"intra_frame_pruning": true,
"index_strategy": "pseudo_attn"        // or adiacent_cosine
```

Merging is implemented for intra-frame reduction only.

### 4.5 Combining steps: the explicit pipeline (rows 5, 8-10, 15-17)

When more than one reducer is needed, list the steps in order in `pruning_pipeline`; it
replaces the flag-based behaviour. Each step has a `domain` (`cleanup`, `intra`, `inter`,
`storage`), a `scorer`, a `reducer` (`drop`, `merge`, `remove_small_frames`, `offload`) and
`params`. `remove_pct` is the fraction of mutable tokens a step may remove; the actual
removal is capped by the overflow above `mutable_token_size`.

Row 10 (PRUNE INTRA-PA + INTER-COS, the StreamingTOM policy without offloading):

```jsonc
"use_similarity": true, "similarity_merge": false,
"intra_frame_pruning": true, "inter_frame_pruning": true,
"intra_chunk_size": 16, "inter_chunk_size": 16, "cleanup_min_frame_len": 2,
"pruning_pipeline": [
  {"name": "cleanup_tiny_pre",         "domain": "cleanup", "reducer": "remove_small_frames", "params": {"min_len": 2}},
  {"name": "intra_pseudo_attn",        "domain": "intra",   "scorer": "pseudo_attn",      "reducer": "drop", "params": {"chunk_size": 64, "remove_pct": 0.10}},
  {"name": "cleanup_tiny_after_intra", "domain": "cleanup", "reducer": "remove_small_frames", "params": {"min_len": 2}},
  {"name": "inter_cosine",             "domain": "inter",   "scorer": "adiacent_cosine",  "reducer": "drop", "params": {"chunk_size": 64, "remove_pct": 0.15}},
  {"name": "cleanup_tiny_after_inter", "domain": "cleanup", "reducer": "remove_small_frames", "params": {"min_len": 2}}
]
```

Variants: swap the scorers (`value_norm`, `sink_cross_attention` with `q_chunk_size`,
`reduction`, `min_keep_per_frame` in `params`) for rows 8, 9, 15, 16; use `"reducer":
"merge"` on the intra step for the merge + prune hybrid of row 5; set
`"use_hat_tokens": false` for row 17. `cleanup` steps drop frames left with fewer than
`min_len` tokens.

### 4.6 Offloading and retrieval (OFFLOAD, Q-OFF; rows 19-20)

Instead of deleting the excess, write it to disk and bring the most relevant blocks back at
question time (ReKV-style retrieval):

```jsonc
"use_similarity": false, "intra_frame_pruning": false, "inter_frame_pruning": false,
"enable_disk_offload": true,
"offload_dir": "outputs/<config name>/kv_offload",
"offload_min_tokens": 1,
"offload_empty_cache": true,
"offload_quantize": false,              // Q-OFF: true, with the three keys below
"offload_quant_bits": 8,
"offload_quant_scheme": "per_tensor",
"pruning_pipeline": [
  {"name": "offload_left_excess", "domain": "storage", "scorer": "none", "reducer": "offload",
   "params": {"scope": "left_excess"}}                       // Q-OFF: add "quantize": true, "bits": 8, "scheme": "per_tensor"
],
"retrieve": {
  "enabled": true,
  "method": "rekv",
  "top_k_tokens": 8192,                 // mutable tokens rebuilt for the question
  "block_size_frames": 1,
  "include_live_mutable": true,
  "max_offloaded_chunks": null
}
```

Quantization affects the offloaded copy only; the live cache keeps the model dtype.
Offloaded tokens are not counted in the active budget.

### 4.7 StreamingTOM (row 21)

Pruning pipeline of row 10 followed by an int8 `storage` step and retrieval: see
`t4_r21_prune_intra_pa_inter_cos_qoff_streamingtom.json` (intra `pseudo_attn` drop, inter
`adiacent_cosine` drop, `offload` with `quantize: true`, `retrieve` with
`block_size_frames: 2`).

### 4.8 Ada-KV: adaptive head budgets (ADA-HEAD, row 18), Qwen3-VL only

Ada-KV scores mutable tokens with the attention of the latest visual queries and
redistributes one global budget non-uniformly across KV heads. It replaces the similarity
reducers, offloading and retrieval, which must be off:

```jsonc
"use_similarity": false, "similarity_merge": false,
"intra_frame_pruning": false, "inter_frame_pruning": false,
"enable_disk_offload": false, "offload_quantize": false,
"retrieve": {"enabled": false},
"adakv": {
  "enabled": true,
  "total_budget": 8192,                 // average mutable budget per KV head (= mutable_token_size)
  "importance_source": "attention",
  "observation_window": 32,             // most recent visual queries used for scoring
  "kernel_size": 7,
  "pooling": "maxpool",
  "floor_alpha": 0.2,                   // guarantees every head at least alpha * B/H slots
  "min_budget_per_kv_head": 1,
  "gqa_reduction": "mean",
  "attention_backend": "sdpa",          // required
  "compatibility_mode": "exact",        // required
  "log_every_reductions": 100
}
```

Shortcut: `python scripts/create_adakv_config_from_baseline.py <baseline.json> <out.json>`
copies any config, switches the other policies off and inserts this block with
`total_budget = mutable_token_size`.

### 4.9 STC: pre-prefill token compression (Table 11), Qwen3-VL only

STC-Pruner keeps the top `(1 - pruning_ratio)` visual tokens of every incoming frame
*before* they enter the cache, scoring each token by its cosine distance from the frame
mean (spatial anchor) and from the previous frames' means (temporal anchor). The retained
tokens then follow the normal FIFO + SINK + HAT lifecycle:

```jsonc
"frame_aware_ingestion": true,          // required by STC
"use_similarity": false, "intra_frame_pruning": false, "inter_frame_pruning": false,
"pruning_pipeline": [],
"enable_disk_offload": false, "retrieve": {"enabled": false},
"stc_pruner": {
  "enabled": true,
  "implementation": "dual_anchor_paper",
  "pruning_ratio": 0.75,                // keep 25% of the tokens of each frame (112 of 448)
  "alpha": 0.5,                         // weight of the temporal vs spatial anchor
  "history_window": 0,                  // 0 = temporal anchor over all previous frames
  "min_tokens_per_frame": 1,
  "tokens_per_frame": null,
  "first_frame_temporal_anchor": "spatial",
  "score_dtype": "float32",
  "log_every_batches": 100
}
```

Shortcut: `python scripts/create_stc_config_from_baseline.py <baseline.json> <out.json>`.

### 4.10 What can be combined

| | similarity pruning / merging | offloading + retrieval | Ada-KV | STC |
|---|---|---|---|---|
| similarity pruning / merging | pipeline steps | yes (row 21) | no | not evaluated |
| offloading + retrieval | yes | | no | not evaluated |
| Ada-KV | no | no | | no |
| STC | not evaluated | not evaluated | no | |

## 5. Provided configs

| Folder | Content |
|---|---|
| `qwen3vl_8b/t4_r01` to `t4_r21` | Table 4, one file per row (`base_fifo_sink_hat` is the shared base) |
| `qwen3vl_8b/budgets/` | Table 5, row-10 policy at seven mutable budgets |
| `qwen3vl_8b/t11_fifo_sink_hat_stc` | Table 11, STC |
| `internvl3_5_8b/`, `molmo2_8b/` | Table 6 (`t6_r05` to `t6_r12`) plus `extra_*` policies run with the same protocol |
| `single_video/` | one-video presets for each backbone, Ada-KV and STC included |

The row-by-row mapping with the literature labels is in the root [`README.md`](../README.md).

## 6. Single-video configs

Same sections as above, with `paths.video_path` instead of the benchmark paths and an
extra block:

```jsonc
"single_video_test": {
  "until_second": null,                 // stop ingesting after N seconds (null = whole video)
  "question": "Describe this video in detail."
}
```

`run_single_video.py --fast` is a smoke test (4-bit loading, two frames, 32 tokens);
`--until-second`, `--window-size`, `--stride` and `--max-new-tokens` override the config
from the command line.

## 7. Output of a benchmark run

For every question and every valid recall regime the runners store the multiple-choice
answer (`model_output_raw`, used by `evaluate.py`) and an open-ended answer
(`open_ended_output_raw`, generated with the `open_ended_*` parameters) in
`outputs/<config name>/part_<rank>.json`. Both are produced in the same pass, so the
open-ended generation is part of the reported processing time.
