# Configuration reference (field by field)

Exhaustive description of every key accepted by the JSON configs of `run_single_video.py` and `run_benchmark.py`. For a task-oriented introduction (how to pick a backbone, how to switch memory policy, Ada-KV, STC) read [`README.md`](README.md) in this folder first.

The pipeline has five main phases:

1. Load the JSON configuration.
2. Load the Qwen-VL/Qwen3-VL model and processor.
3. Prime the KV cache with the system prompt.
4. Ingest video frames through a sliding-window sampler.
5. Ask a question using the built cache and generate the answer.

The runners expect a sliding-window setup: `cache.type` must be `"FlexibleCache"` and `video_processing.use_sliding_window` must be enabled.

---

## Minimal expected JSON structure

```json
{
  "paths": {},
  "model": {},
  "video_processing": {},
  "generation": {},
  "cache": {},
  "single_video_test": {},
  "verbose": true
}
```

---

## 1. `paths`

Controls file locations.

```json
"paths": {
  "video_path": "<PATH_TO_VIDEO_FILE>",
  "prompt_path": "<PATH_TO_PROMPT_FILE>"
}
```

### `paths.video_path`

**Type:** string  
**Required for single-video tester:** yes  
**Meaning:** Path to the input video file.

The single-video tester resolves relative paths relative to the config file location. Use forward slashes (`/`) or escaped backslashes (`\\`) in JSON.

Example:

```json
"video_path": "data/videos/example_video.mp4"
```

### `paths.prompt_path`

**Type:** string  
**Required:** yes  
**Meaning:** Path to the text prompt used to prime the cache as the system prompt.

During priming, the prompt file is read and inserted as a system message before the video stream starts.

Example:

```json
"prompt_path": "prompts/ingestion_prompt.txt"
```

### Additional path fields used by benchmark scripts

These are not required by the single-video tester, but are used by the dataset/benchmark script.

| Field | Type | Used by | Meaning |
|---|---:|---|---|
| `annotations_file` | string | benchmark script | JSON annotation file for QA tasks. |
| `clip_base_path` | string | benchmark script | Base directory where dataset videos are stored. |
| `output_dir` | string | benchmark script | Directory for generated prediction/metric files. |

---

## 2. `model`

Controls which model is loaded and, optionally, image-token resolution limits.

```json
"model": {
  "model_id": "Qwen/Qwen3-VL-8B-Instruct",
  "min_pixels": 65536,
  "max_pixels": 1638400
}
```

### `model.model_id`

**Type:** string  
**Required:** yes  
**Meaning:** Hugging Face model id or model path.

Examples:

```json
"model_id": "Qwen/Qwen3-VL-8B-Instruct"
```

```json
"model_id": "Qwen/Qwen3-VL-4B-Instruct"
```

```json
"model_id": "models/local_model_directory"
```

### `model.min_pixels`

**Type:** integer  
**Required:** depends on `load_model` implementation  
**Meaning:** Minimum image/video pixel budget passed to the model processor, when supported by the local loader.

Example:

```json
"min_pixels": 65536
```

### `model.max_pixels`

**Type:** integer  
**Required:** depends on `load_model` implementation  
**Meaning:** Maximum image/video pixel budget passed to the model processor, when supported by the local loader.

Example:

```json
"max_pixels": 1638400
```

Note: the ingestion code also explicitly resizes frames using `video_processing.target_resolution`, so `min_pixels` and `max_pixels` are processor/model-loader constraints, while `target_resolution` is the direct frame resize setting.

---

## 3. `video_processing`

Controls how the video is sampled and batched into windows.

```json
"video_processing": {
  "fps_to_process": 1,
  "target_resolution": [896, 512],
  "stride": 16,
  "use_sliding_window": true,
  "dimention_sw": 16,
  "overlapping_sw": false
}
```

### `video_processing.fps_to_process`

**Type:** integer or float  
**Default in scripts:** `1`  
**Meaning:** Target sampling rate from the video. A value of `1` means one frame per second.

Example:

```json
"fps_to_process": 1
```

### `video_processing.target_resolution`

**Type:** array of two integers: `[width, height]`  
**Default in benchmark script:** `[360, 240]`  
**Meaning:** Resolution used when resizing video frames before image processing.

Example:

```json
"target_resolution": [896, 512]
```

Higher resolution usually improves visual detail but increases visual tokens, VRAM, and runtime.

### `video_processing.use_sliding_window`

**Type:** boolean  
**Required for current single-video tester:** `true`  
**Meaning:** Enables sliding-window ingestion. The current single-video tester is written only for sliding-window mode.

Example:

```json
"use_sliding_window": true
```

### `video_processing.dimention_sw`

**Type:** integer  
**Default/fallback:** `16`  
**Meaning:** Sliding-window size in sampled frames. The key name is misspelled in the current code as `dimention_sw`; some helper code also accepts `dimension_sw` as an alias.

Example:

```json
"dimention_sw": 16
```

Note: if the value is non-positive, the tester falls back to `16`.

### `video_processing.dimension_sw`

**Type:** integer  
**Status:** alias, safer for future cleanup  
**Meaning:** Correctly spelled alias for `dimention_sw` in some local tester code. Keep `dimention_sw` for compatibility with existing scripts.

### `video_processing.stride`

**Type:** integer  
**Default/fallback:** `16`  
**Meaning:** Step size, in sampled frames, between consecutive sliding windows.

Common settings:

| `dimention_sw` | `stride` | Behavior |
|---:|---:|---|
| 16 | 16 | Non-overlapping 16-frame windows. |
| 16 | 8 | 50% overlap between windows. |
| 8 | 8 | Smaller non-overlapping windows. |

If `stride <= 0`, the scripts fall back to `16`.

### `video_processing.overlapping_sw`

**Type:** boolean  
**Status:** configuration-level flag; current sampler behavior is mainly determined by `window_size` and `stride`.  
**Meaning:** Intended marker for whether windows overlap. In practice, overlap occurs when `stride < dimention_sw`.

---

## 4. `generation`

Controls text generation after the video has been ingested into the KV cache.

```json
"generation": {
  "max_new_tokens": 4,
  "temperature": 0.01,
  "top_p": 0.9,
  "top_k": 0,
  "repetition_penalty": 1.0,
  "ban_eos_steps": 2,
  "open_ended_max_new_tokens": 48,
  "open_ended_temperature": 0.5,
  "open_ended_top_p": 0.95,
  "open_ended_top_k": 0,
  "open_ended_repetition_penalty": 1.2,
  "open_ended_ban_eos_steps": 2
}
```

### `generation.max_new_tokens`

**Type:** integer  
**Single-video tester default in code:** hardcoded to `256` in the shown tester call unless edited to use the config  
**Benchmark use:** used for MCQA/free generation paths  
**Meaning:** Maximum number of tokens to generate.

For multiple-choice numeric answers, small values such as `4` are enough. For descriptions, use larger values such as `128` or `256`.

### `generation.temperature`

**Type:** float  
**Typical range:** `0.01` to `1.0`  
**Meaning:** Sampling randomness. Lower values are more deterministic.

Recommended:

```json
"temperature": 0.01
```

for multiple-choice or exact-answer tasks.

### `generation.top_p`

**Type:** float  
**Range:** `(0, 1]`  
**Meaning:** Nucleus sampling threshold. `0.9` keeps tokens in the smallest probability mass covering 90%.

### `generation.top_k`

**Type:** integer  
**Common value:** `0`  
**Meaning:** Top-k filtering. In the current generation call, `0` means no top-k filtering.

### `generation.repetition_penalty`

**Type:** float  
**Default in some generation functions:** `1.0` or `1.2`  
**Meaning:** Penalizes tokens that have already been generated. Use `1.0` to disable.

### `generation.ban_eos_steps`

**Type:** integer  
**Used by:** `generation_sw` path, if available  
**Meaning:** Prevents early EOS for the first N steps. Useful for very short generations that might terminate immediately.

### Open-ended generation fields

The benchmark script uses a separate set of generation settings for open-ended answers:

| Field | Type | Meaning |
|---|---:|---|
| `open_ended_max_new_tokens` | integer | Maximum answer length for open-ended answers. |
| `open_ended_temperature` | float | Sampling randomness for open-ended answers. |
| `open_ended_top_p` | float | Nucleus sampling for open-ended answers. |
| `open_ended_top_k` | integer | Top-k filtering for open-ended answers. `0` disables it. |
| `open_ended_repetition_penalty` | float | Repetition penalty for open-ended answers. |
| `open_ended_ban_eos_steps` | integer | Prevents EOS during the first N open-ended generation steps. |

---

## 5. `cache`

Controls the KV cache implementation, capacity, sink/hat regions, pruning policy, optional disk offload, retrieval, and debug output.

```json
"cache": {
  "type": "FlexibleCache",
  "MAX_FRAMES_TO_PROCESS": 1000000,
  "mutable_token_size": 8192,
  "use_sliding_window": true
}
```

The cache is built through `create_cache_from_config`, which maps JSON fields into `FlexibleCacheLayer` constructor arguments.

---

### 5.1 Core cache fields

#### `cache.type`

**Type:** string  
**Supported value:** `"FlexibleCache"`

#### `cache.MAX_FRAMES_TO_PROCESS`

**Type:** integer  
**Default in scripts:** `100000` if missing  
**Meaning:** Upper bound on sampled frames consumed by the sampler.

Example:

```json
"MAX_FRAMES_TO_PROCESS": 1000000
```

#### `cache.mutable_token_size`

**Type:** integer  
**Default in layer:** `4096`  
**Meaning:** Token budget for the mutable part of the KV cache. When the mutable region exceeds this size, the cache either truncates, prunes, or offloads depending on the rest of the cache config.

Example:

```json
"mutable_token_size": 8192
```

#### `cache.use_sliding_window`

**Type:** boolean  
**Default in layer:** `true`  
**Meaning:** Enables capacity enforcement. If disabled, the cache behaves as effectively unbounded in this implementation.

---

### 5.2 Sink token settings

Sink tokens are a preserved prefix of the cache. They usually contain the system prompt and initial instruction context.

```json
"use_sink_tokens": true,
"sink_sys_prompt": true,
"size_sink_tokens": 400
```

#### `cache.use_sink_tokens`

**Type:** boolean  
**Default:** `false`  
**Meaning:** Preserve a prefix block as sink tokens.

#### `cache.sink_sys_prompt`

**Type:** boolean  
**Default:** `true`  
**Meaning:** If `true`, the entire priming prompt sequence is used as the sink block. If `false`, `size_sink_tokens` controls the sink size.

Recommended for this pipeline:

```json
"use_sink_tokens": true,
"sink_sys_prompt": true
```

#### `cache.size_sink_tokens`

**Type:** integer  
**Default in layer:** `100`; some priming code fallback uses `128`  
**Meaning:** Static sink size when `sink_sys_prompt` is `false`.

Note: when `sink_sys_prompt` is `true`, this field is usually ignored because the sink size is set to the actual priming sequence length.

---

### 5.3 Hat token settings

Hat tokens are a preserved suffix of the cache. They are useful for keeping recent context intact.

```json
"use_hat_tokens": true,
"hat_category": "frames",
"hat_num_frames": 2,
"size_hat_tokens": 400
```

#### `cache.use_hat_tokens`

**Type:** boolean  
**Default:** `false`  
**Meaning:** Preserve a suffix block at the end of the cache.

#### `cache.hat_category`

**Type:** string  
**Supported values:**

| Value | Meaning |
|---|---|
| `"frames"` | Hat size is based on the most recent `hat_num_frames` when frame-aware tracking is active. |
| `"tokens"` | Hat size is a static token count. |

#### `cache.hat_num_frames`

**Type:** integer  
**Default:** `1`  
**Meaning:** Number of recent frames to preserve when `hat_category == "frames"` and frame-aware pruning/offload is active.

#### `cache.size_hat_tokens`

**Type:** integer  
**Alias:** `HAT_SIZE`  
**Meaning:** Static hat size in tokens. Used when `hat_category == "tokens"`, or as a static fallback when frame-aware hat sizing is not active.

---

### 5.4 Similarity and pruning mode

```json
"use_similarity": true,
"similarity_merge": false,
"similarity_keep_old": false,
"intra_frame_pruning": true,
"inter_frame_pruning": false,
"index_strategy": "sink_cross_attention"
```

#### `cache.use_similarity`

**Type:** boolean  
**Default:** `false`  
**Meaning:** If `true`, overflow is handled through similarity/pruning logic. If `false`, overflow is handled with simple left truncation of the mutable region.

#### `cache.similarity_merge`

**Type:** boolean  
**Default:** `false`  
**Meaning:** For pair-based similarity pruning, merge selected token pairs instead of dropping one token. This does not apply to sink-guided token-wise scorers, which currently use `drop`.

#### `cache.similarity_keep_old`

**Type:** boolean  
**Default:** `false`  
**Meaning:** For pair-based reducers, decides which side of a pair is preserved.

| Value | Pair behavior |
|---|---|
| `false` | Prefer keeping the newer token in a pair. |
| `true` | Prefer keeping the older token in a pair. |

#### `cache.intra_frame_pruning`

**Type:** boolean  
**Default:** `false`  
**Meaning:** Enables pruning inside individual frames. This is the recommended mode for reducing visual tokens while preserving temporal coverage.

#### `cache.inter_frame_pruning`

**Type:** boolean  
**Default:** `false`  
**Meaning:** Enables pruning by comparing neighboring frames. Useful for removing redundant information across frames.

#### `cache.index_strategy`

**Type:** string  
**Default:** `"adiacent_cosine"` in the layer constructor. Note the misspelling is present in the current code.

Supported strategies in the current code include:

| Strategy | Used in | Meaning |
|---|---|---|
| `"adiacent_cosine"` | global, intra, inter | Adjacent-token cosine scorer. Misspelled but supported. |
| `"adjacent_cosine"` | global/intra/inter scorer helpers | Correct spelling alias accepted by scoring functions. |
| `"pseudo_attention"` | global, intra, inter | Context-attention-like token importance. |
| `"pseudo_attn"` | global, intra, inter | Alias for `pseudo_attention`. |
| `"pseudo_attention_meanpair"` | global, intra, inter | Pair score from mean pair importance. |
| `"pseudo_attn_meanpair"` | global, intra, inter | Alias for `pseudo_attention_meanpair`. |
| `"sink_cross_attention"` | intra only | StreamMem-style sink-guided token saliency. |
| `"sink_guided_attention"` | intra only | Alias for sink-guided scoring. |
| `"sys_prompt_cross_attention"` | intra only | Alias for sink/system-prompt guided scoring. |
| `"streammem_attention"` | intra only | Alias for StreamMem-style sink-guided scoring. |
| `"value_norm"` | intra only | Drops low-value-norm tokens. |
| `"van_l2"` | intra only | Alias for value-norm scoring. |

Recommended for the provided config:

```json
"index_strategy": "sink_cross_attention"
```

This uses the sink/system-prompt KV block as a saliency guide for visual-token pruning.

---

### 5.5 Intra-frame pruning controls

```json
"intra_remove_pct": 0.10,
"intra_min_keep_per_frame": 1,
"intra_chunk_size": 16
```

#### `cache.intra_remove_pct`

**Type:** float  
**Meaning:** Fraction of current mutable tokens requested for removal during an intra-frame pruning step. The actual removal is capped by the overflow above `mutable_token_size`.

Example:

```json
"intra_remove_pct": 0.10
```

#### `cache.intra_min_keep_per_frame`

**Type:** integer  
**Default:** `1`  
**Meaning:** Minimum number of tokens to keep per frame during token-wise intra-frame pruning.

Use at least `1` to avoid deleting entire frames.

#### `cache.intra_chunk_size`

**Type:** integer  
**Default:** `64`  
**Meaning:** Number of frames/chunks processed at once by intra-frame scoring. Smaller values reduce peak memory; larger values may improve throughput.

Recommended for high resolution:

```json
"intra_chunk_size": 16
```

---

### 5.6 Inter-frame pruning controls

```json
"inter_frame_pruning": false,
"inter_chunk_size": 64
```

#### `cache.inter_chunk_size`

**Type:** integer  
**Default:** `64`  
**Meaning:** Number of inter-frame pairs processed per chunk when inter-frame pruning is enabled.

Inter-frame pruning currently uses `drop` as the reducer.

---

### 5.7 Sink-guided pruning controls

These fields matter when `index_strategy` is one of:

- `sink_cross_attention`
- `sink_guided_attention`
- `sys_prompt_cross_attention`
- `streammem_attention`

```json
"sink_guidance_max_tokens": 512,
"sink_guidance_q_chunk_size": 128,
"sink_cross_attn_reduction": "mean"
```

#### `cache.sink_guidance_max_tokens`

**Type:** integer  
**Default:** `0`  
**Meaning:** Maximum number of sink tokens kept as guidance. `0` means no explicit cap.

Use a cap such as `512` to reduce memory during sink-guided scoring.

#### `cache.sink_guidance_q_chunk_size`

**Type:** integer  
**Default:** `128`  
**Meaning:** Chunk size used when computing sink-guided attention scores.

Smaller values reduce memory; larger values may improve speed.

#### `cache.sink_cross_attn_reduction`

**Type:** string  
**Supported values:**

| Value | Meaning |
|---|---|
| `"mean"` | Average attention over sink/template tokens. Recommended default. |
| `"max"` | Use maximum attention over sink/template tokens. More selective. |

---

### 5.8 Cleanup controls

```json
"cleanup_min_frame_len": 1
```

#### `cache.cleanup_min_frame_len`

**Type:** integer  
**Default:** `2`  
**Meaning:** Removes frames whose tracked token length is less than or equal to this value during cleanup steps.

With:

```json
"cleanup_min_frame_len": 1
```

only frames with 1 token or fewer are removed during cleanup. Use carefully: if too high, entire small frames can disappear.

---

### 5.9 Debug controls

```json
"debug_cache_layout": true,
"debug_tracking": true
```

#### `cache.debug_cache_layout`

**Type:** boolean  
**Default:** `false`  
**Meaning:** Prints sink/mutable/hat frame layout after cache operations.

#### `cache.debug_tracking`

**Type:** boolean  
**Default:** `false`  
**Meaning:** Prints consistency checks comparing physical KV length against tracked frame lengths.

Recommended for debug runs:

```json
"debug_cache_layout": true,
"debug_tracking": true
```

Recommended for production/benchmark runs:

```json
"debug_cache_layout": false,
"debug_tracking": false
```

---

## 6. Optional explicit `cache.pruning_pipeline`

Instead of relying only on boolean flags like `intra_frame_pruning` and `inter_frame_pruning`, the cache can accept an explicit ordered pruning pipeline.

```json
"pruning_pipeline": [
  {
    "name": "cleanup_before",
    "domain": "cleanup",
    "scorer": "adiacent_cosine",
    "reducer": "remove_small_frames",
    "enabled": true,
    "params": {"min_len": 1}
  },
  {
    "name": "sink_guided_intra",
    "domain": "intra",
    "scorer": "sink_cross_attention",
    "reducer": "drop",
    "enabled": true,
    "params": {
      "remove_pct": 0.10,
      "chunk_size": 16,
      "min_keep_per_frame": 1,
      "q_chunk_size": 128,
      "reduction": "mean"
    }
  }
]
```

Each step has the following schema:

| Field | Type | Meaning |
|---|---:|---|
| `name` | string | Human-readable step name. |
| `domain` | string | One of `cleanup`, `intra`, `inter`, `storage`. |
| `scorer` | string | Scoring strategy. Defaults to `adiacent_cosine`. |
| `reducer` | string | Reduction action. Common values: `drop`, `merge`, `remove_small_frames`, `offload`. |
| `enabled` | boolean | Whether the step is active. Defaults to `true`. |
| `params` | object | Step-specific parameters. |

Supported `domain` values:

| Domain | Purpose |
|---|---|
| `cleanup` | Removes tiny/invalid frame fragments. |
| `intra` | Prunes tokens within each frame. |
| `inter` | Prunes tokens by comparing neighboring frames. |
| `storage` | Offloads excess mutable KV to disk instead of permanently deleting it. |

Note: if `pruning_pipeline` is provided, it is parsed into internal `PruningStep` objects and used instead of the automatically built default pipeline.

---

## 7. Optional disk offloading

Disk offload preserves excess KV chunks on disk instead of deleting them from memory. This is useful for long videos and later QA-time retrieval.

```json
"enable_disk_offload": true,
"offload_dir": "kv_offload",
"offload_min_tokens": 1,
"offload_empty_cache": true,
"offload_quantize": true,
"offload_quant_bits": 8,
"offload_quant_scheme": "per_tensor",
"pruning_pipeline": [
  {
    "name": "offload_left_excess",
    "domain": "storage",
    "reducer": "offload",
    "enabled": true,
    "params": {
      "scope": "left_excess",
      "quantize": true,
      "bits": 8,
      "scheme": "per_tensor"
    }
  }
]
```

### `cache.enable_disk_offload`

**Type:** boolean  
**Default:** `false`  
**Meaning:** Enables per-layer disk offload stores.

### `cache.offload_dir`

**Type:** string  
**Default:** `"kv_offload"`  
**Meaning:** Root directory for temporary offloaded KV chunks.

### `cache.offload_min_tokens`

**Type:** integer  
**Default:** `1`  
**Meaning:** Minimum number of tokens required before an offload operation is performed.

### `cache.offload_empty_cache`

**Type:** boolean  
**Default:** `true`  
**Meaning:** Calls `torch.cuda.empty_cache()` after offloading when CUDA is available.

### `cache.offload_quantize`

**Type:** boolean  
**Default:** `false`  
**Meaning:** Quantizes offloaded KV chunks on disk. This affects storage only; it does not change the live forward-pass cache.

### `cache.offload_quant_bits`

**Type:** integer  
**Supported value:** `8`  
**Meaning:** Bit width for quantized offload. The current store supports int8 only.

### `cache.offload_quant_scheme`

**Type:** string  
**Supported value:** `"per_tensor"`  
**Meaning:** Quantization scheme for offloaded tensors.

### Storage step params

| Field | Type | Meaning |
|---|---:|---|
| `scope` | string | Currently supports `"left_excess"`. |
| `max_tokens` | integer or null | Optional cap on tokens offloaded per step. |
| `quantize` | boolean | Overrides global `offload_quantize` for this step. |
| `bits` / `quant_bits` | integer | Overrides global offload bit width. Currently `8`. |
| `scheme` / `quant_scheme` | string | Overrides global offload scheme. Currently `per_tensor`. |

---

## 8. Optional QA-time retrieval

When disk offload is enabled, the config can enable retrieval before answering a question.

```json
"retrieve": {
  "enabled": true,
  "method": "rekv",
  "top_k_tokens": 8192,
  "block_size_frames": 1,
  "include_live_mutable": true,
  "max_offloaded_chunks": null
}
```

### `cache.retrieve.enabled`

**Type:** boolean  
**Default:** `false`  
**Meaning:** Enables QA-time retrieval before generation.

### `cache.retrieve.method`

**Type:** string  
**Supported value:** `"rekv"`  
**Meaning:** ReKV-like retrieval. It scores blocks using cosine similarity between the mean key vector of a block and a text-only question key proxy.

### `cache.retrieve.top_k_tokens`

**Type:** integer  
**Default:** `cache.mutable_token_size`  
**Meaning:** Maximum number of retrieved mutable tokens to rebuild into the QA cache.

### `cache.retrieve.block_size_frames`

**Type:** integer  
**Default:** `1`  
**Meaning:** Number of consecutive frames per retrieval block.

### `cache.retrieve.include_live_mutable`

**Type:** boolean  
**Default:** `true`  
**Meaning:** Whether live mutable blocks are considered together with offloaded blocks.

### `cache.retrieve.max_offloaded_chunks`

**Type:** integer or null  
**Default:** `null`  
**Meaning:** Optional limit on how many recent offloaded chunks are searched.

---

## 9. `single_video_test`

Controls single-video test behavior.

```json
"single_video_test": {
  "until_second": null,
  "question": "Describe this video in detail."
}
```

### `single_video_test.until_second`

**Type:** number or null  
**Meaning:** Optional cutoff time in seconds. If set, video ingestion stops at this timestamp.

Examples:

```json
"until_second": null
```

Process the whole video.

```json
"until_second": 120
```

Process only the first 120 seconds.

### `single_video_test.question`

**Type:** string  
**Default in tester helper:** `"Describe this video in detail."`  
**Meaning:** User question asked after the cache has been filled.

Example:

```json
"question": "Describe this video in detail."
```

---

## 10. `verbose`

```json
"verbose": true
```

**Type:** boolean  
**Meaning:** Enables debug prints at the script level. For reviewer runs, `true` is useful because it shows config metadata, input-file metadata, CUDA device, cache size, and optional offload/debug stats. For double-blind review, keep paths anonymized or use relative paths before sharing logs.

For large benchmark runs, set:

```json
"verbose": false
```

---

## Recommended default configuration

This configuration is a good user-facing default: it uses sink-guided intra-frame pruning, keeps the system prompt as sink, preserves the last two frames as hat, and prints detailed cache tracking information.

```json
{
  "paths": {
    "video_path": "data/videos/example_video.mp4",
    "prompt_path": "prompts/ingestion_prompt.txt"
  },
  "model": {
    "model_id": "Qwen/Qwen3-VL-8B-Instruct",
    "min_pixels": 65536,
    "max_pixels": 1638400
  },
  "video_processing": {
    "fps_to_process": 1,
    "target_resolution": [896, 512],
    "stride": 16,
    "use_sliding_window": true,
    "dimention_sw": 16,
    "overlapping_sw": false
  },
  "generation": {
    "max_new_tokens": 4,
    "temperature": 0.01,
    "top_p": 0.9,
    "open_ended_max_new_tokens": 48,
    "open_ended_temperature": 0.5,
    "open_ended_top_p": 0.95,
    "open_ended_top_k": 0,
    "open_ended_repetition_penalty": 1.2,
    "open_ended_ban_eos_steps": 2
  },
  "cache": {
    "type": "FlexibleCache",
    "MAX_FRAMES_TO_PROCESS": 1000000,
    "mutable_token_size": 8192,
    "use_sliding_window": true,
    "use_sink_tokens": true,
    "sink_sys_prompt": true,
    "size_sink_tokens": 400,
    "use_hat_tokens": true,
    "hat_category": "frames",
    "hat_num_frames": 2,
    "size_hat_tokens": 400,
    "use_similarity": true,
    "similarity_merge": false,
    "similarity_keep_old": false,
    "intra_frame_pruning": true,
    "inter_frame_pruning": false,
    "index_strategy": "sink_cross_attention",
    "intra_remove_pct": 0.10,
    "intra_min_keep_per_frame": 1,
    "intra_chunk_size": 16,
    "sink_guidance_max_tokens": 512,
    "sink_guidance_q_chunk_size": 128,
    "sink_cross_attn_reduction": "mean",
    "cleanup_min_frame_len": 1,
    "debug_cache_layout": true,
    "debug_tracking": true
  },
  "single_video_test": {
    "until_second": null,
    "question": "Describe this video in detail."
  },
  "verbose": true
}
```

---

## Suggested presets

### A. Baseline sliding-window cache, no similarity pruning

Use this to compare against simple left-truncation.

```json
"cache": {
  "type": "FlexibleCache",
  "mutable_token_size": 8192,
  "use_sliding_window": true,
  "use_sink_tokens": true,
  "sink_sys_prompt": true,
  "use_hat_tokens": true,
  "hat_category": "frames",
  "hat_num_frames": 2,
  "size_hat_tokens": 400,
  "use_similarity": false,
  "intra_frame_pruning": false,
  "inter_frame_pruning": false,
  "debug_cache_layout": false,
  "debug_tracking": false
}
```

### B. Sink-guided intra-frame pruning

Use this as the main optimized configuration.

```json
"cache": {
  "type": "FlexibleCache",
  "mutable_token_size": 8192,
  "use_sliding_window": true,
  "use_sink_tokens": true,
  "sink_sys_prompt": true,
  "use_hat_tokens": true,
  "hat_category": "frames",
  "hat_num_frames": 2,
  "use_similarity": true,
  "intra_frame_pruning": true,
  "inter_frame_pruning": false,
  "index_strategy": "sink_cross_attention",
  "intra_remove_pct": 0.10,
  "intra_min_keep_per_frame": 1,
  "intra_chunk_size": 16,
  "sink_guidance_max_tokens": 512,
  "sink_guidance_q_chunk_size": 128,
  "sink_cross_attn_reduction": "mean"
}
```

### C. Intra + inter frame pruning

Use this to test both within-frame and across-frame redundancy removal.

```json
"cache": {
  "type": "FlexibleCache",
  "mutable_token_size": 8192,
  "use_sliding_window": true,
  "use_sink_tokens": true,
  "sink_sys_prompt": true,
  "use_hat_tokens": true,
  "hat_category": "frames",
  "hat_num_frames": 2,
  "use_similarity": true,
  "similarity_merge": false,
  "similarity_keep_old": false,
  "intra_frame_pruning": true,
  "inter_frame_pruning": true,
  "index_strategy": "adiacent_cosine",
  "intra_remove_pct": 0.10,
  "intra_min_keep_per_frame": 1,
  "intra_chunk_size": 32,
  "inter_chunk_size": 32,
  "cleanup_min_frame_len": 1
}
```

### D. Disk offload + ReKV retrieval

Use this for very long videos where older KV should be recoverable at QA time.

```json
"cache": {
  "type": "FlexibleCache",
  "mutable_token_size": 8192,
  "use_sliding_window": true,
  "use_sink_tokens": true,
  "sink_sys_prompt": true,
  "use_hat_tokens": true,
  "hat_category": "frames",
  "hat_num_frames": 2,
  "use_similarity": false,
  "intra_frame_pruning": false,
  "inter_frame_pruning": false,
  "enable_disk_offload": true,
  "offload_dir": "kv_offload",
  "offload_min_tokens": 1,
  "offload_empty_cache": true,
  "offload_quantize": true,
  "offload_quant_bits": 8,
  "offload_quant_scheme": "per_tensor",
  "pruning_pipeline": [
    {
      "name": "offload_left_excess",
      "domain": "storage",
      "reducer": "offload",
      "enabled": true,
      "params": {
        "scope": "left_excess",
        "quantize": true,
        "bits": 8,
        "scheme": "per_tensor"
      }
    }
  ],
  "retrieve": {
    "enabled": true,
    "method": "rekv",
    "top_k_tokens": 8192,
    "block_size_frames": 1,
    "include_live_mutable": true,
    "max_offloaded_chunks": null
  }
}
```

---

## Practical notes

1. Use `cache.type = "FlexibleCache"` for this tester.
2. Keep both `video_processing.use_sliding_window` and `cache.use_sliding_window` set to `true`.
3. Keep the misspelled `dimention_sw` key unless the scripts are updated everywhere.
4. For sink-guided pruning, set `use_sink_tokens = true` and `sink_sys_prompt = true`; otherwise there may be no useful sink guidance.
5. For high-resolution frames, reduce `intra_chunk_size` and/or `sink_guidance_q_chunk_size` if VRAM is tight.
6. Set `debug_cache_layout` and `debug_tracking` to `true` only for inspection runs.
7. Disk offload should be paired with a `storage` step in `pruning_pipeline`; enabling `enable_disk_offload` alone creates stores but does not by itself define when to offload.
8. Offload quantization is storage-only. It does not quantize the live KV cache used during forward passes.
9. `retrieve.enabled` is useful only when the cache object supports `retrieve_for_qa`, which `FlexibleCache` does.
10. If `until_second` is `null`, the full video is processed; set it to a number for fast smoke tests.
