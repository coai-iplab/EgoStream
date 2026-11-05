# Sliding Window Cache Filling - Complete Documentation

This folder contains comprehensive documentation for the Sliding Window (SW) cache filling implementation.

## Documentation Structure

1. **[overview.md](file:///home/rforte/nas_home/OEM-VQA/ECCV/docs/sliding_window/overview.md)** - High-level architecture and flow
2. **[rope_explained.md](file:///home/rforte/nas_home/OEM-VQA/ECCV/docs/sliding_window/rope_explained.md)** - Detailed explanation of ROPE (Rotary Position Embeddings)
3. **[attention_strategy.md](file:///home/rforte/nas_home/OEM-VQA/ECCV/docs/sliding_window/attention_strategy.md)** - Active query vs cached context (why we use embeddings)
4. **[tensor_ops_functions.md](file:///home/rforte/nas_home/OEM-VQA/ECCV/docs/sliding_window/tensor_ops_functions.md)** - Deep dive into each support function
5. **[cache_filling_flow.md](file:///home/rforte/nas_home/OEM-VQA/ECCV/docs/sliding_window/cache_filling_flow.md)** - Step-by-step walkthrough of fill_cache_sw
6. **[generation_flow.md](file:///home/rforte/nas_home/OEM-VQA/ECCV/docs/sliding_window/generation_flow.md)** - How generation works with SW cache

## Quick Reference

**Purpose**: Process long videos efficiently using a sliding window approach that treats batches of frames as sequences of images rather than a single video.

**Key Files**:
- [`src/model_operations.py::fill_cache_sw`](file:///home/rforte/nas_home/OEM-VQA/ECCV/src/model_operations.py#L131) - Main cache filling function
- [`src/tensor_ops_sw.py`](file:///home/rforte/nas_home/OEM-VQA/ECCV/src/tensor_ops_sw.py) - Support functions for processing
- [`src/generation_sw.py`](file:///home/rforte/nas_home/OEM-VQA/ECCV/src/generation_sw.py) - Generation with SW cache
- [`src/video_sampler.py::SlidingWindowSampler`](file:///home/rforte/nas_home/OEM-VQA/ECCV/src/video_sampler.py) - Frame sampling logic

**Key Configuration Parameters**:
```json
{
  "video_processing": {
    "use_sliding_window": true,
    "dimention_sw": 20,        // Window size (frames per batch)
    "stride": 10,              // Non-overlapping stride
    "overlapping_sw": false
  }
}
```

## Core Concepts

### Sliding Window Strategy
Instead of processing the entire video as one unit, we:
1. Split video into overlapping/non-overlapping windows of N frames
2. Process each window as N separate images in a single forward pass
3. Use cache eviction to maintain fixed memory footprint

### ROPE (Rotary Position Embeddings)
Vision-language models use ROPE to encode positional information. For videos, ROPE must account for both spatial (within-frame) and temporal (across-frame) dimensions.

### Sequence-of-Images vs Video Processing
- **Video mode**: Treats all frames as a temporal sequence (one video tensor)
- **Sequence-of-Images mode** (our approach): Treats each frame independently with separate image placeholders
- This gives finer control over cache management and token positioning
