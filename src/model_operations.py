import torch
import cv2
import time
import torch.nn.functional as F

from .utils import resize_video_frame
from .tensor_ops import (
    process_pil_image, compute_visual_embeddings, compute_projected_embeddings,
    process_text_conversation, compute_text_embeddings, replace_placeholder
)

from Caches.CacheFactory import create_cache_from_config
from .stc_pruner import build_stc_pruner


def sampling_generation_function(
    model,
    processor,
    cache,
    inputs,
    max_new_tokens=128,
    temperature=0.01,
    top_p=0.9,
    repetition_penalty=1.2,
    verbose=True,
):
    decoded_sequence = ""
    current_inputs_ids = inputs["input_ids"]
    current_cache = cache
    generated_token_ids = []
    temperature = max(1e-8, temperature)

    if hasattr(cache, "global_token_counter"):
        current_global_step = cache.global_token_counter
    else:
        current_global_step = cache.get_seq_length()

    im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_token_id = processor.tokenizer.eos_token_id
    temp_attn_mask = torch.ones_like(current_inputs_ids)

    _, text_rope_deltas = model.model.get_rope_index(
        input_ids=current_inputs_ids,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=temp_attn_mask
    )

    for i in range(max_new_tokens):
        with torch.no_grad():
            current_seq_len = current_inputs_ids.shape[1]
            physical_past_len = current_cache.get_seq_length()

            attention_mask = torch.ones(
                (1, physical_past_len + current_seq_len),
                device=model.device
            )
            position_ids = torch.arange(
                current_global_step,
                current_global_step + current_seq_len,
                device=model.device
            ).unsqueeze(0)

            output = model.forward(
                input_ids=current_inputs_ids,
                past_key_values=current_cache,
                attention_mask=attention_mask,
                position_ids=position_ids,
                rope_deltas=text_rope_deltas if i == 0 else None,
                use_cache=True
            )

            logits = output.logits[:, -1, :]

            if repetition_penalty != 1.0 and len(generated_token_ids) > 0:
                current_logits = logits[0]
                unique_seen = list(set(generated_token_ids))
                for token_id in unique_seen:
                    if current_logits[token_id] < 0:
                        current_logits[token_id] *= repetition_penalty
                    else:
                        current_logits[token_id] /= repetition_penalty
                logits[0] = current_logits

            if i == 0:
                logits[:, eos_token_id] = -float("inf")
                if im_end_id is not None:
                    logits[:, im_end_id] = -float("inf")

            logits = logits.float() / temperature
            probs = F.softmax(logits, dim=-1)

            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                probs[indices_to_remove] = 0.0

                if probs.sum() == 0:
                    probs = F.softmax(logits, dim=-1)
                else:
                    probs = probs / probs.sum(dim=-1, keepdim=True)

            next_token_index = torch.multinomial(probs, num_samples=1)
            token_item = next_token_index.squeeze().item()

            if verbose and i < 3:
                top_probs, top_indices = torch.topk(probs[0], 5)
                top_tokens = [processor.tokenizer.decode([idx.item()]) for idx in top_indices]
                print(
                    f"  Step {i}: token={processor.tokenizer.decode([token_item])} | "
                    f"Top 5: {list(zip(top_tokens, top_probs.tolist()))}"
                )

            decoded_token = processor.decode(token_item)
            decoded_sequence += decoded_token
            generated_token_ids.append(token_item)

            current_cache = output.past_key_values
            current_inputs_ids = next_token_index
            current_global_step += current_seq_len

            if token_item == eos_token_id or token_item == im_end_id:
                break

    cache.global_token_counter = current_global_step
    return decoded_sequence


def _prime_cache_sw(model, processor, cache_cfg, prompt_path, verbose=True):
    if verbose:
        print("DEBUG: Starting Priming Phase...", flush=True)

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_content = f.read()

    priming_conversation = [
        {"role": "system", "content": [{"type": "text", "text": prompt_content}]},
        {"role": "user", "content": [{"type": "text", "text": "Video stream starts now:"}]},
    ]

    priming_input_ids = process_text_conversation(
        priming_conversation, processor, model.device
    )
    priming_embeds, _ = compute_text_embeddings(
        priming_input_ids, 0, processor, model
    )

    global_token_counter = 0

    with torch.no_grad():
        priming_seq_len = priming_embeds.shape[1]

        token_budget = cache_cfg.get("mutable_token_size", 1024)

        sink_tokens_size = 0
        if cache_cfg.get("use_sink_tokens", False):
            if cache_cfg.get("sink_sys_prompt", False):
                sink_tokens_size = priming_seq_len
            else:
                sink_tokens_size = cache_cfg.get("size_sink_tokens", 128)

        hat_tokens_size = 0
        if cache_cfg.get("use_hat_tokens", False):
            if cache_cfg.get("hat_category", "frames") == "frames":
                hat_tokens_size = cache_cfg.get("hat_num_frames", 1) * 448
            else:
                hat_tokens_size = cache_cfg.get("size_hat_tokens", 400)

        total_k_capacity = sink_tokens_size + token_budget + hat_tokens_size

        cache_instance = create_cache_from_config(
            config=model.model.config,
            cache_cfg=cache_cfg,
            K=total_k_capacity,
        )

        # STC-Pruner is an opt-in visual-token prefill policy. It is attached
        # to the per-video cache object only to share lifecycle and metrics.
        stc_pruner = build_stc_pruner(cache_cfg)
        if stc_pruner is not None:
            cache_instance.stc_pruner = stc_pruner

        if hasattr(cache_instance, "set_sink_size"):
            cache_instance.set_sink_size(sink_tokens_size)

        if hasattr(cache_instance, "set_hat_size"):
            cache_instance.set_hat_size(hat_tokens_size)
            if verbose:
                print("settati gli hat tokens", flush=True)

        priming_position_ids = torch.arange(
            0, priming_seq_len, dtype=torch.long, device=model.device
        ).unsqueeze(0)

        _ = model.forward(
            inputs_embeds=priming_embeds,
            past_key_values=cache_instance,
            position_ids=priming_position_ids,
        )

        if hasattr(cache_instance, "capture_sink_guidance"):
            cache_instance.capture_sink_guidance()

        global_token_counter += priming_seq_len

    cache_instance.global_token_counter = global_token_counter

    if verbose:
        print(f"[DEBUG CACHE] cache_instance type: {type(cache_instance)}", flush=True)

        if hasattr(cache_instance, "layers") and len(cache_instance.layers) > 0:
            l0 = cache_instance.layers[0]
            print(f"[DEBUG CACHE] layer0 type: {type(l0)}", flush=True)
            print(f"[DEBUG CACHE] layer0.use_similarity: {getattr(l0, 'use_similarity', None)}", flush=True)
            print(f"[DEBUG CACHE] layer0.intra_frame_pruning: {getattr(l0, 'intra_frame_pruning', None)}", flush=True)
            print(f"[DEBUG CACHE] layer0.inter_frame_pruning: {getattr(l0, 'inter_frame_pruning', None)}", flush=True)
            print(f"[DEBUG CACHE] layer0.enable_disk_offload: {getattr(l0, 'enable_disk_offload', None)}", flush=True)
            print(f"[DEBUG CACHE] layer0.similarity_merge: {getattr(l0, 'similarity_merge', None)}", flush=True)
            print(f"[DEBUG CACHE] layer0.similarity_keep_old: {getattr(l0, 'similarity_keep_old', None)}", flush=True)

        print(f"[DEBUG CACHE] cache_cfg: {cache_cfg}", flush=True)
        print(f"[DEBUG CACHE] total_k_capacity: {total_k_capacity}", flush=True)
        print(f"[DEBUG CACHE] sink_tokens_size: {sink_tokens_size}", flush=True)
        print(f"[DEBUG CACHE] hat_tokens_size: {hat_tokens_size}", flush=True)

    return cache_instance


def _extract_frame_token_counts_from_grid(image_grid_thw: torch.Tensor, processor=None, model=None):
    """
    Derives the REAL number of visual tokens per frame starting from image_grid_thw.

    For Qwen-VL, image_grid_thw is typically pre-spatial-merge.
    So the final tokens per frame are:

        t * h * w / (merge_size ** 2)

    In your case merge_size=2, hence /4.
    """
    if image_grid_thw is None:
        raise ValueError("image_grid_thw is None")

    if image_grid_thw.dim() != 2 or image_grid_thw.shape[-1] != 3:
        raise ValueError(f"Unexpected image_grid_thw shape: {tuple(image_grid_thw.shape)}")

    merge_size = 2

    candidates = [
        getattr(getattr(processor, "image_processor", None), "merge_size", None),
        getattr(getattr(processor, "image_processor", None), "spatial_merge_size", None),
        getattr(getattr(model, "config", None), "vision_config", None),
        getattr(getattr(getattr(model, "model", None), "config", None), "vision_config", None),
    ]

    for c in candidates:
        if c is None:
            continue
        if isinstance(c, int):
            merge_size = int(c)
            break
        if hasattr(c, "merge_size") and getattr(c, "merge_size") is not None:
            merge_size = int(c.merge_size)
            break
        if hasattr(c, "spatial_merge_size") and getattr(c, "spatial_merge_size") is not None:
            merge_size = int(c.spatial_merge_size)
            break

    denom = merge_size * merge_size

    frame_token_counts = []
    for row in image_grid_thw:
        t, h, w = [int(x) for x in row.tolist()]
        raw = int(t * h * w)

        if raw % denom != 0:
            raise ValueError(
                f"Raw frame token count {raw} is not divisible by merge denom {denom}. "
                f"grid row={row.tolist()}, merge_size={merge_size}"
            )

        frame_token_counts.append(raw // denom)

    return frame_token_counts


def _is_storage_offload_step(step) -> bool:
    """
    Recognizes a storage/offload step both before and after the config parsing.

    Before parsing:
        step is a dict.

    After CacheFactory._parse_pruning_pipeline:
        step is a PruningStep dataclass.
    """
    if isinstance(step, dict):
        domain = str(step.get("domain", "")).strip().lower()
        reducer = str(step.get("reducer", "")).strip().lower()
    else:
        domain = str(getattr(step, "domain", "")).strip().lower()
        reducer = str(getattr(step, "reducer", "")).strip().lower()

    return domain == "storage" and reducer == "offload"


def _cache_has_storage_offload(cache_instance) -> bool:
    """
    Determines whether the cache must use frame-aware prepare/finalize even
    when intra/inter pruning are false but storage/offload is active.
    """
    if not hasattr(cache_instance, "cache_config"):
        return False

    cfg = cache_instance.cache_config
    pipeline = cfg.get("pruning_pipeline", []) or []

    has_storage_offload = any(_is_storage_offload_step(s) for s in pipeline)

    return bool(
        cfg.get("enable_disk_offload", False)
        and has_storage_offload
    )


def _append_frame_batches_to_existing_cache_sw(
    frame_batches,
    model,
    processor,
    cache_instance,
    size=(360, 240),
    verbose=True,
):
    """
    Low-level helper:
    takes directly a sequence of already prepared frame batches
    and appends them to an existing cache using the same ingestion
    logic as fill_cache_sw.
    """
    if not hasattr(cache_instance, "global_token_counter"):
        raise ValueError("cache_instance must have global_token_counter set before append")

    global_token_counter = cache_instance.global_token_counter

    frame_ingestion_times = []
    cache_size_history = []
    processed_batches_count = 0

    for frames_batch in frame_batches:
        if not isinstance(frames_batch, list):
            frames_batch = [frames_batch]

        if len(frames_batch) == 0:
            continue

        if verbose:
            print(
                f"DEBUG: Batch {processed_batches_count}: Processing {len(frames_batch)} frames.",
                flush=True,
            )

        with torch.no_grad():
            pil_images = [resize_video_frame(frm, size) for frm in frames_batch]

            video_inputs = processor.image_processor(
                images=pil_images,
                return_tensors="pt",
            )
            video_inputs = video_inputs.to(model.device)

            combined_pixel_values = video_inputs.pixel_values.to(torch.bfloat16)
            image_grid_thw = video_inputs.image_grid_thw

            vision_output, total_vision_tokens = compute_visual_embeddings(
                combined_pixel_values, model, grid_thw=image_grid_thw
            )

            if verbose and (processed_batches_count < 3 or processed_batches_count % 50 == 0):
                print(f"DEBUG: Total vision tokens: {total_vision_tokens}", flush=True)

            projected_vision_embeddings, deepstack_visual_embeds = compute_projected_embeddings(
                vision_output
            )

            frame_token_counts = _extract_frame_token_counts_from_grid(
                image_grid_thw, processor=processor, model=model
            )

            if int(sum(frame_token_counts)) != int(total_vision_tokens):
                raise ValueError(
                    f"Mismatch frame_token_counts={sum(frame_token_counts)} "
                    f"vs total_vision_tokens={total_vision_tokens}"
                )

            stc_pruner = getattr(cache_instance, "stc_pruner", None)
            if stc_pruner is not None and getattr(stc_pruner, "enabled", False):
                stc_result = stc_pruner.prune(
                    projected_embeddings=projected_vision_embeddings,
                    frame_token_counts=frame_token_counts,
                    deepstack_visual_embeds=deepstack_visual_embeds,
                )
                projected_vision_embeddings = stc_result.projected_embeddings
                deepstack_visual_embeds = stc_result.deepstack_visual_embeds
                frame_token_counts = stc_result.frame_token_counts
                total_vision_tokens = stc_result.retained_tokens

                if verbose and stc_pruner.should_log():
                    metrics = stc_pruner.metrics()
                    print(
                        "[STC-Pruner] "
                        f"batch={metrics['total_batches']} | "
                        f"frames={len(frame_token_counts)} | "
                        f"input={stc_result.input_tokens} | "
                        f"retained={stc_result.retained_tokens} | "
                        f"retained_per_frame={frame_token_counts} | "
                        f"alpha={metrics['alpha']:.3f} | "
                        f"history={metrics['history_size_current']}",
                        flush=True,
                    )

            content_list = [{"type": "image"} for _ in range(len(frames_batch))]
            conversation = [{"role": "user", "content": content_list}]

            input_ids_initial = process_text_conversation(
                conversation, processor, model.device
            )
            text_embeddings, expanded_input_ids = compute_text_embeddings(
                input_ids_initial, total_vision_tokens, processor, model
            )

            input_embeddings, visual_pos_mask = replace_placeholder(
                expanded_input_ids,
                text_embeddings,
                projected_vision_embeddings,
                processor,
            )

            # Qwen's dense-grid RoPE helper assumes image_grid_thw still
            # describes every dense visual token. Its output is unused in this
            # ingestion path; preserve the legacy call unless STC has pruned.
            if stc_pruner is None or not getattr(stc_pruner, "enabled", False):
                temp_attention_mask = torch.ones_like(expanded_input_ids)
                _, rope_deltas = model.model.get_rope_index(
                    input_ids=expanded_input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=None,
                    attention_mask=temp_attention_mask,
                )

            current_seq_len = input_embeddings.shape[1]

            frame_aware_pruning = False
            if hasattr(cache_instance, "cache_config"):
                cfg = cache_instance.cache_config

                frame_aware_pruning = bool(
                    cfg.get("frame_aware_ingestion", False)
                    or cfg.get("intra_frame_pruning", False)
                    or cfg.get("inter_frame_pruning", False)
                    or bool((cfg.get("adakv", {}) or {}).get("enabled", False))
                    or _cache_has_storage_offload(cache_instance)
                )

            # IMPORTANT:
            # - frame_aware_pruning=True:
            #     uses prepare/finalize and then keeps only the visual tokens of the append.
            #     Also needed for offload-only, because without frame-aware tracking
            #     we would not have correct mutable/hat lengths.
            #
            # - frame_aware_pruning=False:
            #     uses the classic add_frame_tokens tracking.
            if frame_aware_pruning and hasattr(cache_instance, "prepare_frame_batch"):
                cache_instance.prepare_frame_batch(
                    frame_token_counts=frame_token_counts,
                    visual_pos_mask=visual_pos_mask,
                    appended_seq_len=current_seq_len,
                )
            elif hasattr(cache_instance, "add_frame_tokens"):
                cache_instance.add_frame_tokens(frame_token_counts)

            physical_past_len = cache_instance.get_seq_length()

            attention_mask = torch.ones(
                (1, physical_past_len + current_seq_len),
                dtype=torch.long,
                device=model.device,
            )

            position_ids = torch.arange(
                global_token_counter,
                global_token_counter + current_seq_len,
                dtype=torch.long,
                device=model.device,
            ).unsqueeze(0)

            t_frame_start = time.perf_counter()

            _ = model.model.language_model(
                inputs_embeds=input_embeddings,
                past_key_values=cache_instance,
                attention_mask=attention_mask,
                position_ids=position_ids,
                visual_pos_masks=visual_pos_mask,
                deepstack_visual_embeds=deepstack_visual_embeds,
                use_cache=True,
            )

            if frame_aware_pruning and hasattr(cache_instance, "finalize_frame_batch"):
                cache_instance.finalize_frame_batch()

            if verbose and hasattr(cache_instance, "get_frame_boundaries"):
                boundaries = cache_instance.get_frame_boundaries()
                print(f"[DEBUG FRAME CACHE] after batch {processed_batches_count}", flush=True)
                for b in boundaries:
                    print(
                        f"  frame_idx={b['frame_idx']} | region={b['region']} | "
                        f"len={b['length']} | abs=[{b['abs_start']},{b['abs_end']})",
                        flush=True,
                    )

            t_frame_end = time.perf_counter()

            global_token_counter += current_seq_len
            frame_ingestion_times.append(t_frame_end - t_frame_start)
            processed_batches_count += 1

            current_cache_size = cache_instance.get_seq_length()
            cache_size_history.append(current_cache_size)

            if verbose:
                print(
                    f"Processed Batch {processed_batches_count} | Frames: {len(frames_batch)} | "
                    f"Cache Size: {current_cache_size} tokens",
                    flush=True,
                )

    if verbose:
        print("DEBUG: Finished Video Ingestion.", flush=True)

    cache_instance.global_token_counter = global_token_counter
    return cache_instance, frame_ingestion_times, cache_size_history


def _append_sampler_to_existing_cache_sw(
    sampler,
    model,
    processor,
    cache_instance,
    size=(360, 240),
    verbose=True,
):
    normalized_batches = []

    for data_item in sampler:
        frames_batch = data_item if isinstance(data_item, list) else [data_item]
        if len(frames_batch) > 0:
            normalized_batches.append(frames_batch)

    return _append_frame_batches_to_existing_cache_sw(
        frame_batches=normalized_batches,
        model=model,
        processor=processor,
        cache_instance=cache_instance,
        size=size,
        verbose=verbose,
    )


def fill_cache_sw(
    sampler,
    model,
    processor,
    cache_cfg,
    prompt_path,
    size=(360, 240),
    verbose=True,
):
    cache_instance = _prime_cache_sw(
        model=model,
        processor=processor,
        cache_cfg=cache_cfg,
        prompt_path=prompt_path,
        verbose=verbose,
    )

    cache_instance, frame_ingestion_times, cache_size_history = _append_sampler_to_existing_cache_sw(
        sampler=sampler,
        model=model,
        processor=processor,
        cache_instance=cache_instance,
        size=size,
        verbose=verbose,
    )

    avg_frame_time = (
        sum(frame_ingestion_times) / len(frame_ingestion_times)
        if frame_ingestion_times else 0.0
    )

    return cache_instance, avg_frame_time, cache_size_history
