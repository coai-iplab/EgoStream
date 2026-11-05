import time

import torch
import torch.nn.functional as F

from .utils import resize_video_frame
from Caches.CacheFactory import create_cache_from_config


def _get_lm(model):
    return model.language_model


def _embed_tokens(model, input_ids):
    return model.get_input_embeddings()(input_ids)


def _token_id(processor, token):
    token_id = processor.tokenizer.convert_tokens_to_ids(token)
    if token_id == processor.tokenizer.unk_token_id:
        return None
    return token_id


def _eos_token_ids(processor):
    ids = {processor.tokenizer.eos_token_id}
    for token in ("<|im_end|>", "</s>"):
        token_id = _token_id(processor, token)
        if token_id is not None:
            ids.add(token_id)
    return {x for x in ids if x is not None}


def _molmo2_compatible_conversation(conversation):
    system_chunks = []
    normalized = []

    for message in conversation:
        role = str(message.get("role", "user")).lower()
        content = message.get("content", [])
        if role == "system":
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    system_chunks.append(str(item.get("text", "")))
            continue
        normalized.append({"role": role, "content": content})

    if system_chunks:
        prefix = "\n\n".join(chunk for chunk in system_chunks if chunk)
        if normalized and normalized[0]["role"] == "user":
            normalized[0] = {
                "role": "user",
                "content": [{"type": "text", "text": prefix}] + list(normalized[0].get("content", [])),
            }
        else:
            normalized.insert(0, {"role": "user", "content": [{"type": "text", "text": prefix}]})

    return normalized


def _prompt_from_conversation(processor, conversation, add_generation_prompt=True):
    return processor.apply_chat_template(
        _molmo2_compatible_conversation(conversation),
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )


def _tokenize_text_conversation(conversation, processor, device, add_generation_prompt=False):
    prompt = _prompt_from_conversation(
        processor,
        conversation,
        add_generation_prompt=add_generation_prompt,
    )
    inputs = processor.tokenizer(prompt, return_tensors="pt")
    return inputs.input_ids.to(device)


def _image_patch_id(model, processor):
    token_id = getattr(getattr(model, "config", None), "image_patch_id", None)
    if token_id is not None:
        return int(token_id)

    image_tokens = getattr(processor, "image_token_ids", None)
    if image_tokens:
        return int(image_tokens[0])

    token_id = _token_id(processor, "<im_patch>")
    if token_id is not None:
        return int(token_id)

    raise ValueError("Could not determine Molmo2 image patch token id")


def _frame_token_counts_from_input_ids(input_ids, image_patch_id):
    positions = torch.where(input_ids[0] == image_patch_id)[0].tolist()
    if not positions:
        return []

    counts = []
    run_len = 1
    for prev, cur in zip(positions, positions[1:]):
        if cur == prev + 1:
            run_len += 1
        else:
            counts.append(run_len)
            run_len = 1
    counts.append(run_len)
    return counts


def _prepare_visual_inputs(frames_batch, processor, model, size):
    pil_images = [resize_video_frame(frame, size) for frame in frames_batch]
    conversation = [
        {
            "role": "user",
            "content": [{"type": "image"} for _ in pil_images],
        }
    ]
    prompt = _prompt_from_conversation(
        processor,
        conversation,
        add_generation_prompt=False,
    )
    inputs = processor(
        text=prompt,
        images=pil_images,
        return_tensors="pt",
    )
    return inputs.to(model.device)


def _is_storage_offload_step(step) -> bool:
    if isinstance(step, dict):
        domain = str(step.get("domain", "")).strip().lower()
        reducer = str(step.get("reducer", "")).strip().lower()
    else:
        domain = str(getattr(step, "domain", "")).strip().lower()
        reducer = str(getattr(step, "reducer", "")).strip().lower()

    return domain == "storage" and reducer == "offload"


def _cache_has_storage_offload(cache_instance) -> bool:
    if not hasattr(cache_instance, "cache_config"):
        return False

    cfg = cache_instance.cache_config
    pipeline = cfg.get("pruning_pipeline", []) or []
    has_storage_offload = any(_is_storage_offload_step(s) for s in pipeline)
    return bool(cfg.get("enable_disk_offload", False) and has_storage_offload)


def _forward_kwargs_from_inputs(inputs):
    keys = (
        "input_ids",
        "pixel_values",
        "image_token_pooling",
        "image_grids",
        "image_num_crops",
        "pixel_values_videos",
        "video_token_pooling",
        "video_grids",
    )
    return {k: inputs[k] for k in keys if k in inputs and inputs[k] is not None}


def sampling_generation_function_molmo2_sw(
    model,
    processor,
    cache,
    inputs,
    max_new_tokens=128,
    temperature=0.4,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.02,
    ban_eos_steps=2,
    do_sample=False,
    position_mode="global",
    verbose=True,
):
    decoded_sequence = ""
    current_inputs_ids = inputs["input_ids"].to(model.device)
    current_cache = cache
    generated_token_ids = []
    temperature = max(0.1, float(temperature))

    current_global_step = getattr(cache, "global_token_counter", cache.get_seq_length())
    eos_token_ids = _eos_token_ids(processor)

    if verbose:
        print(f"[GEN_MOLMO2] Starting generation with cache size: {current_cache.get_seq_length()} tokens")
        print(f"[GEN_MOLMO2] Question length: {current_inputs_ids.shape[1]} tokens")

    first_iteration = True
    question_embeds = _embed_tokens(model, current_inputs_ids)

    for i in range(max_new_tokens):
        with torch.no_grad():
            physical_past_len = current_cache.get_seq_length()

            if first_iteration:
                current_seq_len = question_embeds.shape[1]
                model_inputs = {"inputs_embeds": question_embeds}
                first_iteration = False
            else:
                current_seq_len = current_inputs_ids.shape[1]
                model_inputs = {"input_ids": current_inputs_ids}

            attention_mask = torch.ones(
                (1, physical_past_len + current_seq_len),
                dtype=torch.long,
                device=model.device,
            )
            position_start = physical_past_len if position_mode == "compact" else current_global_step
            position_ids = torch.arange(
                position_start,
                position_start + current_seq_len,
                dtype=torch.long,
                device=model.device,
            ).unsqueeze(0)
            cache_position = torch.arange(
                physical_past_len,
                physical_past_len + current_seq_len,
                dtype=torch.long,
                device=model.device,
            )

            output = model.forward(
                **model_inputs,
                past_key_values=current_cache,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                use_cache=True,
            )

            logits = output.logits[:, -1, :].float()

            if repetition_penalty != 1.0 and generated_token_ids:
                current_logits = logits[0]
                for token_id in set(generated_token_ids):
                    if current_logits[token_id] < 0:
                        current_logits[token_id] *= repetition_penalty
                    else:
                        current_logits[token_id] /= repetition_penalty
                logits[0] = current_logits

            if i < ban_eos_steps:
                for eos_id in eos_token_ids:
                    logits[:, eos_id] = -float("inf")

            if do_sample:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)

                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    probs[indices_to_remove] = 0.0
                    probs = probs / probs.sum(dim=-1, keepdim=True)

                if top_k is not None and top_k > 0 and top_k < probs.shape[-1]:
                    topk_probs, topk_indices = torch.topk(probs, k=top_k, dim=-1)
                    filtered_probs = torch.zeros_like(probs)
                    filtered_probs.scatter_(1, topk_indices, topk_probs)
                    probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

                next_token_index = torch.multinomial(probs, num_samples=1)
            else:
                probs = F.softmax(logits, dim=-1)
                next_token_index = torch.argmax(logits, dim=-1, keepdim=True)

            token_item = int(next_token_index.item())

            if verbose and i < 5:
                top_probs, top_indices = torch.topk(probs[0], 5)
                top_tokens = [processor.tokenizer.decode([idx.item()]) for idx in top_indices]
                print(
                    f"  [GEN_MOLMO2] Step {i}: token={processor.tokenizer.decode([token_item])} | "
                    f"Top 5: {list(zip(top_tokens, [f'{p:.3f}' for p in top_probs.tolist()]))}"
                )

            generated_token_ids.append(token_item)
            decoded_sequence = processor.tokenizer.decode(
                generated_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            current_cache = output.past_key_values
            current_inputs_ids = next_token_index
            current_global_step += current_seq_len

            if token_item in eos_token_ids:
                break

    cache.global_token_counter = current_global_step
    return decoded_sequence


def _prime_cache_sw(model, processor, cache_cfg, prompt_path, verbose=True):
    if verbose:
        print("DEBUG: Starting Molmo2 Priming Phase...", flush=True)

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_content = f.read()

    priming_conversation = [
        {"role": "system", "content": [{"type": "text", "text": prompt_content}]},
        {"role": "user", "content": [{"type": "text", "text": "Video stream starts now:"}]},
    ]

    priming_input_ids = _tokenize_text_conversation(
        priming_conversation,
        processor,
        model.device,
        add_generation_prompt=False,
    )
    priming_embeds = _embed_tokens(model, priming_input_ids)

    global_token_counter = 0

    with torch.no_grad():
        priming_seq_len = priming_embeds.shape[1]
        token_budget = cache_cfg.get("mutable_token_size", 1024)

        sink_tokens_size = 0
        if cache_cfg.get("use_sink_tokens", False):
            sink_tokens_size = priming_seq_len if cache_cfg.get("sink_sys_prompt", False) else cache_cfg.get("size_sink_tokens", 128)

        image_seq_length = int(cache_cfg.get("molmo2_image_seq_length", 576))
        hat_tokens_size = 0
        if cache_cfg.get("use_hat_tokens", False):
            if cache_cfg.get("hat_category", "frames") == "frames":
                hat_tokens_size = cache_cfg.get("hat_num_frames", 1) * image_seq_length
            else:
                hat_tokens_size = cache_cfg.get("size_hat_tokens", 400)

        total_k_capacity = sink_tokens_size + token_budget + hat_tokens_size

        cache_instance = create_cache_from_config(
            config=model.config,
            cache_cfg=cache_cfg,
            K=total_k_capacity,
        )

        if hasattr(cache_instance, "set_sink_size"):
            cache_instance.set_sink_size(sink_tokens_size)

        if hasattr(cache_instance, "set_hat_size"):
            cache_instance.set_hat_size(hat_tokens_size)
            if verbose:
                print("settati gli hat tokens", flush=True)

        position_ids = torch.arange(
            0, priming_seq_len, dtype=torch.long, device=model.device
        ).unsqueeze(0)
        cache_position = torch.arange(
            0, priming_seq_len, dtype=torch.long, device=model.device
        )

        _ = _get_lm(model)(
            inputs_embeds=priming_embeds,
            past_key_values=cache_instance,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=True,
        )

        if hasattr(cache_instance, "capture_sink_guidance"):
            cache_instance.capture_sink_guidance()

        global_token_counter += priming_seq_len

    cache_instance.global_token_counter = global_token_counter

    if verbose:
        print(f"[DEBUG CACHE] cache_instance type: {type(cache_instance)}", flush=True)
        print(f"[DEBUG CACHE] total_k_capacity: {total_k_capacity}", flush=True)
        print(f"[DEBUG CACHE] sink_tokens_size: {sink_tokens_size}", flush=True)
        print(f"[DEBUG CACHE] hat_tokens_size: {hat_tokens_size}", flush=True)

    return cache_instance


def _append_frame_batches_to_existing_cache_sw(
    frame_batches,
    model,
    processor,
    cache_instance,
    size=(360, 240),
    position_mode="global",
    verbose=True,
):
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
                f"DEBUG: Molmo2 batch {processed_batches_count}: Processing {len(frames_batch)} frames as images.",
                flush=True,
            )

        with torch.no_grad():
            visual_inputs = _prepare_visual_inputs(
                frames_batch=frames_batch,
                processor=processor,
                model=model,
                size=size,
            )
            input_ids = visual_inputs.input_ids.to(model.device)
            image_patch_id = _image_patch_id(model, processor)
            visual_pos_mask = input_ids == image_patch_id
            frame_token_counts = _frame_token_counts_from_input_ids(input_ids, image_patch_id)
            current_seq_len = input_ids.shape[1]

            if int(sum(frame_token_counts)) != int(visual_pos_mask.sum().item()):
                raise ValueError(
                    f"Mismatch frame_token_counts={sum(frame_token_counts)} "
                    f"vs visual_slots={int(visual_pos_mask.sum().item())}"
                )

            frame_aware_pruning = False
            if hasattr(cache_instance, "cache_config"):
                cfg = cache_instance.cache_config
                frame_aware_pruning = bool(
                    cfg.get("intra_frame_pruning", False)
                    or cfg.get("inter_frame_pruning", False)
                    or _cache_has_storage_offload(cache_instance)
                )

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
            position_start = physical_past_len if position_mode == "compact" else global_token_counter
            position_ids = torch.arange(
                position_start,
                position_start + current_seq_len,
                dtype=torch.long,
                device=model.device,
            ).unsqueeze(0)
            cache_position = torch.arange(
                physical_past_len,
                physical_past_len + current_seq_len,
                dtype=torch.long,
                device=model.device,
            )

            forward_kwargs = _forward_kwargs_from_inputs(visual_inputs)
            forward_kwargs["input_ids"] = input_ids

            t_frame_start = time.perf_counter()

            _ = model.forward(
                **forward_kwargs,
                past_key_values=cache_instance,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
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
                    f"Processed Molmo2 Batch {processed_batches_count} | "
                    f"Frames: {len(frames_batch)} | "
                    f"Frame tokens: {frame_token_counts} | "
                    f"Cache Size: {current_cache_size} tokens",
                    flush=True,
                )

    if verbose:
        print("DEBUG: Finished Molmo2 Video Ingestion.", flush=True)

    cache_instance.global_token_counter = global_token_counter
    return cache_instance, frame_ingestion_times, cache_size_history


def _append_sampler_to_existing_cache_sw(
    sampler,
    model,
    processor,
    cache_instance,
    size=(360, 240),
    position_mode="global",
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
        position_mode=position_mode,
        verbose=verbose,
    )
