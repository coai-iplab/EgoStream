import torch
import torch.nn.functional as F

def sampling_generation_function_sw(
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
    verbose=True
):
    decoded_sequence = ""
    current_inputs_ids = inputs["input_ids"]
    current_cache = cache
    generated_token_ids = []
    temperature = max(0.1, float(temperature))

    if hasattr(cache, "global_token_counter"):
        current_global_step = cache.global_token_counter
    else:
        current_global_step = cache.get_seq_length()

    im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_token_id = processor.tokenizer.eos_token_id

    if verbose:
        print(f"[GEN_SW] Starting generation with cache size: {current_cache.get_seq_length()} tokens")
        print(f"[GEN_SW] Question length: {current_inputs_ids.shape[1]} tokens")
        print(f"[GEN_SW] Temperature: {temperature}, top_p: {top_p}, top_k: {top_k}")

    first_iteration = True
    question_embeds = model.model.language_model.embed_tokens(current_inputs_ids)

    for i in range(max_new_tokens):
        with torch.no_grad():
            physical_past_len = current_cache.get_seq_length()

            if first_iteration:
                current_seq_len = question_embeds.shape[1]
                attention_mask = torch.ones((1, physical_past_len + current_seq_len), device=model.device)
                position_ids = torch.arange(
                    current_global_step,
                    current_global_step + current_seq_len,
                    device=model.device
                ).unsqueeze(0)

                output = model.forward(
                    inputs_embeds=question_embeds,
                    past_key_values=current_cache,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=True
                )
                first_iteration = False
            else:
                current_seq_len = current_inputs_ids.shape[1]
                attention_mask = torch.ones((1, physical_past_len + current_seq_len), device=model.device)
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
                    use_cache=True
                )

            logits = output.logits[:, -1, :].float()

            if repetition_penalty != 1.0 and len(generated_token_ids) > 0:
                current_logits = logits[0]
                for token_id in set(generated_token_ids):
                    if current_logits[token_id] < 0:
                        current_logits[token_id] *= repetition_penalty
                    else:
                        current_logits[token_id] /= repetition_penalty
                logits[0] = current_logits

            if i < ban_eos_steps:
                logits[:, eos_token_id] = -float("inf")
                if im_end_id is not None:
                    logits[:, im_end_id] = -float("inf")

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

                if probs.sum(dim=-1, keepdim=True).eq(0).any():
                    probs = F.softmax(logits, dim=-1)
                else:
                    probs = probs / probs.sum(dim=-1, keepdim=True)

            if top_k is not None and top_k > 0 and top_k < probs.shape[-1]:
                topk_probs, topk_indices = torch.topk(probs, k=top_k, dim=-1)
                filtered_probs = torch.zeros_like(probs)
                filtered_probs.scatter_(1, topk_indices, topk_probs)
                probs = filtered_probs

                if probs.sum(dim=-1, keepdim=True).eq(0).any():
                    probs = F.softmax(logits, dim=-1)
                else:
                    probs = probs / probs.sum(dim=-1, keepdim=True)

            next_token_index = torch.multinomial(probs, num_samples=1)
            token_item = next_token_index.item()

            if verbose and i < 5:
                top_probs, top_indices = torch.topk(probs[0], 5)
                top_tokens = [processor.tokenizer.decode([idx.item()]) for idx in top_indices]
                print(f"  [GEN_SW] Step {i}: token={processor.tokenizer.decode([token_item])} | Top 5: {list(zip(top_tokens, [f'{p:.3f}' for p in top_probs.tolist()]))}")

            decoded_token = processor.tokenizer.decode([token_item], skip_special_tokens=False)
            decoded_sequence += decoded_token
            generated_token_ids.append(token_item)

            current_cache = output.past_key_values
            current_inputs_ids = next_token_index
            current_global_step += current_seq_len

            if token_item == eos_token_id or token_item == im_end_id:
                if verbose:
                    print(f"[GEN_SW] Generation ended at step {i} with EOS token")
                break

    if hasattr(cache, "global_token_counter"):
        cache.global_token_counter = current_global_step

    if verbose:
        print(f"[GEN_SW] Generated {len(generated_token_ids)} tokens: {decoded_sequence.strip()}")

    return decoded_sequence