import torch

def process_pil_image(pil_image, processor, device):
    """
    Input: PIL Image object, HuggingFace processor, and torch device (cpu/cuda).
    Output: Returns (pixel_values, image_sizes, grid_thw) tensors ready for model ingestion.
    """
    inputs = processor.image_processor(images=pil_image, return_tensors="pt")
    inputs = inputs.to(device)
    pixel_values = inputs.pixel_values.to(torch.bfloat16)
    grid_thw = inputs.image_grid_thw
    
    if hasattr(inputs, 'image_sizes'):
        image_sizes = inputs.image_sizes
    else:
        image_sizes = torch.tensor([[pil_image.height, pil_image.width]], device=device)
        
    return pixel_values, image_sizes, grid_thw
import torch

def _find_first_tensor(obj):
    if torch.is_tensor(obj):
        return obj

    if isinstance(obj, (list, tuple)):
        for item in obj:
            t = _find_first_tensor(item)
            if t is not None:
                return t

    return None


def _normalize_to_batched_seq(x: torch.Tensor) -> torch.Tensor:
    # Want [1, N, D]
    if x.ndim == 2:
        return x.unsqueeze(0)
    if x.ndim == 3:
        if x.shape[0] == 1:
            return x
        return x.reshape(1, -1, x.shape[-1])
    raise ValueError(f"Unexpected tensor shape: {tuple(x.shape)}")

def compute_visual_embeddings(frame_pixel_values, model, grid_thw=None):
    """
    Returns:
      - vision_output_raw: raw output from Qwen vision model
      - num_vision_tokens: visual token count from merged output if possible
    """
    vision_output = model.model.visual(frame_pixel_values, grid_thw=grid_thw)

    # Prefer official merged output
    merged = None
    if hasattr(vision_output, "pooler_output") and torch.is_tensor(vision_output.pooler_output):
        merged = vision_output.pooler_output
    else:
        # fallback: recursively find first tensor anywhere in the structure
        merged = _find_first_tensor(vision_output)

    if merged is None:
        raise ValueError(
            f"Could not extract a tensor from vision_output. type={type(vision_output)}"
        )

    merged = _normalize_to_batched_seq(merged)
    num_vision_tokens = int(merged.shape[1])

    return vision_output, num_vision_tokens

def compute_projected_embeddings(vision_output):
    """
    Returns:
      - projected_vision_embeddings: [1, N, D]
      - deepstack_visual_embeds: list of [1, N, D] or None
    """
    # Main visual branch: prefer official merged output
    if hasattr(vision_output, "pooler_output") and torch.is_tensor(vision_output.pooler_output):
        projected = vision_output.pooler_output
    else:
        projected = _find_first_tensor(vision_output)

    if projected is None:
        raise ValueError(
            f"Could not extract projected tensor from vision_output. type={type(vision_output)}"
        )

    projected = _normalize_to_batched_seq(projected)

    deepstack_visual_embeds = None
    if hasattr(vision_output, "deepstack_features") and vision_output.deepstack_features is not None:
        deepstack_visual_embeds = []
        for feat in vision_output.deepstack_features:
            if not torch.is_tensor(feat):
                feat = _find_first_tensor(feat)
            if feat is None:
                continue
            deepstack_visual_embeds.append(_normalize_to_batched_seq(feat))

    return projected, deepstack_visual_embeds


def process_text_conversation(conversation, processor, device):
    """
    Input: List of dicts representing conversation messages, processor, and device (cpu/cuda).
    Output: Returns tokenized input_ids tensor for the conversation prompts.
    """
    inputs = processor.apply_chat_template(
                conversation, 
                add_generation_prompt=True, 
                tokenize=True, 
                return_dict=True, 
                return_tensors="pt"
            )
    input_ids = inputs["input_ids"].to(device)
    return input_ids

def compute_text_embeddings(input_ids, num_vision_tokens, processor, model):
    visual_token_id = processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    
    # Check how many images the template expects
    placeholder_indices = torch.where(input_ids[0] == visual_token_id)[0]
    
    if len(placeholder_indices) == 0:
        return model.model.language_model.embed_tokens(input_ids), input_ids

    # We need to calculate how many tokens go into EACH placeholder. 
    # If total is 6720 and we have 15 images, each gets 448 tokens.
    tokens_per_image = num_vision_tokens // len(placeholder_indices)

    # Reconstruct the sequence by expanding every placeholder found
    new_input_ids = []
    last_idx = 0
    for idx in placeholder_indices:
        new_input_ids.append(input_ids[0, last_idx:idx])
        new_input_ids.append(torch.full((tokens_per_image,), visual_token_id, device=input_ids.device))
        last_idx = idx + 1
    new_input_ids.append(input_ids[0, last_idx:])
    
    expanded_input_ids = torch.cat(new_input_ids).unsqueeze(0)
    text_embeddings = model.model.language_model.embed_tokens(expanded_input_ids)
    
    return text_embeddings, expanded_input_ids

def replace_placeholder(input_ids, text_embeddings, projected_vision_embeddings, processor):
    visual_token_id = processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    visual_pos_mask = (input_ids == visual_token_id)

    num_slots = int(visual_pos_mask.sum().item())
    num_visual_tokens = int(projected_vision_embeddings.shape[1])

    if num_slots == 0:
        return text_embeddings, visual_pos_mask

    if num_slots != num_visual_tokens:
        raise ValueError(
            f"Placeholder/vision mismatch: slots={num_slots}, visual_tokens={num_visual_tokens}"
        )

    input_embeddings = text_embeddings.clone()
    input_embeddings[visual_pos_mask] = projected_vision_embeddings.reshape(
        -1, projected_vision_embeddings.shape[-1]
    ).to(input_embeddings.dtype)

    return input_embeddings, visual_pos_mask