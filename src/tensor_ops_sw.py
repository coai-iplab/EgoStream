import torch

def process_video_frames_sw(frames_batch, processor, device, size):
    """
    Input: List of numpy frames (from sliding window sampler), processor, device, and target size.
    Output: Returns (pixel_values, grid_thw) tensors for VIDEO processing.
    
    This function processes frames as a video, not as individual images.
    The grid_thw will have temporal dimension set to len(frames_batch).
    """
    from .utils import resize_video_frame
    from PIL import Image
    
    # Process each frame and collect
    processed_frames = []
    for frame in frames_batch:
        pil_image = resize_video_frame(frame, size)
        processed_frames.append(pil_image)
    
    # Use the video processor (processes as temporal sequence)
    # According to Qwen3 VL docs, we should pass a list of PIL images as video
    inputs = processor.image_processor(images=processed_frames, return_tensors="pt")
    inputs = inputs.to(device)
    
    pixel_values = inputs.pixel_values.to(torch.bfloat16)
    grid_thw = inputs.image_grid_thw
    
    return pixel_values, grid_thw


def compute_video_embeddings_sw(pixel_values, model, grid_thw=None):
    """
    Input: Processed pixel values for video, model instance, and grid configuration.
    Output: Returns (vision_embeddings, num_vision_tokens) extracted from the vision encoder.
    
    This processes the frames as a cohesive video using the visual encoder.
    """
    vision_output = model.model.visual(pixel_values, grid_thw=grid_thw)
    
    if hasattr(vision_output, 'last_hidden_state'):
        vision_embeddings = vision_output.last_hidden_state
    elif isinstance(vision_output, tuple):
        vision_embeddings = vision_output[0]
    else:
        vision_embeddings = vision_output
    
    # Calculate total vision tokens
    if len(vision_embeddings.shape) == 3:
        num_vision_tokens = vision_embeddings.size(0) * vision_embeddings.size(1)
    elif len(vision_embeddings.shape) == 2:
        vision_embeddings = vision_embeddings.unsqueeze(0)
        num_vision_tokens = vision_embeddings.size(1)
    else:
        num_vision_tokens = vision_embeddings.size(1) if vision_embeddings.size(1) > 0 else 1
    
    return vision_embeddings, num_vision_tokens


def compute_projected_embeddings_sw(vision_embeddings):
    """
    Input: Raw vision embeddings from the encoder.
    Output: Returns projected embeddings reshaped/unsqueezed to match language model dim.
    """
    if len(vision_embeddings.shape) == 3:
        projected_vision_embeddings = vision_embeddings.view(1, -1, vision_embeddings.size(2))
    elif len(vision_embeddings.shape) == 2:
        projected_vision_embeddings = vision_embeddings.unsqueeze(0)
    else:
        projected_vision_embeddings = vision_embeddings
    return projected_vision_embeddings


def compute_text_embeddings_sw(input_ids, num_vision_tokens, processor, model):
    """
    Input: Tokenized input_ids, number of video tokens, processor, and model.
    Output: Returns (text_embeddings, expanded_input_ids) with video placeholders expanded.
    
    This function expands <|video_pad|> placeholders for video tokens.
    NOTE: Keep attention to RoPE - expanded_input_ids will be used for rope_deltas calculation.
    """
    # For Qwen3 VL, video uses the same placeholder token as images
    visual_token_id = processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    
    # Check how many video placeholders the template expects
    placeholder_indices = torch.where(input_ids[0] == visual_token_id)[0]
    
    if len(placeholder_indices) == 0:
        return model.model.language_model.embed_tokens(input_ids), input_ids

    # For video, we typically have one placeholder that gets expanded to all vision tokens
    # But we need to expand it to match num_vision_tokens
    tokens_per_placeholder = num_vision_tokens // len(placeholder_indices)

    # Reconstruct the sequence by expanding every placeholder found
    new_input_ids = []
    last_idx = 0
    for idx in placeholder_indices:
        new_input_ids.append(input_ids[0, last_idx:idx])
        new_input_ids.append(torch.full((tokens_per_placeholder,), visual_token_id, device=input_ids.device))
        last_idx = idx + 1
    new_input_ids.append(input_ids[0, last_idx:])
    
    expanded_input_ids = torch.cat(new_input_ids).unsqueeze(0)
    text_embeddings = model.model.language_model.embed_tokens(expanded_input_ids)
    
    return text_embeddings, expanded_input_ids


def replace_video_placeholder_sw(input_ids, text_embeddings, projected_vision_embeddings, processor):
    """
    Input: Expanded input_ids, text embeddings, projected video embeddings, and processor.
    Output: Returns final input embeddings with video tokens replaced by actual vision embeddings.
    """
    visual_token_id = processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    placeholder_indices = torch.where(input_ids[0] == visual_token_id)[0]
    
    if len(placeholder_indices) == 0:
        return text_embeddings

    # Create a copy to modify
    input_embeddings = text_embeddings.clone()
    
    # Replace all placeholder positions with the corresponding video embeddings
    # projected_vision_embeddings is [1, num_tokens, dim]
    # We need to ensure we're replacing the right number of positions
    num_vision_tokens = projected_vision_embeddings.shape[1]
    
    if len(placeholder_indices) != num_vision_tokens:
        # Take only as many as we have
        min_tokens = min(len(placeholder_indices), num_vision_tokens)
        input_embeddings[0, placeholder_indices[:min_tokens]] = projected_vision_embeddings[0, :min_tokens]
    else:
        input_embeddings[0, placeholder_indices] = projected_vision_embeddings[0]
    
    return input_embeddings

