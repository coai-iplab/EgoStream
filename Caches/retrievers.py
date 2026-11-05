from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gc
import torch
import torch.nn.functional as F


@dataclass
class _BlockCandidate:
    source: str  # "offload" or "live"
    score: float
    token_len: int
    frame_lengths: List[int]
    temporal_key: Tuple[int, int, int]
    chunk_idx: Optional[int] = None
    start: int = 0
    end: int = 0


def _flatten_keys_to_token_vectors(k: torch.Tensor) -> torch.Tensor:
    """[1,H,L,D] -> [L,H*D], normalized per token."""
    if k.dim() != 4 or k.shape[0] != 1:
        raise ValueError(f"Expected key tensor [1,H,L,D], got {tuple(k.shape)}")
    token_vecs = k[0].permute(1, 0, 2).reshape(k.shape[-2], -1).float()
    return F.normalize(token_vecs, dim=-1)


def _block_vector_from_keys(k: torch.Tensor) -> torch.Tensor:
    """Mean-key block vector, normalized."""
    token_vecs = _flatten_keys_to_token_vectors(k)
    if token_vecs.numel() == 0:
        raise ValueError("Empty block keys")
    return F.normalize(token_vecs.mean(dim=0), dim=-1)


def _score_block_against_query(k_block: torch.Tensor, query_vec: torch.Tensor) -> float:
    block_vec = _block_vector_from_keys(k_block)
    return float(torch.dot(block_vec, query_vec).item())


def _iter_frame_blocks(frame_lengths: Sequence[int], block_size_frames: int):
    """
    Yield blocks of consecutive frames.

    Returns tuples:
        token_start, token_end, block_frame_lengths, first_frame_idx
    """
    fls = [int(x) for x in frame_lengths if int(x) > 0]
    block_size_frames = max(1, int(block_size_frames))

    cursor = 0
    frame_idx = 0

    while frame_idx < len(fls):
        block_lengths = fls[frame_idx:frame_idx + block_size_frames]
        block_len = int(sum(block_lengths))
        start = cursor
        end = cursor + block_len

        yield start, end, list(block_lengths), frame_idx

        cursor = end
        frame_idx += block_size_frames


def _extract_keys_from_past(past_key_values, layer_idx: int = 0) -> torch.Tensor:
    """
    Extracts the keys of layer_idx from different past_key_values formats.
    """
    pkv = past_key_values

    if hasattr(pkv, "layers"):
        layer = pkv.layers[layer_idx]
        if hasattr(layer, "keys") and layer.keys is not None:
            return layer.keys
        if hasattr(layer, "key_cache") and layer.key_cache is not None:
            return layer.key_cache

    if hasattr(pkv, "key_cache"):
        return pkv.key_cache[layer_idx]

    if isinstance(pkv, (tuple, list)):
        layer = pkv[layer_idx]
        if isinstance(layer, (tuple, list)):
            return layer[0]
        if hasattr(layer, "keys"):
            return layer.keys

    raise ValueError(f"Unsupported past_key_values format: {type(pkv)}")


def _build_question_key_proxy(
    model,
    processor,
    question: str,
    device=None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Builds a query vector in key space using a text-only probe forward.

    It is a conservative ReKV-internal proxy:
        question_vector = mean(keys of the question/chat tokens)

    It does not modify the main video cache.
    """
    if device is None:
        device = model.device

    conversation = [
        {
            "role": "user",
            "content": [{"type": "text", "text": question}],
        }
    ]

    text_inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)

    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": True,
    }

    try:
        _, text_rope_deltas = model.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=None,
            video_grid_thw=None,
            attention_mask=attention_mask,
        )
        position_ids = torch.arange(
            0,
            input_ids.shape[1],
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        kwargs["position_ids"] = position_ids
        kwargs["rope_deltas"] = text_rope_deltas
    except Exception:
        # Some wrappers do not require explicit rope_deltas.
        pass

    with torch.no_grad():
        output = model.forward(**kwargs)

    q_keys = _extract_keys_from_past(output.past_key_values, layer_idx=0)
    if dtype is not None:
        q_keys = q_keys.to(dtype)

    token_vecs = _flatten_keys_to_token_vectors(q_keys)
    query_vec = F.normalize(token_vecs.mean(dim=0), dim=-1)

    try:
        del output, text_inputs, input_ids, attention_mask
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return query_vec


def _available_offload_chunk_indices(layer0, max_offloaded_chunks: Optional[int]) -> List[int]:
    store = getattr(layer0, "offload_store", None)
    if store is None:
        return []
    indices = list(range(len(getattr(store, "chunks", []))))
    if max_offloaded_chunks is not None:
        max_offloaded_chunks = int(max_offloaded_chunks)
        if max_offloaded_chunks > 0:
            indices = indices[-max_offloaded_chunks:]
    return indices


def _score_offloaded_blocks(
    layer0,
    query_vec: torch.Tensor,
    chunk_indices: Sequence[int],
    block_size_frames: int,
    dtype: torch.dtype,
    device,
    verbose: bool = False,
) -> List[_BlockCandidate]:
    store = getattr(layer0, "offload_store", None)
    if store is None:
        return []

    out: List[_BlockCandidate] = []

    for chunk_idx in chunk_indices:
        keys, _values, payload = store.load_chunk(
            chunk_idx=int(chunk_idx),
            dtype=dtype,
            device=device,
        )
        frame_lengths = [int(x) for x in payload.get("frame_lengths", []) if int(x) > 0]

        if not frame_lengths or keys is None or keys.numel() == 0:
            continue

        for start, end, block_lengths, frame_idx in _iter_frame_blocks(
            frame_lengths,
            block_size_frames=block_size_frames,
        ):
            if end <= start:
                continue
            k_block = keys[..., start:end, :]
            score = _score_block_against_query(k_block, query_vec)
            out.append(
                _BlockCandidate(
                    source="offload",
                    score=score,
                    token_len=int(end - start),
                    frame_lengths=block_lengths,
                    temporal_key=(0, int(chunk_idx), int(start)),
                    chunk_idx=int(chunk_idx),
                    start=int(start),
                    end=int(end),
                )
            )

        del keys, _values
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if verbose:
        print(f"[RETRIEVE][rekv] scored_offloaded_blocks={len(out)}", flush=True)

    return out


def _score_live_mutable_blocks(
    layer0,
    query_vec: torch.Tensor,
    block_size_frames: int,
    verbose: bool = False,
) -> List[_BlockCandidate]:
    if not hasattr(layer0, "get_mutable_for_retrieve"):
        return []

    mut_k, _mut_v, mutable_lengths = layer0.get_mutable_for_retrieve()
    if mut_k is None or mut_k.numel() == 0:
        return []

    out: List[_BlockCandidate] = []

    for start, end, block_lengths, frame_idx in _iter_frame_blocks(
        mutable_lengths,
        block_size_frames=block_size_frames,
    ):
        if end <= start:
            continue
        k_block = mut_k[..., start:end, :]
        score = _score_block_against_query(k_block, query_vec)
        out.append(
            _BlockCandidate(
                source="live",
                score=score,
                token_len=int(end - start),
                frame_lengths=block_lengths,
                temporal_key=(1, 0, int(start)),
                chunk_idx=None,
                start=int(start),
                end=int(end),
            )
        )

    if verbose:
        print(f"[RETRIEVE][rekv] scored_live_blocks={len(out)}", flush=True)

    return out


def _select_top_blocks_bounded(
    candidates: Sequence[_BlockCandidate],
    top_k_tokens: int,
) -> List[_BlockCandidate]:
    top_k_tokens = int(top_k_tokens)
    if top_k_tokens <= 0:
        return []

    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
    selected: List[_BlockCandidate] = []
    used = 0

    for cand in ranked:
        if cand.token_len <= 0:
            continue

        if used + cand.token_len > top_k_tokens:
            # We keep whole blocks. If no block has been selected yet,
            # we accept the best one even if it exceeds the budget.
            if not selected:
                selected.append(cand)
            continue

        selected.append(cand)
        used += cand.token_len

        if used >= top_k_tokens:
            break

    selected = sorted(selected, key=lambda c: c.temporal_key)
    return selected


def _rebuild_one_layer_from_candidates(
    layer,
    selected: Sequence[_BlockCandidate],
    dtype: torch.dtype,
    device,
) -> int:
    if not selected:
        return 0

    if not hasattr(layer, "rebuild_mutable_for_retrieve"):
        raise ValueError("FlexibleCacheLayer is missing rebuild_mutable_for_retrieve")

    live_mut_k, live_mut_v, _live_lengths = layer.get_mutable_for_retrieve()
    store = getattr(layer, "offload_store", None)

    k_parts = []
    v_parts = []
    new_frame_lengths: List[int] = []
    loaded_chunks: Dict[int, Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]] = {}

    try:
        for cand in selected:
            if cand.source == "live":
                k_piece = live_mut_k[..., cand.start:cand.end, :]
                v_piece = live_mut_v[..., cand.start:cand.end, :]
            elif cand.source == "offload":
                if store is None:
                    continue
                chunk_idx = int(cand.chunk_idx)
                if chunk_idx not in loaded_chunks:
                    loaded_chunks[chunk_idx] = store.load_chunk(
                        chunk_idx=chunk_idx,
                        dtype=dtype,
                        device=device,
                    )
                chunk_k, chunk_v, _payload = loaded_chunks[chunk_idx]
                k_piece = chunk_k[..., cand.start:cand.end, :]
                v_piece = chunk_v[..., cand.start:cand.end, :]
            else:
                raise ValueError(f"Unsupported candidate source: {cand.source}")

            if k_piece.numel() == 0:
                continue

            k_parts.append(k_piece)
            v_parts.append(v_piece)
            new_frame_lengths.extend([int(x) for x in cand.frame_lengths if int(x) > 0])

        if not k_parts:
            return 0

        new_mut_k = torch.cat(k_parts, dim=-2)
        new_mut_v = torch.cat(v_parts, dim=-2)

        return int(layer.rebuild_mutable_for_retrieve(
            new_mut_k=new_mut_k,
            new_mut_v=new_mut_v,
            new_mutable_frame_lengths=new_frame_lengths,
        ))
    finally:
        loaded_chunks.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def rekv(
    cache,
    model,
    processor,
    question: str,
    top_k_tokens: Optional[int] = None,
    block_size_frames: int = 1,
    include_live_mutable: bool = True,
    max_offloaded_chunks: Optional[int] = None,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    device=None,
    verbose: bool = False,
):
    """
    ReKV-like internal retrieval QA-time.

    Score:
        block_score = cosine(mean_key_block, mean_key_question_proxy)

    Candidates:
        offloaded mutable blocks + live mutable blocks

    Output:
        sink + selected_mutable_blocks_temporal_order + hat

    Sink and hat never take part in the selection and remain preserved by the layer.
    """
    if not hasattr(cache, "layers") or not cache.layers:
        return {"retrieved": False, "reason": "cache_has_no_layers"}

    layer0 = cache.layers[0]

    if top_k_tokens is None:
        top_k_tokens = int(getattr(layer0, "mutable_token_size", 8192))
    if device is None:
        device = getattr(model, "device", None)
    if dtype is None:
        dtype = getattr(layer0, "dtype", None) or torch.bfloat16

    query_vec = _build_question_key_proxy(
        model=model,
        processor=processor,
        question=question,
        device=device,
        dtype=dtype,
    ).to(device)

    chunk_indices = _available_offload_chunk_indices(
        layer0=layer0,
        max_offloaded_chunks=max_offloaded_chunks,
    )

    candidates: List[_BlockCandidate] = []

    candidates.extend(
        _score_offloaded_blocks(
            layer0=layer0,
            query_vec=query_vec,
            chunk_indices=chunk_indices,
            block_size_frames=block_size_frames,
            dtype=dtype,
            device=device,
            verbose=verbose,
        )
    )

    if include_live_mutable:
        candidates.extend(
            _score_live_mutable_blocks(
                layer0=layer0,
                query_vec=query_vec,
                block_size_frames=block_size_frames,
                verbose=verbose,
            )
        )

    if not candidates:
        return {
            "retrieved": False,
            "reason": "no_candidates",
            "live_seq_len_after_retrieve": int(cache.get_seq_length()),
        }

    selected = _select_top_blocks_bounded(
        candidates=candidates,
        top_k_tokens=int(top_k_tokens),
    )

    if not selected:
        return {
            "retrieved": False,
            "reason": "no_selected_blocks",
            "num_candidates": int(len(candidates)),
            "live_seq_len_after_retrieve": int(cache.get_seq_length()),
        }

    selected_tokens = int(sum(c.token_len for c in selected))
    selected_offload_blocks = int(sum(1 for c in selected if c.source == "offload"))
    selected_live_blocks = int(sum(1 for c in selected if c.source == "live"))

    # Prevents the rebuilt cache from being immediately compressed/offloaded.
    if hasattr(cache, "set_QA"):
        cache.set_QA(True)

    per_layer_tokens: Dict[str, int] = {}
    for idx, layer in enumerate(cache.layers):
        inserted = _rebuild_one_layer_from_candidates(
            layer=layer,
            selected=selected,
            dtype=dtype,
            device=device,
        )
        per_layer_tokens[str(idx)] = int(inserted)

    stats = {
        "retrieved": True,
        "method": "rekv",
        "score": "cosine(mean_key_block, mean_key_question_proxy)",
        "num_candidates": int(len(candidates)),
        "num_selected_blocks": int(len(selected)),
        "selected_offload_blocks": selected_offload_blocks,
        "selected_live_blocks": selected_live_blocks,
        "selected_tokens_per_layer": selected_tokens,
        "selected_total_layer_tokens": int(selected_tokens * len(cache.layers)),
        "top_k_tokens": int(top_k_tokens),
        "block_size_frames": int(block_size_frames),
        "include_live_mutable": bool(include_live_mutable),
        "max_offloaded_chunks": None if max_offloaded_chunks is None else int(max_offloaded_chunks),
        "scored_offloaded_chunks": int(len(chunk_indices)),
        "live_seq_len_after_retrieve": int(cache.get_seq_length()),
        "per_layer_inserted_tokens": per_layer_tokens,
    }

    if verbose:
        print("[RETRIEVE][rekv]", stats, flush=True)

    return stats
