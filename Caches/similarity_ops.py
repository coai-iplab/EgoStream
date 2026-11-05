from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F


@dataclass
class InterPairMeta:
    """
    Descrive una coppia inter-frame source -> reference.
    Il source è sempre mutable e il reference può essere mutable o hat.
    Gli offset source sono sempre riferiti alla mutable.
    """
    src_frame_pos: int
    src_start_mut: int
    src_start_all: int
    src_len: int
    ref_start_all: int
    ref_len: int
    ref_region: str


def flatten_token_vectors(k_block: torch.Tensor) -> torch.Tensor:
    """
    Appiattisce un blocco [B,H,L,D] in vettori token [L,H*D].
    Normalizza i vettori per rendere i cosine score più stabili.
    Ritorna un tensore 2D pronto per scoring single-chunk.
    """
    token_vecs = k_block[0].permute(1, 0, 2).reshape(k_block.shape[-2], -1)
    return F.normalize(token_vecs, dim=-1)


def flatten_token_vectors_batched(k_blocks: torch.Tensor) -> torch.Tensor:
    """
    Appiattisce un batch [N,H,L,D] in [N,L,H*D].
    Normalizza lungo l'ultima dimensione per confronti cosine affidabili.
    Ritorna una rappresentazione batchata token-wise.
    """
    token_vecs = k_blocks.permute(0, 2, 1, 3).reshape(
        k_blocks.shape[0], k_blocks.shape[2], -1
    )
    return F.normalize(token_vecs, dim=-1)


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Applica una softmax ignorando le posizioni invalide.
    Le posizioni fuori mask vengono azzerate sia prima sia dopo la softmax.
    Ritorna probabilità normalizzate solo sulle entry valide.
    """
    very_neg = torch.finfo(logits.dtype).min
    masked_logits = logits.masked_fill(~mask, very_neg)
    probs = torch.softmax(masked_logits, dim=dim)
    probs = probs.masked_fill(~mask, 0.0)
    denom = probs.sum(dim=dim, keepdim=True).clamp_min(1e-12)
    return probs / denom


def compute_adjacent_pair_scores_single(
    k_block: torch.Tensor,
    strategy: str,
    similarity_keep_old: bool,
) -> torch.Tensor:
    """
    Calcola score pair-wise adiacenti dentro un singolo chunk/frame.
    Supporta cosine e pseudo-attention nelle stesse convenzioni della cache attuale.
    Ritorna score di forma [L-1] dove ogni elemento corrisponde a una coppia.
    """
    strategy = str(strategy).lower()

    if k_block.shape[-2] <= 1:
        return torch.empty(0, device=k_block.device, dtype=torch.float32)

    # FAST PATH: adjacent cosine.
    # Importante: deve stare PRIMA di flatten_token_vectors(),
    # altrimenti paghi comunque il costo grosso.
    if strategy in ("adiacent_cosine", "adjacent_cosine"):
        x0 = k_block[0, :, :-1, :]
        x1 = k_block[0, :, 1:, :]

        dot = (x0 * x1).sum(dim=(0, 2), dtype=torch.float32)
        n0 = (x0 * x0).sum(dim=(0, 2), dtype=torch.float32).clamp_min(1e-12).sqrt()
        n1 = (x1 * x1).sum(dim=(0, 2), dtype=torch.float32).clamp_min(1e-12).sqrt()

        return dot / (n0 * n1)

    # Slow path solo per pseudo-attention.
    token_vecs = flatten_token_vectors(k_block)

    dim = token_vecs.shape[-1]
    context = F.normalize(token_vecs.mean(dim=0, keepdim=True), dim=-1)
    attn_logits = torch.matmul(token_vecs, context.transpose(0, 1)).squeeze(-1) / (dim ** 0.5)
    token_importance = F.softmax(attn_logits, dim=0)

    if strategy in ("pseudo_attention", "pseudo_attn"):
        if similarity_keep_old:
            return -token_importance[1:]
        return -token_importance[:-1]

    if strategy in ("pseudo_attention_meanpair", "pseudo_attn_meanpair"):
        return -(token_importance[:-1] + token_importance[1:]) / 2.0

    raise ValueError(f"Unsupported adjacent pair strategy: {strategy}")

def compute_adjacent_pair_scores_batched(
    k_blocks: torch.Tensor,
    valid_token_mask: torch.Tensor,
    strategy: str,
    similarity_keep_old: bool,
) -> torch.Tensor:
    """
    Calcola score pair-wise adiacenti in forma batchata su più frame/chunk.
    Le posizioni invalide vengono marcate a -inf così non possono essere selezionate.
    Ritorna un tensore [N,L-1] allineato al packer intra-frame.
    """
    strategy = str(strategy).lower()

    if k_blocks.shape[2] <= 1:
        return torch.empty(
            (k_blocks.shape[0], 0),
            device=k_blocks.device,
            dtype=torch.float32,
        )

    # FAST PATH: adjacent cosine batchato.
    # Evita flatten_token_vectors_batched() e F.normalize().
    if strategy in ("adiacent_cosine", "adjacent_cosine"):
        x0 = k_blocks[:, :, :-1, :]
        x1 = k_blocks[:, :, 1:, :]

        dot = (x0 * x1).sum(dim=(1, 3), dtype=torch.float32)
        n0 = (x0 * x0).sum(dim=(1, 3), dtype=torch.float32).clamp_min(1e-12).sqrt()
        n1 = (x1 * x1).sum(dim=(1, 3), dtype=torch.float32).clamp_min(1e-12).sqrt()

        scores = dot / (n0 * n1)

        valid_pair_mask = valid_token_mask[:, :-1] & valid_token_mask[:, 1:]
        return scores.masked_fill(~valid_pair_mask, float("-inf"))

    # Slow path solo per pseudo-attention.
    token_vecs = flatten_token_vectors_batched(k_blocks)

    dim = token_vecs.shape[-1]
    valid_f = valid_token_mask.unsqueeze(-1).to(token_vecs.dtype)
    context = (token_vecs * valid_f).sum(dim=1) / valid_f.sum(dim=1).clamp_min(1.0)
    context = F.normalize(context, dim=-1)

    attn_logits = torch.sum(token_vecs * context.unsqueeze(1), dim=-1) / (dim ** 0.5)
    token_importance = masked_softmax(attn_logits, valid_token_mask, dim=1)

    if strategy in ("pseudo_attention", "pseudo_attn"):
        if similarity_keep_old:
            scores = -token_importance[:, 1:]
        else:
            scores = -token_importance[:, :-1]
    elif strategy in ("pseudo_attention_meanpair", "pseudo_attn_meanpair"):
        scores = -(token_importance[:, :-1] + token_importance[:, 1:]) / 2.0
    else:
        raise ValueError(f"Unsupported adjacent pair strategy: {strategy}")

    valid_pair_mask = valid_token_mask[:, :-1] & valid_token_mask[:, 1:]
    return scores.masked_fill(~valid_pair_mask, float("-inf"))


def build_intra_frame_batch(
    mut_k: torch.Tensor,
    frame_lengths,
):
    """
    Costruisce un pack batchato dei frame mutable per scoring intra-frame.
    Scarta i frame con lunghezza <= 1 perché non hanno coppie adiacenti valide.
    Ritorna k_blocks, valid_mask, frame_offsets e frame_indices originali.
    """
    frame_lengths = [int(x) for x in frame_lengths if int(x) > 0]
    valid_frame_lengths = [int(x) for x in frame_lengths if int(x) > 1]

    if not valid_frame_lengths:
        return None, None, None, None

    device = mut_k.device
    dtype = mut_k.dtype
    n_frames = len(valid_frame_lengths)
    max_len = max(valid_frame_lengths)
    _, n_heads, _, head_dim = mut_k.shape

    k_blocks = torch.zeros(
        (n_frames, n_heads, max_len, head_dim),
        device=device,
        dtype=dtype,
    )
    valid_token_mask = torch.zeros(
        (n_frames, max_len),
        device=device,
        dtype=torch.bool,
    )

    frame_offsets = []
    frame_indices = []
    cursor = 0

    for original_idx, frame_len in enumerate(frame_lengths):
        frame_len = int(frame_len)

        if frame_len <= 1:
            cursor += max(frame_len, 0)
            continue

        batched_idx = len(frame_offsets)
        local_k = mut_k[0, :, cursor:cursor + frame_len, :]

        k_blocks[batched_idx, :, :frame_len, :] = local_k
        valid_token_mask[batched_idx, :frame_len] = True

        frame_offsets.append(cursor)
        frame_indices.append(original_idx)
        cursor += frame_len

    return k_blocks, valid_token_mask, frame_offsets, frame_indices

def select_topk_non_overlapping_pairs(pair_scores: torch.Tensor, k: int) -> torch.Tensor:
    """
    Selezione veloce approssimata per coppie adiacenti non-overlap.

    Idea:
    - tutte le coppie pari non si sovrappongono tra loro
    - tutte le coppie dispari non si sovrappongono tra loro
    - scegliamo il lato, pari o dispari, con somma score migliore

    Molto più veloce del greedy Python.
    """
    k = int(k)

    if k <= 0 or pair_scores.numel() == 0:
        return torch.tensor([], device=pair_scores.device, dtype=torch.long)

    even_scores = pair_scores[0::2]
    odd_scores = pair_scores[1::2]

    ke = min(k, even_scores.numel())
    ko = min(k, odd_scores.numel())

    if ke > 0:
        even_idx = torch.topk(even_scores, k=ke, largest=True).indices * 2
        even_sum = pair_scores[even_idx].sum()
    else:
        even_idx = torch.empty(0, device=pair_scores.device, dtype=torch.long)
        even_sum = torch.tensor(float("-inf"), device=pair_scores.device)

    if ko > 0:
        odd_idx = torch.topk(odd_scores, k=ko, largest=True).indices * 2 + 1
        odd_sum = pair_scores[odd_idx].sum()
    else:
        odd_idx = torch.empty(0, device=pair_scores.device, dtype=torch.long)
        odd_sum = torch.tensor(float("-inf"), device=pair_scores.device)

    selected = even_idx if even_sum >= odd_sum else odd_idx
    return torch.sort(selected).values.to(torch.long)


def _frame_offsets_from_lengths(frame_lengths):
    offsets = []
    cursor = 0
    for fl in frame_lengths:
        fl = int(fl)
        offsets.append((cursor, cursor + fl))
        cursor += fl
    return offsets

def pack_intra_frame_chunk(
    mut_k: torch.Tensor,
    frame_lengths,
    frame_positions,
    offsets=None,
):
    frame_lengths = [int(x) for x in frame_lengths if int(x) > 0]
    frame_positions = [int(x) for x in frame_positions]

    if offsets is None:
        offsets = _frame_offsets_from_lengths(frame_lengths)

    valid_positions = [p for p in frame_positions if int(frame_lengths[p]) > 1]
    if not valid_positions:
        return None, None, None, None

    device = mut_k.device
    dtype = mut_k.dtype
    chunk_lengths = [int(frame_lengths[p]) for p in valid_positions]
    n_frames = len(valid_positions)
    max_len = max(chunk_lengths)
    _, n_heads, _, head_dim = mut_k.shape

    k_blocks = torch.zeros(
        (n_frames, n_heads, max_len, head_dim),
        device=device,
        dtype=dtype,
    )
    valid_token_mask = torch.zeros(
        (n_frames, max_len),
        device=device,
        dtype=torch.bool,
    )

    frame_offsets = []

    for row_idx, frame_pos in enumerate(valid_positions):
        start, end = offsets[frame_pos]
        frame_len = end - start

        k_blocks[row_idx, :, :frame_len, :] = mut_k[0, :, start:end, :]
        valid_token_mask[row_idx, :frame_len] = True
        frame_offsets.append(int(start))

    return k_blocks, valid_token_mask, frame_offsets, valid_positions

def compute_intra_pair_indices_parallel(
    mut_k: torch.Tensor,
    frame_lengths,
    to_remove: int,
    strategy: str,
    similarity_keep_old: bool,
    chunk_size: int = 64,
) -> torch.Tensor:
    frame_lengths = [int(x) for x in frame_lengths if int(x) > 0]
    to_remove = int(to_remove)
    chunk_size = int(chunk_size)

    if to_remove <= 0 or not frame_lengths or mut_k.shape[-2] <= 1:
        return torch.tensor([], device=mut_k.device, dtype=torch.long)

    valid_frame_positions = [i for i, fl in enumerate(frame_lengths) if int(fl) > 1]
    if not valid_frame_positions:
        return torch.tensor([], device=mut_k.device, dtype=torch.long)

    offsets = _frame_offsets_from_lengths(frame_lengths)

    candidate_scores = []
    candidate_global_indices = []
    candidate_frame_positions = []
    candidate_local_pair_ids = []

    for start in range(0, len(valid_frame_positions), chunk_size):
        end = min(start + chunk_size, len(valid_frame_positions))
        local_positions = valid_frame_positions[start:end]

        packed = pack_intra_frame_chunk(
            mut_k=mut_k,
            frame_lengths=frame_lengths,
            frame_positions=local_positions,
            offsets=offsets,
        )
        if packed[0] is None:
            continue

        k_blocks, valid_token_mask, frame_offsets, chunk_frame_positions = packed

        pair_scores = compute_adjacent_pair_scores_batched(
            k_blocks=k_blocks,
            valid_token_mask=valid_token_mask,
            strategy=strategy,
            similarity_keep_old=similarity_keep_old,
        )

        valid_pair_mask = torch.isfinite(pair_scores)
        if not valid_pair_mask.any():
            continue

        n_frames, max_pairs = pair_scores.shape

        row_ids = (
            torch.arange(n_frames, device=mut_k.device)
            .unsqueeze(1)
            .expand(n_frames, max_pairs)
        )
        local_pair_ids = (
            torch.arange(max_pairs, device=mut_k.device)
            .unsqueeze(0)
            .expand(n_frames, max_pairs)
        )

        flat_scores = pair_scores[valid_pair_mask]
        flat_row_ids = row_ids[valid_pair_mask]
        flat_local_pair_ids = local_pair_ids[valid_pair_mask]

        if flat_scores.numel() == 0:
            continue

        frame_offsets_tensor = torch.tensor(frame_offsets, device=mut_k.device, dtype=torch.long)
        frame_positions_tensor = torch.tensor(chunk_frame_positions, device=mut_k.device, dtype=torch.long)

        flat_global_indices = frame_offsets_tensor[flat_row_ids] + flat_local_pair_ids
        flat_frame_positions = frame_positions_tensor[flat_row_ids]

        candidate_scores.append(flat_scores)
        candidate_global_indices.append(flat_global_indices)
        candidate_frame_positions.append(flat_frame_positions)
        candidate_local_pair_ids.append(flat_local_pair_ids)

    if not candidate_scores:
        return torch.tensor([], device=mut_k.device, dtype=torch.long)

    candidate_scores = torch.cat(candidate_scores, dim=0)
    candidate_global_indices = torch.cat(candidate_global_indices, dim=0)
    candidate_frame_positions = torch.cat(candidate_frame_positions, dim=0)
    candidate_local_pair_ids = torch.cat(candidate_local_pair_ids, dim=0)

    cand = min(max(to_remove * 8, to_remove), candidate_scores.numel())
    top = torch.topk(candidate_scores, cand)

    selected = []
    used_local_by_frame = {}

    for idx_in_all in top.indices.tolist():
        frame_pos = int(candidate_frame_positions[idx_in_all].item())
        local_idx = int(candidate_local_pair_ids[idx_in_all].item())

        used = used_local_by_frame.setdefault(frame_pos, set())
        if local_idx in used or (local_idx - 1) in used or (local_idx + 1) in used:
            continue

        used.add(local_idx)
        selected.append(int(candidate_global_indices[idx_in_all].item()))

        if len(selected) == to_remove:
            break

    if not selected:
        return torch.tensor([], device=mut_k.device, dtype=torch.long)

    return torch.tensor(sorted(selected), device=mut_k.device, dtype=torch.long)

def build_inter_pair_meta(
    mutable_frame_lengths,
    hat_frame_lengths,
) -> List[InterPairMeta]:
    """
    Costruisce il metadata delle coppie inter-frame source -> reference.
    Le coppie sono mutable[i]->mutable[i+1] più last_mutable->first_hat se esiste.
    Ritorna una lista ordinata di descrittori leggeri senza allocare tensoroni.
    """
    mutable_frame_lengths = [int(x) for x in mutable_frame_lengths if int(x) > 0]
    hat_frame_lengths = [int(x) for x in hat_frame_lengths if int(x) > 0]

    if len(mutable_frame_lengths) == 0:
        return []

    all_lengths = mutable_frame_lengths + hat_frame_lengths
    num_mutable = len(mutable_frame_lengths)

    if len(all_lengths) <= 1:
        return []

    offsets = []
    cursor = 0
    for frame_len in all_lengths:
        offsets.append((cursor, cursor + int(frame_len)))
        cursor += int(frame_len)

    pair_meta: List[InterPairMeta] = []

    for src_pos in range(num_mutable):
        ref_pos = src_pos + 1
        if ref_pos >= len(all_lengths):
            continue

        src_start_all, src_end_all = offsets[src_pos]
        ref_start_all, ref_end_all = offsets[ref_pos]

        src_len = src_end_all - src_start_all
        ref_len = ref_end_all - ref_start_all

        if src_len <= 0 or ref_len <= 0:
            continue

        pair_meta.append(
            InterPairMeta(
                src_frame_pos=int(src_pos),
                src_start_mut=int(src_start_all),
                src_start_all=int(src_start_all),
                src_len=int(src_len),
                ref_start_all=int(ref_start_all),
                ref_len=int(ref_len),
                ref_region="mutable" if ref_pos < num_mutable else "hat",
            )
        )

    return pair_meta


def pack_inter_pair_chunk(
    mut_k: torch.Tensor,
    hat_k: torch.Tensor,
    pair_meta_chunk: List[InterPairMeta],
):
    """
    Materializza un piccolo chunk di coppie inter-frame in tensori padded.
    Usa mut_k + hat_k come spazio combined ma mantiene source offsets riferiti alla mutable.
    Ritorna old_blocks/new_blocks, mask e metadata di mapping per la selezione.
    """
    if not pair_meta_chunk:
        return None, None, None, None, None, None

    combined_parts = [mut_k]
    if hat_k is not None and hat_k.numel() > 0:
        combined_parts.append(hat_k)
    combined_k = torch.cat(combined_parts, dim=-2)

    device = combined_k.device
    dtype = combined_k.dtype
    _, n_heads, _, head_dim = combined_k.shape

    max_old_len = max(x.src_len for x in pair_meta_chunk)
    max_new_len = max(x.ref_len for x in pair_meta_chunk)
    n_pairs = len(pair_meta_chunk)

    old_blocks = torch.zeros((n_pairs, n_heads, max_old_len, head_dim), device=device, dtype=dtype)
    new_blocks = torch.zeros((n_pairs, n_heads, max_new_len, head_dim), device=device, dtype=dtype)
    valid_old_mask = torch.zeros((n_pairs, max_old_len), device=device, dtype=torch.bool)
    valid_new_mask = torch.zeros((n_pairs, max_new_len), device=device, dtype=torch.bool)

    src_offsets = []
    src_lengths = []

    for pair_idx, meta in enumerate(pair_meta_chunk):
        old_blocks[pair_idx, :, :meta.src_len, :] = combined_k[0, :, meta.src_start_all:meta.src_start_all + meta.src_len, :]
        new_blocks[pair_idx, :, :meta.ref_len, :] = combined_k[0, :, meta.ref_start_all:meta.ref_start_all + meta.ref_len, :]

        valid_old_mask[pair_idx, :meta.src_len] = True
        valid_new_mask[pair_idx, :meta.ref_len] = True

        src_offsets.append(int(meta.src_start_mut))
        src_lengths.append(int(meta.src_len))

    return old_blocks, new_blocks, valid_old_mask, valid_new_mask, src_offsets, src_lengths


def compute_inter_token_scores_batched(
    old_blocks: torch.Tensor,
    new_blocks: torch.Tensor,
    valid_old_mask: torch.Tensor,
    valid_new_mask: torch.Tensor,
    strategy: str,
) -> torch.Tensor:
    """
    Calcola score token-wise del source frame rispetto al reference frame.
    Supporta max-cosine e context-attention per il caso inter-frame.
    Ritorna score di forma [Npairs, Lold].
    """
    strategy = str(strategy).lower()

    old_vecs = flatten_token_vectors_batched(old_blocks)
    new_vecs = flatten_token_vectors_batched(new_blocks)

    if strategy in ("adiacent_cosine", "adjacent_cosine"):
        sim = torch.matmul(old_vecs, new_vecs.transpose(1, 2))
        valid_sim_mask = valid_old_mask.unsqueeze(-1) & valid_new_mask.unsqueeze(1)
        sim = sim.masked_fill(~valid_sim_mask, float("-inf"))

        token_scores = sim.max(dim=-1).values
        return token_scores.masked_fill(~valid_old_mask, float("-inf"))

    if strategy in (
        "pseudo_attention",
        "pseudo_attn",
        "pseudo_attention_meanpair",
        "pseudo_attn_meanpair",
    ):
        dim = old_vecs.shape[-1]
        valid_new_f = valid_new_mask.unsqueeze(-1).to(new_vecs.dtype)
        next_context = (new_vecs * valid_new_f).sum(dim=1) / valid_new_f.sum(dim=1).clamp_min(1.0)
        next_context = F.normalize(next_context, dim=-1)

        logits = torch.sum(old_vecs * next_context.unsqueeze(1), dim=-1) / (dim ** 0.5)
        return logits.masked_fill(~valid_old_mask, float("-inf"))

    raise ValueError(f"Unsupported inter-frame strategy: {strategy}")


def collect_inter_token_candidates_chunked(
    mut_k: torch.Tensor,
    hat_k: torch.Tensor,
    mutable_frame_lengths,
    hat_frame_lengths,
    strategy: str,
    chunk_size: int,
):
    """
    Score-a le coppie inter-frame a chunk per evitare picchi di memoria.
    Accumula candidati globali senza costruire in anticipo tutti i tensoroni.
    Ritorna score, indici globali, pair_ids e src_lengths per la selezione finale.
    """
    pair_meta = build_inter_pair_meta(
        mutable_frame_lengths=mutable_frame_lengths,
        hat_frame_lengths=hat_frame_lengths,
    )

    if not pair_meta:
        empty = torch.tensor([], device=mut_k.device, dtype=torch.long)
        empty_f = torch.tensor([], device=mut_k.device, dtype=torch.float32)
        return empty_f, empty, empty, []

    chunk_size = int(chunk_size)
    candidate_scores = []
    candidate_global_indices = []
    candidate_pair_ids = []

    for start in range(0, len(pair_meta), chunk_size):
        end = min(start + chunk_size, len(pair_meta))
        local_meta = pair_meta[start:end]

        packed = pack_inter_pair_chunk(
            mut_k=mut_k,
            hat_k=hat_k,
            pair_meta_chunk=local_meta,
        )
        if packed[0] is None:
            continue

        old_blocks, new_blocks, valid_old_mask, valid_new_mask, src_offsets, _ = packed

        token_scores = compute_inter_token_scores_batched(
            old_blocks=old_blocks,
            new_blocks=new_blocks,
            valid_old_mask=valid_old_mask,
            valid_new_mask=valid_new_mask,
            strategy=strategy,
        )

        valid_mask = torch.isfinite(token_scores)
        if not valid_mask.any():
            continue

        chunk_pairs, max_old_len = token_scores.shape
        pair_ids = (
            torch.arange(start, end, device=mut_k.device)
            .unsqueeze(1)
            .expand(chunk_pairs, max_old_len)
        )
        local_tok_ids = (
            torch.arange(max_old_len, device=mut_k.device)
            .unsqueeze(0)
            .expand(chunk_pairs, max_old_len)
        )

        flat_scores = token_scores[valid_mask]
        flat_pair_ids = pair_ids[valid_mask]
        flat_local_tok_ids = local_tok_ids[valid_mask]

        if flat_scores.numel() == 0:
            continue

        src_offsets_tensor = torch.tensor(src_offsets, device=mut_k.device, dtype=torch.long)
        flat_global_indices = src_offsets_tensor[(flat_pair_ids - start).to(torch.long)] + flat_local_tok_ids

        candidate_scores.append(flat_scores)
        candidate_global_indices.append(flat_global_indices)
        candidate_pair_ids.append(flat_pair_ids)

    if not candidate_scores:
        empty = torch.tensor([], device=mut_k.device, dtype=torch.long)
        empty_f = torch.tensor([], device=mut_k.device, dtype=torch.float32)
        return empty_f, empty, empty, [x.src_len for x in pair_meta]

    return (
        torch.cat(candidate_scores, dim=0),
        torch.cat(candidate_global_indices, dim=0),
        torch.cat(candidate_pair_ids, dim=0),
        [x.src_len for x in pair_meta],
    )


def select_topk_tokens_per_source(
    candidate_scores: torch.Tensor,
    candidate_global_indices: torch.Tensor,
    candidate_pair_ids: torch.Tensor,
    src_lengths,
    k: int,
) -> torch.Tensor:
    """
    Seleziona i migliori token globali imponendo un budget per frame source.
    Mantiene almeno un token nel frame source quindi non consuma mai tutto il frame.
    Ritorna indici token-wise riferiti alla mutable.
    """
    k = int(k)

    if k <= 0 or candidate_scores.numel() == 0:
        return torch.tensor([], device=candidate_scores.device, dtype=torch.long)

    cand = min(max(k * 8, k), candidate_scores.numel())
    top = torch.topk(candidate_scores, cand)

    selected = []
    selected_per_pair = {i: 0 for i in range(len(src_lengths))}

    for idx_in_all in top.indices.tolist():
        pair_bidx = int(candidate_pair_ids[idx_in_all].item())
        global_idx_in_mutable = int(candidate_global_indices[idx_in_all].item())

        max_for_this_pair = max(0, int(src_lengths[pair_bidx]) - 1)
        if selected_per_pair[pair_bidx] >= max_for_this_pair:
            continue

        selected.append(global_idx_in_mutable)
        selected_per_pair[pair_bidx] += 1

        if len(selected) == k:
            break

    if not selected:
        return torch.tensor([], device=candidate_scores.device, dtype=torch.long)

    return torch.tensor(sorted(selected), device=candidate_scores.device, dtype=torch.long)

def flatten_value_vectors_batched(v_blocks: torch.Tensor) -> torch.Tensor:
    # [N,H,L,D] -> [N,L,H*D]
    return v_blocks.permute(0, 2, 1, 3).reshape(
        v_blocks.shape[0], v_blocks.shape[2], -1
    )


# -----------------------------------------------------------------------------
# StreamMem-style sink/template-guided intra-frame token scoring
# -----------------------------------------------------------------------------

def _normalize_guidance_k(guidance_k: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Normalize guidance tensor to [H, Q, D].

    In this KV-only implementation this is normally the cached sink K block
    with shape [1, H, Q, D]. If later you patch the model to pass real Q states,
    the same scorer can consume true query states with the same layout.
    """
    if guidance_k is None or guidance_k.numel() == 0:
        return guidance_k

    guidance_k = guidance_k.to(device)

    if guidance_k.dim() == 4:
        guidance_k = guidance_k[0]  # [B,H,Q,D] -> [H,Q,D]
    elif guidance_k.dim() != 3:
        raise ValueError(
            f"Expected guidance_k shape [1,H,Q,D] or [H,Q,D], "
            f"got {tuple(guidance_k.shape)}"
        )

    return guidance_k


def compute_sink_cross_attention_scores_batched(
    k_blocks: torch.Tensor,
    valid_token_mask: torch.Tensor,
    guidance_k: torch.Tensor,
    q_chunk_size: int = 128,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    StreamMem-style saliency for visual tokens inside each frame.

    For each frame block K_f and sink/template guidance G, compute:

        softmax(G K_f^T / sqrt(d))

    over the tokens of the frame, then aggregate over sink/template tokens and heads.

    Returns:
        [Nframes, Lmax], where higher score = more important.
        Invalid padded tokens are returned as -inf.
    """
    if guidance_k is None or guidance_k.numel() == 0:
        return torch.empty(
            (k_blocks.shape[0], k_blocks.shape[2]),
            device=k_blocks.device,
        )

    reduction = str(reduction).lower()
    if reduction not in {"mean", "max"}:
        raise ValueError(f"Unsupported sink cross-attention reduction: {reduction}")

    guidance = _normalize_guidance_k(guidance_k, k_blocks.device)

    # k_blocks: [N,H,L,D]
    # guidance: [H,Q,D]
    n_frames, n_heads, max_len, head_dim = k_blocks.shape
    g_heads, q_len, g_dim = guidance.shape

    if g_dim != head_dim:
        raise ValueError(
            f"Guidance head_dim mismatch: guidance={g_dim}, frame_k={head_dim}"
        )

    if g_heads != n_heads:
        # Safe fallback for GQA/head mismatch.
        # We average heads into a shared pseudo-head.
        frame_vecs = k_blocks.mean(dim=1, keepdim=True)       # [N,1,L,D]
        guidance_vecs = guidance.mean(dim=0, keepdim=True)   # [1,Q,D]
    else:
        frame_vecs = k_blocks
        guidance_vecs = guidance

    q_chunk_size = max(1, int(q_chunk_size))
    token_mask = valid_token_mask.to(torch.bool)

    scores_sum = torch.zeros(
        (n_frames, max_len),
        device=k_blocks.device,
        dtype=torch.float32,
    )
    scores_max = torch.full(
        (n_frames, max_len),
        float("-inf"),
        device=k_blocks.device,
        dtype=torch.float32,
    )

    q_seen = 0
    scale = float(head_dim) ** 0.5

    for q_start in range(0, q_len, q_chunk_size):
        q_end = min(q_start + q_chunk_size, q_len)
        q = guidance_vecs[:, q_start:q_end, :]  # [H_or_1,Qc,D]

        # [N,H,Qc,L]
        logits = torch.einsum(
            "hqd,nhld->nhql",
            q.float(),
            frame_vecs.float(),
        ) / scale

        logits = logits.masked_fill(~token_mask[:, None, None, :], float("-inf"))

        attn = torch.softmax(logits, dim=-1)
        attn = attn.masked_fill(~token_mask[:, None, None, :], 0.0)

        if reduction == "mean":
            # [N,L], mean over heads and guidance tokens in this chunk.
            chunk_scores = attn.mean(dim=(1, 2))
            scores_sum += chunk_scores * (q_end - q_start)
        else:
            # Max over guidance tokens, then mean over heads.
            chunk_scores = attn.max(dim=2).values.mean(dim=1)
            scores_max = torch.maximum(scores_max, chunk_scores)

        q_seen += q_end - q_start

    if reduction == "mean":
        scores = scores_sum / max(1, q_seen)
    else:
        scores = scores_max

    return scores.masked_fill(~token_mask, float("-inf"))


def collect_intra_sink_attention_candidates_chunked(
    mut_k: torch.Tensor,
    guidance_k: torch.Tensor,
    frame_lengths,
    chunk_size: int,
    q_chunk_size: int = 128,
    reduction: str = "mean",
):
    """
    Collect token-wise candidates for sink/template-guided intra-frame pruning.

    Returns:
        candidate_scores: lower = prune first
        candidate_global_indices: token indices relative to mutable
        candidate_frame_positions: frame id for per-frame min-keep constraints
    """
    frame_lengths = [int(x) for x in frame_lengths if int(x) > 0]
    chunk_size = int(chunk_size)

    if (
        not frame_lengths
        or mut_k.shape[-2] == 0
        or guidance_k is None
        or guidance_k.numel() == 0
    ):
        empty_idx = torch.tensor([], device=mut_k.device, dtype=torch.long)
        empty_score = torch.tensor([], device=mut_k.device, dtype=torch.float32)
        empty_frame = torch.tensor([], device=mut_k.device, dtype=torch.long)
        return empty_score, empty_idx, empty_frame

    valid_frame_positions = [i for i, fl in enumerate(frame_lengths) if int(fl) > 0]
    offsets = _frame_offsets_from_lengths(frame_lengths)

    candidate_scores = []
    candidate_global_indices = []
    candidate_frame_positions = []

    for start in range(0, len(valid_frame_positions), chunk_size):
        end = min(start + chunk_size, len(valid_frame_positions))
        local_positions = valid_frame_positions[start:end]

        packed = pack_intra_frame_chunk(
            mut_k=mut_k,
            frame_lengths=frame_lengths,
            frame_positions=local_positions,
            offsets=offsets,
        )
        if packed[0] is None:
            continue

        k_blocks, valid_token_mask, frame_offsets, chunk_frame_positions = packed

        token_scores = compute_sink_cross_attention_scores_batched(
            k_blocks=k_blocks,
            valid_token_mask=valid_token_mask,
            guidance_k=guidance_k,
            q_chunk_size=q_chunk_size,
            reduction=reduction,
        )

        valid_mask = torch.isfinite(token_scores)
        if not valid_mask.any():
            continue

        n_frames, max_len = token_scores.shape

        row_ids = (
            torch.arange(n_frames, device=mut_k.device)
            .unsqueeze(1)
            .expand(n_frames, max_len)
        )
        local_tok_ids = (
            torch.arange(max_len, device=mut_k.device)
            .unsqueeze(0)
            .expand(n_frames, max_len)
        )

        flat_scores = token_scores[valid_mask]
        flat_row_ids = row_ids[valid_mask]
        flat_local_tok_ids = local_tok_ids[valid_mask]

        frame_offsets_tensor = torch.tensor(
            frame_offsets,
            device=mut_k.device,
            dtype=torch.long,
        )
        frame_positions_tensor = torch.tensor(
            chunk_frame_positions,
            device=mut_k.device,
            dtype=torch.long,
        )

        flat_global_indices = frame_offsets_tensor[flat_row_ids] + flat_local_tok_ids
        flat_frame_positions = frame_positions_tensor[flat_row_ids]

        candidate_scores.append(flat_scores)
        candidate_global_indices.append(flat_global_indices)
        candidate_frame_positions.append(flat_frame_positions)

    if not candidate_scores:
        empty_idx = torch.tensor([], device=mut_k.device, dtype=torch.long)
        empty_score = torch.tensor([], device=mut_k.device, dtype=torch.float32)
        empty_frame = torch.tensor([], device=mut_k.device, dtype=torch.long)
        return empty_score, empty_idx, empty_frame

    return (
        torch.cat(candidate_scores, dim=0),
        torch.cat(candidate_global_indices, dim=0),
        torch.cat(candidate_frame_positions, dim=0),
    )


def select_lowest_score_tokens_per_frame(
    candidate_scores: torch.Tensor,
    candidate_global_indices: torch.Tensor,
    candidate_frame_positions: torch.Tensor,
    frame_lengths,
    k: int,
    min_keep_per_frame: int = 1,
) -> torch.Tensor:
    """
    Select lowest-saliency tokens while preserving at least min_keep_per_frame
    tokens in every frame.
    """
    k = int(k)
    min_keep_per_frame = max(0, int(min_keep_per_frame))

    if k <= 0 or candidate_scores.numel() == 0:
        return torch.tensor([], device=candidate_scores.device, dtype=torch.long)

    frame_lengths = [int(x) for x in frame_lengths if int(x) > 0]

    max_remove_by_frame = {
        i: max(0, int(fl) - min_keep_per_frame)
        for i, fl in enumerate(frame_lengths)
    }
    removed_by_frame = {i: 0 for i in range(len(frame_lengths))}

    order = torch.argsort(candidate_scores, descending=False)

    selected = []
    seen = set()

    for idx in order.tolist():
        frame_pos = int(candidate_frame_positions[idx].item())
        global_idx = int(candidate_global_indices[idx].item())

        if global_idx in seen:
            continue

        if removed_by_frame.get(frame_pos, 0) >= max_remove_by_frame.get(frame_pos, 0):
            continue

        selected.append(global_idx)
        seen.add(global_idx)
        removed_by_frame[frame_pos] = removed_by_frame.get(frame_pos, 0) + 1

        if len(selected) == k:
            break

    if not selected:
        return torch.tensor([], device=candidate_scores.device, dtype=torch.long)

    return torch.tensor(
        sorted(selected),
        device=candidate_scores.device,
        dtype=torch.long,
    )

def pack_intra_value_chunk(
    mut_v: torch.Tensor,
    frame_lengths,
    frame_positions,
    offsets=None,
):
    frame_lengths = [int(x) for x in frame_lengths if int(x) > 0]
    frame_positions = [int(x) for x in frame_positions]

    if offsets is None:
        offsets = _frame_offsets_from_lengths(frame_lengths)

    valid_positions = [p for p in frame_positions if int(frame_lengths[p]) > 0]
    if not valid_positions:
        return None, None, None, None

    device = mut_v.device
    dtype = mut_v.dtype
    chunk_lengths = [int(frame_lengths[p]) for p in valid_positions]
    n_frames = len(valid_positions)
    max_len = max(chunk_lengths)
    _, n_heads, _, head_dim = mut_v.shape

    v_blocks = torch.zeros(
        (n_frames, n_heads, max_len, head_dim),
        device=device,
        dtype=dtype,
    )
    valid_token_mask = torch.zeros(
        (n_frames, max_len),
        device=device,
        dtype=torch.bool,
    )

    frame_offsets = []

    for row_idx, frame_pos in enumerate(valid_positions):
        start, end = offsets[frame_pos]
        frame_len = end - start

        v_blocks[row_idx, :, :frame_len, :] = mut_v[0, :, start:end, :]
        valid_token_mask[row_idx, :frame_len] = True
        frame_offsets.append(int(start))

    return v_blocks, valid_token_mask, frame_offsets, valid_positions


def collect_intra_value_norm_candidates_chunked(
    mut_v: torch.Tensor,
    frame_lengths,
    chunk_size: int,
):
    frame_lengths = [int(x) for x in frame_lengths if int(x) > 0]
    chunk_size = int(chunk_size)

    if not frame_lengths or mut_v.shape[-2] == 0:
        empty_idx = torch.tensor([], device=mut_v.device, dtype=torch.long)
        empty_score = torch.tensor([], device=mut_v.device, dtype=torch.float32)
        empty_frame = torch.tensor([], device=mut_v.device, dtype=torch.long)
        return empty_score, empty_idx, empty_frame

    valid_frame_positions = [i for i, fl in enumerate(frame_lengths) if int(fl) > 0]
    if not valid_frame_positions:
        empty_idx = torch.tensor([], device=mut_v.device, dtype=torch.long)
        empty_score = torch.tensor([], device=mut_v.device, dtype=torch.float32)
        empty_frame = torch.tensor([], device=mut_v.device, dtype=torch.long)
        return empty_score, empty_idx, empty_frame

    offsets = _frame_offsets_from_lengths(frame_lengths)

    candidate_scores = []
    candidate_global_indices = []
    candidate_frame_positions = []

    for start in range(0, len(valid_frame_positions), chunk_size):
        end = min(start + chunk_size, len(valid_frame_positions))
        local_positions = valid_frame_positions[start:end]

        packed = pack_intra_value_chunk(
            mut_v=mut_v,
            frame_lengths=frame_lengths,
            frame_positions=local_positions,
            offsets=offsets,
        )
        if packed[0] is None:
            continue

        v_blocks, valid_token_mask, frame_offsets, chunk_frame_positions = packed

        # [N,H,L,D] -> [N,L,H*D]
        token_vecs = flatten_value_vectors_batched(v_blocks)

        # value norm per token
        token_scores = token_vecs.norm(dim=-1)   # [N,L]
        token_scores = token_scores.masked_fill(~valid_token_mask, float("inf"))

        valid_mask = torch.isfinite(token_scores)
        if not valid_mask.any():
            continue

        n_frames, max_len = token_scores.shape
        row_ids = (
            torch.arange(n_frames, device=mut_v.device)
            .unsqueeze(1)
            .expand(n_frames, max_len)
        )
        local_tok_ids = (
            torch.arange(max_len, device=mut_v.device)
            .unsqueeze(0)
            .expand(n_frames, max_len)
        )

        flat_scores = token_scores[valid_mask]
        flat_row_ids = row_ids[valid_mask]
        flat_local_tok_ids = local_tok_ids[valid_mask]

        frame_offsets_tensor = torch.tensor(frame_offsets, device=mut_v.device, dtype=torch.long)
        frame_positions_tensor = torch.tensor(chunk_frame_positions, device=mut_v.device, dtype=torch.long)

        flat_global_indices = frame_offsets_tensor[flat_row_ids] + flat_local_tok_ids
        flat_frame_positions = frame_positions_tensor[flat_row_ids]

        candidate_scores.append(flat_scores)
        candidate_global_indices.append(flat_global_indices)
        candidate_frame_positions.append(flat_frame_positions)

    if not candidate_scores:
        empty_idx = torch.tensor([], device=mut_v.device, dtype=torch.long)
        empty_score = torch.tensor([], device=mut_v.device, dtype=torch.float32)
        empty_frame = torch.tensor([], device=mut_v.device, dtype=torch.long)
        return empty_score, empty_idx, empty_frame

    return (
        torch.cat(candidate_scores, dim=0),
        torch.cat(candidate_global_indices, dim=0),
        torch.cat(candidate_frame_positions, dim=0),
    )


def select_lowest_score_tokens(
    candidate_scores: torch.Tensor,
    candidate_global_indices: torch.Tensor,
    k: int,
) -> torch.Tensor:
    k = int(k)

    if k <= 0 or candidate_scores.numel() == 0:
        return torch.tensor([], device=candidate_scores.device, dtype=torch.long)

    k = min(k, candidate_scores.numel())

    # value_norm: rimuovi i token con score PIU' BASSO
    chosen = torch.topk(candidate_scores, k=k, largest=False).indices
    removed = candidate_global_indices[chosen]

    return torch.sort(removed).values.to(torch.long)