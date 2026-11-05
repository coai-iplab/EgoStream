from __future__ import annotations

from typing import List, Tuple

import torch


def remove_token_indices(
    mut_k: torch.Tensor,
    mut_v: torch.Tensor,
    remove_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rimuove esplicitamente token-wise le posizioni indicate da remove_indices.
    Le posizioni sono riferite alla regione mutable corrente del layer.
    Ritorna i nuovi tensori mut_k e mut_v filtrati.
    """
    if remove_indices is None or remove_indices.numel() == 0:
        return mut_k, mut_v

    remove_indices = remove_indices.to(mut_k.device)
    keep = torch.ones(mut_k.shape[-2], device=mut_k.device, dtype=torch.bool)
    keep[remove_indices] = False

    mut_k = mut_k[..., keep, :]
    mut_v = mut_v[..., keep, :]
    return mut_k, mut_v


def drop_tokens_from_pair_indices(
    mut_k: torch.Tensor,
    mut_v: torch.Tensor,
    pair_indices: torch.Tensor,
    similarity_keep_old: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converte pair indices in token indices da droppare e applica il drop.
    Se similarity_keep_old è True elimina il secondo token della coppia, altrimenti il primo.
    Ritorna mut_k, mut_v aggiornati e removed_indices effettivi.
    """
    if pair_indices is None or pair_indices.numel() == 0:
        empty = torch.tensor([], device=mut_k.device, dtype=torch.long)
        return mut_k, mut_v, empty

    pair_indices = pair_indices.to(mut_k.device)

    if similarity_keep_old:
        removed_indices = pair_indices + 1
    else:
        removed_indices = pair_indices

    mut_k, mut_v = remove_token_indices(mut_k, mut_v, removed_indices)
    return mut_k, mut_v, removed_indices


def merge_pairs_by_indices(
    mut_k: torch.Tensor,
    mut_v: torch.Tensor,
    pair_indices: torch.Tensor,
    similarity_keep_old: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fonde le coppie indicate aggiornando il value del token kept e rimuovendo l'altro.
    Se similarity_keep_old è True mergea il token nuovo dentro il vecchio, altrimenti il contrario.
    Ritorna mut_k, mut_v aggiornati e removed_indices effettivi.
    """
    if pair_indices is None or pair_indices.numel() == 0:
        empty = torch.tensor([], device=mut_k.device, dtype=torch.long)
        return mut_k, mut_v, empty

    pair_indices = pair_indices.to(mut_k.device)

    if similarity_keep_old:
        merge_into = pair_indices
        merge_from = pair_indices + 1
    else:
        merge_into = pair_indices + 1
        merge_from = pair_indices

    mut_v = mut_v.clone()
    mut_v.index_copy_(
        -2,
        merge_into,
        0.5 * (
            mut_v.index_select(-2, merge_from) +
            mut_v.index_select(-2, merge_into)
        )
    )

    keep = torch.ones(mut_k.shape[-2], device=mut_k.device, dtype=torch.bool)
    keep[merge_from] = False

    mut_k = mut_k[..., keep, :]
    mut_v = mut_v[..., keep, :]

    return mut_k, mut_v, merge_from


def update_frame_lengths_after_remove_indices(frame_lengths, removed_indices: torch.Tensor) -> List[int]:
    """
    Aggiorna le frame lengths dopo una rimozione token-wise esplicita.
    Ogni indice rimosso viene mappato nel frame di appartenenza e decrementa quel frame.
    Ritorna la nuova lista di frame lengths senza elementi non positivi.
    """
    frame_lengths = [int(x) for x in frame_lengths if int(x) > 0]

    if removed_indices is None or removed_indices.numel() == 0:
        return frame_lengths

    offsets = []
    cursor = 0
    for fl in frame_lengths:
        offsets.append((cursor, cursor + int(fl)))
        cursor += int(fl)

    updated = list(frame_lengths)

    for ridx in removed_indices.detach().cpu().tolist():
        for frame_pos, (start, end) in enumerate(offsets):
            if start <= ridx < end:
                updated[frame_pos] -= 1
                break

    return [int(x) for x in updated if int(x) > 0]


def update_frame_lengths_after_pair_reduction(
    frame_lengths,
    pair_indices: torch.Tensor,
    similarity_keep_old: bool,
) -> List[int]:
    """
    Aggiorna le frame lengths dopo un'operazione pair-wise intra-frame.
    Converte i pair indices nei token rimossi effettivi prima di fare il mapping ai frame.
    Ritorna la nuova lista di frame lengths già pulita.
    """
    frame_lengths = [int(x) for x in frame_lengths if int(x) > 0]

    if pair_indices is None or pair_indices.numel() == 0:
        return frame_lengths

    if similarity_keep_old:
        removed_indices = pair_indices + 1
    else:
        removed_indices = pair_indices

    return update_frame_lengths_after_remove_indices(frame_lengths, removed_indices)


def remove_small_frames(
    mut_k: torch.Tensor,
    mut_v: torch.Tensor,
    frame_lengths,
    min_len: int,
):
    """
    Elimina interamente i frame con lunghezza <= min_len.
    È utile come cleanup dopo passaggi intra/inter che lasciano frammenti minuscoli.
    Ritorna mut_k, mut_v, new_frame_lengths e removed_indices.
    """
    frame_lengths = [int(x) for x in frame_lengths if int(x) > 0]
    min_len = int(min_len)

    if not frame_lengths:
        empty = torch.tensor([], device=mut_k.device, dtype=torch.long)
        return mut_k, mut_v, frame_lengths, empty

    offsets = []
    cursor = 0
    for fl in frame_lengths:
        offsets.append((cursor, cursor + int(fl)))
        cursor += int(fl)

    removed = []
    kept_lengths = []

    for fl, (start, end) in zip(frame_lengths, offsets):
        if int(fl) <= min_len:
            removed.extend(range(start, end))
        else:
            kept_lengths.append(int(fl))

    if not removed:
        empty = torch.tensor([], device=mut_k.device, dtype=torch.long)
        return mut_k, mut_v, frame_lengths, empty

    removed_indices = torch.tensor(removed, device=mut_k.device, dtype=torch.long)
    mut_k, mut_v = remove_token_indices(mut_k, mut_v, removed_indices)

    return mut_k, mut_v, kept_lengths, removed_indices


def left_trim_to_target(
    mut_k: torch.Tensor,
    mut_v: torch.Tensor,
    frame_lengths,
    target_len: int,
):
    """
    Taglia temporalmente da sinistra se la mutable è ancora sopra target_len.
    Aggiorna sia i tensori sia la distribuzione delle frame lengths residue.
    Ritorna mut_k, mut_v, new_frame_lengths e removed_indices.
    """
    target_len = int(target_len)
    current_len = int(mut_k.shape[-2])
    frame_lengths = [int(x) for x in frame_lengths if int(x) > 0]

    if current_len <= target_len:
        empty = torch.tensor([], device=mut_k.device, dtype=torch.long)
        return mut_k, mut_v, frame_lengths, empty

    to_trim = current_len - target_len
    removed_indices = torch.arange(to_trim, device=mut_k.device, dtype=torch.long)

    mut_k, mut_v = remove_token_indices(mut_k, mut_v, removed_indices)

    out = []
    remaining = int(to_trim)

    for frame_len in frame_lengths:
        if remaining <= 0:
            out.append(int(frame_len))
            continue

        if remaining >= int(frame_len):
            remaining -= int(frame_len)
            continue

        out.append(int(frame_len) - remaining)
        remaining = 0

    return mut_k, mut_v, [int(x) for x in out if int(x) > 0], removed_indices
