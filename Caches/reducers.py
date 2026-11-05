from __future__ import annotations

from typing import List, Tuple

import torch


def remove_token_indices(
    mut_k: torch.Tensor,
    mut_v: torch.Tensor,
    remove_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Explicitly removes token-wise the positions indicated by remove_indices.
    The positions are relative to the current mutable region of the layer.
    Returns the new filtered mut_k and mut_v tensors.
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
    Converts pair indices into token indices to drop and applies the drop.
    If similarity_keep_old is True removes the second token of the pair, otherwise the first.
    Returns the updated mut_k, mut_v and the actual removed_indices.
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
    Merges the indicated pairs by updating the value of the kept token and removing the other.
    If similarity_keep_old is True merges the new token into the old one, otherwise the opposite.
    Returns the updated mut_k, mut_v and the actual removed_indices.
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
    Updates the frame lengths after an explicit token-wise removal.
    Each removed index is mapped to the frame it belongs to and decrements that frame.
    Returns the new list of frame lengths without non-positive elements.
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
    Updates the frame lengths after a pair-wise intra-frame operation.
    Converts the pair indices into the actual removed tokens before mapping them to frames.
    Returns the new list of frame lengths already cleaned.
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
    Entirely removes the frames with length <= min_len.
    It is useful as a cleanup after intra/inter passes that leave tiny fragments.
    Returns mut_k, mut_v, new_frame_lengths and removed_indices.
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
    Temporally trims from the left if the mutable is still above target_len.
    Updates both the tensors and the distribution of the remaining frame lengths.
    Returns mut_k, mut_v, new_frame_lengths and removed_indices.
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
