from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import torch


@dataclass
class FrameState:
    """
    Maintains the state of the frames in the cache.
    Keeps track of both the historical lengths and the pending append.
    Exposes simple helpers for the mutable/hat split and the visual-only finalize.
    """
    frame_token_history: List[int] = field(default_factory=list)
    next_frame_idx: int = 0

    pending_frame_token_counts: Optional[List[int]] = None
    pending_visual_indices: Optional[torch.Tensor] = None
    pending_total_seq_len: int = 0

    def add_frame_tokens(self, frame_token_counts) -> None:
        """
        Adds the per-frame tokens to the historical tracking.
        Accepts a single int or a list of positive ints.
        Also updates the counter of frames seen.
        """
        if isinstance(frame_token_counts, int):
            frame_token_counts = [frame_token_counts]

        for tok in frame_token_counts:
            tok = int(tok)
            if tok > 0:
                self.frame_token_history.append(tok)
                self.next_frame_idx += 1

    def split_frame_lengths_by_hat(
        self,
        use_hat_tokens: bool,
        hat_category: str,
        hat_num_frames: int,
        frame_pruning_enabled: bool,
    ) -> Tuple[List[int], List[int]]:
        """
        Splits the frame lengths into mutable and hat.
        The frame-level split applies only if hat_category == 'frames' and frame-aware pruning is active.
        Returns two lists: mutable_lengths and hat_lengths.
        """
        frame_lengths = [int(x) for x in self.frame_token_history if int(x) > 0]

        if not frame_lengths:
            return [], []

        if (
            frame_pruning_enabled
            and bool(use_hat_tokens)
            and str(hat_category) == "frames"
            and int(hat_num_frames) > 0
        ):
            hat_count = min(int(hat_num_frames), len(frame_lengths))
            return frame_lengths[:-hat_count], frame_lengths[-hat_count:]

        return frame_lengths, []

    def set_frame_lengths(self, mutable_lengths, hat_lengths) -> None:
        """
        Completely overwrites the tracked frame lengths.
        Filters out any non-positive frames before saving.
        Keeps the convention: mutable followed by hat.
        """
        mutable_lengths = [int(x) for x in mutable_lengths if int(x) > 0]
        hat_lengths = [int(x) for x in hat_lengths if int(x) > 0]
        self.frame_token_history = mutable_lengths + hat_lengths

    def frame_offsets(self, frame_lengths) -> List[Tuple[int, int]]:
        """
        Converts the frame lengths into [start, end) intervals.
        The offsets are relative only to the mutable/combined region passed as input.
        Returns an ordered list of (start, end) tuples.
        """
        offsets = []
        cursor = 0

        for frame_len in frame_lengths:
            frame_len = int(frame_len)
            offsets.append((cursor, cursor + frame_len))
            cursor += frame_len

        return offsets

    def trim_frame_lengths_from_left(self, frame_lengths, tokens_to_trim: int) -> List[int]:
        """
        Simulates a temporal trim from the left on the frame lengths.
        Removes tokens from the oldest frames until tokens_to_trim is consumed.
        Returns the new list of remaining lengths.
        """
        frame_lengths = [int(x) for x in frame_lengths if int(x) > 0]
        tokens_to_trim = int(tokens_to_trim)

        if tokens_to_trim <= 0:
            return frame_lengths

        out = []
        remaining = tokens_to_trim

        for frame_len in frame_lengths:
            if remaining <= 0:
                out.append(frame_len)
                continue

            if remaining >= frame_len:
                remaining -= frame_len
                continue

            out.append(frame_len - remaining)
            remaining = 0

        return [int(x) for x in out if int(x) > 0]

    def get_hat_size(
        self,
        use_hat_tokens: bool,
        hat_category: str,
        hat_num_frames: int,
        static_hat_size: int,
        frame_pruning_enabled: bool,
    ) -> int:
        """
        Computes the current hat size.
        If the hat is defined in frames and the pruning is frame-aware, uses the tracked frame lengths.
        Otherwise returns the static hat size passed by the caller.
        """
        if (
            frame_pruning_enabled
            and bool(use_hat_tokens)
            and str(hat_category) == "frames"
            and int(hat_num_frames) > 0
        ):
            return int(sum(self.frame_token_history[-int(hat_num_frames):]))

        return int(static_hat_size)

    def get_frame_boundaries(
        self,
        sink_size: int,
        use_hat_tokens: bool,
        hat_category: str,
        hat_num_frames: int,
        frame_pruning_enabled: bool,
    ):
        """
        Builds the absolute frame boundaries in the current cache.
        Mutable frames start after sink_size and the hat frames sit at the tail.
        Returns a list of dicts useful for debugging and printing.
        """
        mutable_lengths, hat_lengths = self.split_frame_lengths_by_hat(
            use_hat_tokens=use_hat_tokens,
            hat_category=hat_category,
            hat_num_frames=hat_num_frames,
            frame_pruning_enabled=frame_pruning_enabled,
        )

        info = []
        abs_cursor = int(sink_size)
        frame_idx = 0

        for length in mutable_lengths:
            length = int(length)
            info.append({
                "frame_idx": frame_idx,
                "region": "mutable",
                "length": length,
                "mutable_start": abs_cursor - int(sink_size),
                "mutable_end": abs_cursor - int(sink_size) + length,
                "abs_start": abs_cursor,
                "abs_end": abs_cursor + length,
            })
            abs_cursor += length
            frame_idx += 1

        for length in hat_lengths:
            length = int(length)
            info.append({
                "frame_idx": frame_idx,
                "region": "hat",
                "length": length,
                "mutable_start": None,
                "mutable_end": None,
                "abs_start": abs_cursor,
                "abs_end": abs_cursor + length,
            })
            abs_cursor += length
            frame_idx += 1

        return info

    def prepare_frame_append(
        self,
        frame_token_counts,
        visual_pos_mask: torch.Tensor,
        appended_seq_len: int,
    ) -> None:
        """
        Stores the pending state needed to finalize a visual-only append.
        Checks that the number of visual tokens in the mask matches frame_token_counts.
        Does not modify keys/values yet: it only prepares the metadata.
        """
        if isinstance(frame_token_counts, int):
            frame_token_counts = [frame_token_counts]

        frame_token_counts = [int(x) for x in frame_token_counts if int(x) > 0]

        if visual_pos_mask is None:
            raise ValueError("visual_pos_mask is required in prepare_frame_append")

        visual_mask = visual_pos_mask
        if visual_mask.dim() > 1:
            visual_mask = visual_mask[0]

        visual_indices = torch.where(visual_mask.to(torch.bool))[0].to(torch.long)
        expected_visual_tokens = int(sum(frame_token_counts))

        if visual_indices.numel() != expected_visual_tokens:
            raise ValueError(
                f"Mismatch between visual_pos_mask ({visual_indices.numel()}) "
                f"and frame_token_counts ({expected_visual_tokens})"
            )

        self.pending_frame_token_counts = frame_token_counts
        self.pending_visual_indices = visual_indices.detach().cpu()
        self.pending_total_seq_len = int(appended_seq_len)

    def finalize_frame_append_to_visual_only(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Keeps only the visual tokens of the last append using the pending metadata.
        Updates the frame history only after filtering keys/values.
        Returns the new keys and values tensors already cleaned.
        """
        if self.pending_visual_indices is None:
            return keys, values

        appended_seq_len = int(self.pending_total_seq_len)
        current_len = keys.shape[-2]
        prefix_len = max(0, current_len - appended_seq_len)

        keep_indices = self.pending_visual_indices.to(keys.device)

        appended_k = keys[..., prefix_len:, :]
        appended_v = values[..., prefix_len:, :]

        kept_k = appended_k.index_select(-2, keep_indices)
        kept_v = appended_v.index_select(-2, keep_indices)

        keys = torch.cat([keys[..., :prefix_len, :], kept_k], dim=-2)
        values = torch.cat([values[..., :prefix_len, :], kept_v], dim=-2)

        if self.pending_frame_token_counts:
            self.frame_token_history.extend(self.pending_frame_token_counts)
            self.next_frame_idx += len(self.pending_frame_token_counts)

        self.pending_frame_token_counts = None
        self.pending_visual_indices = None
        self.pending_total_seq_len = 0

        return keys, values
