from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import torch


@dataclass
class FrameState:
    """
    Mantiene lo stato dei frame nella cache.
    Tiene traccia sia delle lunghezze storiche sia del pending append.
    Espone helper semplici per split mutable/hat e finalize visual-only.
    """
    frame_token_history: List[int] = field(default_factory=list)
    next_frame_idx: int = 0

    pending_frame_token_counts: Optional[List[int]] = None
    pending_visual_indices: Optional[torch.Tensor] = None
    pending_total_seq_len: int = 0

    def add_frame_tokens(self, frame_token_counts) -> None:
        """
        Aggiunge i token per frame al tracking storico.
        Accetta un int singolo o una lista di int positivi.
        Aggiorna anche il contatore dei frame visti.
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
        Divide le lunghezze dei frame in mutable e hat.
        La split a frame vale solo se hat_category == 'frames' e pruning frame-aware è attivo.
        Ritorna due liste: mutable_lengths e hat_lengths.
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
        Sovrascrive completamente le frame lengths tracciate.
        Filtra eventuali frame non positivi prima di salvare.
        Mantiene la convenzione: mutable seguiti da hat.
        """
        mutable_lengths = [int(x) for x in mutable_lengths if int(x) > 0]
        hat_lengths = [int(x) for x in hat_lengths if int(x) > 0]
        self.frame_token_history = mutable_lengths + hat_lengths

    def frame_offsets(self, frame_lengths) -> List[Tuple[int, int]]:
        """
        Converte le lunghezze frame in intervalli [start, end).
        Gli offset sono relativi alla sola regione mutable/combined passata in input.
        Ritorna una lista ordinata di tuple (start, end).
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
        Simula un trim temporale da sinistra sulle frame lengths.
        Rimuove token dai frame più vecchi fino a consumare tokens_to_trim.
        Ritorna la nuova lista di lunghezze residue.
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
        Calcola la hat size corrente.
        Se la hat è definita a frame e il pruning è frame-aware, usa le frame lengths tracciate.
        Altrimenti ritorna la hat size statica passata dal chiamante.
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
        Costruisce i boundary assoluti dei frame nella cache corrente.
        I frame mutable partono dopo sink_size e gli hat stanno in coda.
        Ritorna una lista di dict utili per debug e stampa.
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
        Salva il pending stato necessario a finalizzare un append visual-only.
        Verifica che il numero di token visuali nel mask coincida con frame_token_counts.
        Non modifica ancora keys/values: prepara solo il metadata.
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
        Tiene solo i token visuali dell'ultimo append usando il pending metadata.
        Aggiorna lo storico dei frame solo dopo aver filtrato keys/values.
        Ritorna i nuovi tensori keys e values già puliti.
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
