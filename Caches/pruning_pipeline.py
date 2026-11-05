from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch
import math

    
from .similarity_ops import (
    collect_inter_token_candidates_chunked,
    compute_intra_pair_indices_parallel,
    select_topk_tokens_per_source,
    collect_intra_value_norm_candidates_chunked,
    collect_intra_sink_attention_candidates_chunked,
    select_lowest_score_tokens,
    select_lowest_score_tokens_per_frame,
)
from .reducers import (
    drop_tokens_from_pair_indices,
    left_trim_to_target,
    merge_pairs_by_indices,
    remove_small_frames,
    remove_token_indices,
    update_frame_lengths_after_pair_reduction,
    update_frame_lengths_after_remove_indices,
)

@dataclass
class TraceOp:
    """
    Rappresenta una singola operazione replayable sui layer follower.
    Salva il tipo di op, gli indici usati e metadati minimi per debug.
    È la forma standard della trace prodotta dalla pipeline.
    """
    step_name: str
    op_type: str
    pair_indices: torch.Tensor | None = None
    removed_indices: torch.Tensor | None = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PruningStep:
    """
    Descrive un passo della pipeline di pruning.
    domain decide intra/inter/cleanup/storage, scorer decide lo scoring e reducer l'azione.
    params contiene opzioni locali tipo chunk_size, min_len, remove_pct, remove_tokens.
    """
    name: str
    domain: str
    scorer: str = "adiacent_cosine"
    reducer: str = "drop"
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)


class PruningPipeline:
    """
    Esegue una lista ordinata di step di pruning sulla mutable.
    Supporta pipeline miste intra/inter/cleanup con trace replayable.

    Nota importante:
    - domain="storage" NON modifica direttamente i tensori.
    - storage/offload è gestito dal FlexibleCacheLayer perché deve scrivere su disco.
    - qui lo step storage funziona come barriera terminale: la pipeline ritorna subito.
    """

    def __init__(
        self,
        steps: List[PruningStep],
        similarity_keep_old: bool = False,
    ):
        """
        Inizializza la pipeline con gli step ordinati e le convenzioni pair-wise.
        similarity_keep_old controlla la semantica shared di drop/merge pair-based.
        La pipeline non possiede KV: lavora solo sui tensori passati ai metodi.
        """
        self.steps = list(steps)
        self.similarity_keep_old = bool(similarity_keep_old)

    def execute_first_layer(
        self,
        mut_k: torch.Tensor,
        mut_v: torch.Tensor,
        hat_k: torch.Tensor,
        mutable_frame_lengths,
        hat_frame_lengths,
        target_len: int,
        allow_fallback_trim: bool = True,
        guidance_k: torch.Tensor | None = None,
    ) -> Tuple[List[TraceOp], torch.Tensor, torch.Tensor, List[int]]:
        """
        Esegue la pipeline completa sul first layer e produce la trace ufficiale.

        Se incontra uno step storage/offload:
        - ritorna immediatamente;
        - non fa fallback left-trim;
        - lascia al FlexibleCacheLayer il compito di offloadare l'eccesso.

        Ritorna:
        - trace_ops
        - mut_k
        - mut_v
        - mutable_frame_lengths aggiornate
        """
        mutable_frame_lengths = [int(x) for x in mutable_frame_lengths if int(x) > 0]
        hat_frame_lengths = [int(x) for x in hat_frame_lengths if int(x) > 0]
        target_len = int(target_len)

        trace_ops: List[TraceOp] = []

        while mut_k.shape[-2] > target_len:
            progress = False

            for step in self.steps:
                if not step.enabled:
                    continue

                if mut_k.shape[-2] <= target_len:
                    break

                if step.domain == "storage":
                    # IMPORTANTISSIMO:
                    # storage/offload non modifica mut_k dentro la pipeline.
                    # Se facessimo progress=True + break, il while ripartirebbe
                    # all'infinito perché mut_k.shape[-2] resterebbe > target_len.
                    #
                    # Quindi ritorniamo subito. Il layer chiamante applicherà
                    # _apply_storage_offload_steps(...) fuori dalla pipeline.
                    return trace_ops, mut_k, mut_v, mutable_frame_lengths

                if step.domain == "cleanup":
                    ops, mut_k, mut_v, mutable_frame_lengths, changed = self._run_cleanup_step(
                        mut_k=mut_k,
                        mut_v=mut_v,
                        mutable_frame_lengths=mutable_frame_lengths,
                        step=step,
                    )
                elif step.domain == "intra":
                    ops, mut_k, mut_v, mutable_frame_lengths, changed = self._run_intra_step(
                        mut_k=mut_k,
                        mut_v=mut_v,
                        mutable_frame_lengths=mutable_frame_lengths,
                        target_len=target_len,
                        step=step,
                        guidance_k=guidance_k,
                    )
                elif step.domain == "inter":
                    ops, mut_k, mut_v, mutable_frame_lengths, changed = self._run_inter_step(
                        mut_k=mut_k,
                        mut_v=mut_v,
                        hat_k=hat_k,
                        mutable_frame_lengths=mutable_frame_lengths,
                        hat_frame_lengths=hat_frame_lengths,
                        target_len=target_len,
                        step=step,
                    )
                else:
                    raise ValueError(f"Unsupported pruning step domain: {step.domain}")

                if changed:
                    trace_ops.extend(ops)
                    progress = True

                if mut_k.shape[-2] <= target_len:
                    break

            if not progress:
                break

        if allow_fallback_trim and mut_k.shape[-2] > target_len:
            mut_k, mut_v, mutable_frame_lengths, removed_indices = left_trim_to_target(
                mut_k=mut_k,
                mut_v=mut_v,
                frame_lengths=mutable_frame_lengths,
                target_len=target_len,
            )

            if removed_indices.numel() > 0:
                trace_ops.append(
                    TraceOp(
                        step_name="fallback_left_trim",
                        op_type="remove_tokens",
                        removed_indices=removed_indices.detach().cpu(),
                        meta={"reason": "final_fallback"},
                    )
                )

        return trace_ops, mut_k, mut_v, mutable_frame_lengths

    def _compute_step_budget(self, current_len: int, target_len: int, step: PruningStep) -> int:
        current_len = int(current_len)
        target_len = int(target_len)

        overflow = max(0, current_len - target_len)
        if overflow <= 0:
            return 0

        remove_pct = step.params.get("remove_pct", None)
        remove_tokens = step.params.get("remove_tokens", None)

        requested = overflow

        if remove_pct is not None:
            remove_pct = float(remove_pct)
            if remove_pct <= 0:
                return 0
            requested = max(1, int(math.ceil(current_len * remove_pct)))

        if remove_tokens is not None:
            requested = int(remove_tokens)

        requested = max(0, int(requested))
        return min(requested, overflow)

    def replay_on_follower(
        self,
        mut_k: torch.Tensor,
        mut_v: torch.Tensor,
        mutable_frame_lengths,
        trace_ops: List[TraceOp],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Reapplica in ordine la trace prodotta dal first layer su un follower layer.
        Non ricalcola mai score o selezioni: usa solo gli indici salvati nella trace.
        Ritorna mut_k, mut_v e mutable_frame_lengths aggiornate sul follower.
        """
        mutable_frame_lengths = [int(x) for x in mutable_frame_lengths if int(x) > 0]

        for op in trace_ops:
            mut_k, mut_v, mutable_frame_lengths = self._apply_trace_op(
                mut_k=mut_k,
                mut_v=mut_v,
                mutable_frame_lengths=mutable_frame_lengths,
                op=op,
            )

        return mut_k, mut_v, mutable_frame_lengths

    def _run_cleanup_step(
        self,
        mut_k: torch.Tensor,
        mut_v: torch.Tensor,
        mutable_frame_lengths,
        step: PruningStep,
    ) -> Tuple[List[TraceOp], torch.Tensor, torch.Tensor, List[int], bool]:
        """
        Esegue uno step di sola pulizia dei frame troppo piccoli.
        Il reducer supportato qui è remove_small_frames con parametro min_len.
        Ritorna ops generate, tensori aggiornati, frame lengths e flag changed.
        """
        min_len = int(step.params.get("min_len", 2))

        mut_k, mut_v, mutable_frame_lengths, removed_indices = remove_small_frames(
            mut_k=mut_k,
            mut_v=mut_v,
            frame_lengths=mutable_frame_lengths,
            min_len=min_len,
        )

        if removed_indices.numel() == 0:
            return [], mut_k, mut_v, mutable_frame_lengths, False

        op = TraceOp(
            step_name=step.name,
            op_type="remove_tokens",
            removed_indices=removed_indices.detach().cpu(),
            meta={"min_len": min_len},
        )
        return [op], mut_k, mut_v, mutable_frame_lengths, True

    def _run_intra_step(
        self,
        mut_k: torch.Tensor,
        mut_v: torch.Tensor,
        mutable_frame_lengths,
        target_len: int,
        step: PruningStep,
        guidance_k: torch.Tensor | None = None,
    ) -> Tuple[List[TraceOp], torch.Tensor, torch.Tensor, List[int], bool]:
        step_budget = self._compute_step_budget(
            current_len=mut_k.shape[-2],
            target_len=target_len,
            step=step,
        )
        if step_budget <= 0:
            return [], mut_k, mut_v, mutable_frame_lengths, False

        chunk_size = int(step.params.get("chunk_size", 64))

        # Token-wise StreamMem/sink-guided intra-frame pruning.
        # Higher attention = more important, so we remove lowest-score tokens.
        if str(step.scorer).lower() in {
            "sink_cross_attention",
            "sink_guided_attention",
            "sys_prompt_cross_attention",
            "streammem_attention",
        }:
            if step.reducer != "drop":
                raise ValueError(
                    "sink_cross_attention intra currently supports only reducer='drop'"
                )

            if guidance_k is None or guidance_k.numel() == 0:
                return [], mut_k, mut_v, mutable_frame_lengths, False

            q_chunk_size = int(step.params.get("q_chunk_size", 128))
            reduction = str(step.params.get("reduction", "mean"))
            min_keep_per_frame = int(step.params.get("min_keep_per_frame", 1))

            candidate_scores, candidate_global_indices, candidate_frame_positions = (
                collect_intra_sink_attention_candidates_chunked(
                    mut_k=mut_k,
                    guidance_k=guidance_k,
                    frame_lengths=mutable_frame_lengths,
                    chunk_size=chunk_size,
                    q_chunk_size=q_chunk_size,
                    reduction=reduction,
                )
            )

            removed_indices = select_lowest_score_tokens_per_frame(
                candidate_scores=candidate_scores,
                candidate_global_indices=candidate_global_indices,
                candidate_frame_positions=candidate_frame_positions,
                frame_lengths=mutable_frame_lengths,
                k=step_budget,
                min_keep_per_frame=min_keep_per_frame,
            )

            if removed_indices.numel() == 0:
                return [], mut_k, mut_v, mutable_frame_lengths, False

            mut_k, mut_v = remove_token_indices(
                mut_k=mut_k,
                mut_v=mut_v,
                remove_indices=removed_indices,
            )

            mutable_frame_lengths = update_frame_lengths_after_remove_indices(
                frame_lengths=mutable_frame_lengths,
                removed_indices=removed_indices,
            )

            op = TraceOp(
                step_name=step.name,
                op_type="remove_tokens",
                removed_indices=removed_indices.detach().cpu(),
                meta={
                    "scorer": step.scorer,
                    "chunk_size": chunk_size,
                    "q_chunk_size": q_chunk_size,
                    "reduction": reduction,
                    "min_keep_per_frame": min_keep_per_frame,
                },
            )

            return [op], mut_k, mut_v, mutable_frame_lengths, True

        # Ramo token-wise su value norm.
        if str(step.scorer).lower() in {"value_norm", "van_l2"}:
            if step.reducer != "drop":
                raise ValueError("value_norm intra currently supports only reducer='drop'")

            candidate_scores, candidate_global_indices, _ = (
                collect_intra_value_norm_candidates_chunked(
                    mut_v=mut_v,
                    frame_lengths=mutable_frame_lengths,
                    chunk_size=chunk_size,
                )
            )

            removed_indices = select_lowest_score_tokens(
                candidate_scores=candidate_scores,
                candidate_global_indices=candidate_global_indices,
                k=step_budget,
            )

            if removed_indices.numel() == 0:
                return [], mut_k, mut_v, mutable_frame_lengths, False

            mut_k, mut_v = remove_token_indices(
                mut_k=mut_k,
                mut_v=mut_v,
                remove_indices=removed_indices,
            )
            mutable_frame_lengths = update_frame_lengths_after_remove_indices(
                frame_lengths=mutable_frame_lengths,
                removed_indices=removed_indices,
            )

            op = TraceOp(
                step_name=step.name,
                op_type="remove_tokens",
                removed_indices=removed_indices.detach().cpu(),
                meta={"scorer": step.scorer, "chunk_size": chunk_size},
            )
            return [op], mut_k, mut_v, mutable_frame_lengths, True

        # Ramo pair-wise intra-frame.
        pair_indices = compute_intra_pair_indices_parallel(
            mut_k=mut_k,
            frame_lengths=mutable_frame_lengths,
            to_remove=step_budget,
            strategy=step.scorer,
            similarity_keep_old=self.similarity_keep_old,
            chunk_size=chunk_size,
        )

        if pair_indices.numel() == 0:
            return [], mut_k, mut_v, mutable_frame_lengths, False

        if step.reducer == "drop":
            mut_k, mut_v, removed_indices = drop_tokens_from_pair_indices(
                mut_k=mut_k,
                mut_v=mut_v,
                pair_indices=pair_indices,
                similarity_keep_old=self.similarity_keep_old,
            )
            mutable_frame_lengths = update_frame_lengths_after_pair_reduction(
                frame_lengths=mutable_frame_lengths,
                pair_indices=pair_indices,
                similarity_keep_old=self.similarity_keep_old,
            )
            op = TraceOp(
                step_name=step.name,
                op_type="drop_pairs",
                pair_indices=pair_indices.detach().cpu(),
                removed_indices=removed_indices.detach().cpu(),
                meta={"scorer": step.scorer, "chunk_size": chunk_size},
            )
            return [op], mut_k, mut_v, mutable_frame_lengths, True

        if step.reducer == "merge":
            mut_k, mut_v, removed_indices = merge_pairs_by_indices(
                mut_k=mut_k,
                mut_v=mut_v,
                pair_indices=pair_indices,
                similarity_keep_old=self.similarity_keep_old,
            )
            mutable_frame_lengths = update_frame_lengths_after_pair_reduction(
                frame_lengths=mutable_frame_lengths,
                pair_indices=pair_indices,
                similarity_keep_old=self.similarity_keep_old,
            )
            op = TraceOp(
                step_name=step.name,
                op_type="merge_pairs",
                pair_indices=pair_indices.detach().cpu(),
                removed_indices=removed_indices.detach().cpu(),
                meta={"scorer": step.scorer, "chunk_size": chunk_size},
            )
            return [op], mut_k, mut_v, mutable_frame_lengths, True

        raise ValueError(f"Unsupported intra reducer: {step.reducer}")

    def _run_inter_step(
        self,
        mut_k: torch.Tensor,
        mut_v: torch.Tensor,
        hat_k: torch.Tensor,
        mutable_frame_lengths,
        hat_frame_lengths,
        target_len: int,
        step: PruningStep,
    ) -> Tuple[List[TraceOp], torch.Tensor, torch.Tensor, List[int], bool]:
        if step.reducer != "drop":
            raise ValueError("Inter-frame step currently supports only reducer='drop'")

        step_budget = self._compute_step_budget(
            current_len=mut_k.shape[-2],
            target_len=target_len,
            step=step,
        )
        if step_budget <= 0:
            return [], mut_k, mut_v, mutable_frame_lengths, False

        chunk_size = int(step.params.get("chunk_size", 64))

        candidate_scores, candidate_global_indices, candidate_pair_ids, src_lengths = (
            collect_inter_token_candidates_chunked(
                mut_k=mut_k,
                hat_k=hat_k,
                mutable_frame_lengths=mutable_frame_lengths,
                hat_frame_lengths=hat_frame_lengths,
                strategy=step.scorer,
                chunk_size=chunk_size,
            )
        )

        removed_indices = select_topk_tokens_per_source(
            candidate_scores=candidate_scores,
            candidate_global_indices=candidate_global_indices,
            candidate_pair_ids=candidate_pair_ids,
            src_lengths=src_lengths,
            k=step_budget,
        )
        if removed_indices.numel() == 0:
            return [], mut_k, mut_v, mutable_frame_lengths, False

        mut_k, mut_v = remove_token_indices(
            mut_k=mut_k,
            mut_v=mut_v,
            remove_indices=removed_indices,
        )

        mutable_frame_lengths = update_frame_lengths_after_remove_indices(
            frame_lengths=mutable_frame_lengths,
            removed_indices=removed_indices,
        )

        op = TraceOp(
            step_name=step.name,
            op_type="remove_tokens",
            removed_indices=removed_indices.detach().cpu(),
            meta={"scorer": step.scorer, "chunk_size": chunk_size},
        )

        return [op], mut_k, mut_v, mutable_frame_lengths, True

    def _apply_trace_op(
        self,
        mut_k: torch.Tensor,
        mut_v: torch.Tensor,
        mutable_frame_lengths,
        op: TraceOp,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Reapplica una singola op della trace su un follower layer.
        La semantica dei pair reducer resta quella globale della pipeline.
        Ritorna mut_k, mut_v e mutable_frame_lengths aggiornate dopo l'op.
        """
        if op.op_type == "remove_tokens":
            removed_indices = op.removed_indices
            if removed_indices is None or removed_indices.numel() == 0:
                return mut_k, mut_v, mutable_frame_lengths

            removed_indices = removed_indices.to(mut_k.device)
            mut_k, mut_v = remove_token_indices(
                mut_k=mut_k,
                mut_v=mut_v,
                remove_indices=removed_indices,
            )
            mutable_frame_lengths = update_frame_lengths_after_remove_indices(
                frame_lengths=mutable_frame_lengths,
                removed_indices=removed_indices,
            )
            return mut_k, mut_v, mutable_frame_lengths

        if op.op_type == "drop_pairs":
            pair_indices = op.pair_indices
            if pair_indices is None or pair_indices.numel() == 0:
                return mut_k, mut_v, mutable_frame_lengths

            pair_indices = pair_indices.to(mut_k.device)
            mut_k, mut_v, _ = drop_tokens_from_pair_indices(
                mut_k=mut_k,
                mut_v=mut_v,
                pair_indices=pair_indices,
                similarity_keep_old=self.similarity_keep_old,
            )
            mutable_frame_lengths = update_frame_lengths_after_pair_reduction(
                frame_lengths=mutable_frame_lengths,
                pair_indices=pair_indices,
                similarity_keep_old=self.similarity_keep_old,
            )
            return mut_k, mut_v, mutable_frame_lengths

        if op.op_type == "merge_pairs":
            pair_indices = op.pair_indices
            if pair_indices is None or pair_indices.numel() == 0:
                return mut_k, mut_v, mutable_frame_lengths

            pair_indices = pair_indices.to(mut_k.device)
            mut_k, mut_v, _ = merge_pairs_by_indices(
                mut_k=mut_k,
                mut_v=mut_v,
                pair_indices=pair_indices,
                similarity_keep_old=self.similarity_keep_old,
            )
            mutable_frame_lengths = update_frame_lengths_after_pair_reduction(
                frame_lengths=mutable_frame_lengths,
                pair_indices=pair_indices,
                similarity_keep_old=self.similarity_keep_old,
            )
            return mut_k, mut_v, mutable_frame_lengths

        raise ValueError(f"Unsupported trace op_type: {op.op_type}")