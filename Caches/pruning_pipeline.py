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
    Represents a single replayable operation on the follower layers.
    Stores the op type, the indices used and minimal metadata for debugging.
    It is the standard form of the trace produced by the pipeline.
    """
    step_name: str
    op_type: str
    pair_indices: torch.Tensor | None = None
    removed_indices: torch.Tensor | None = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PruningStep:
    """
    Describes one step of the pruning pipeline.
    domain selects intra/inter/cleanup/storage, scorer selects the scoring and reducer the action.
    params holds local options such as chunk_size, min_len, remove_pct, remove_tokens.
    """
    name: str
    domain: str
    scorer: str = "adiacent_cosine"
    reducer: str = "drop"
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)


class PruningPipeline:
    """
    Executes an ordered list of pruning steps on the mutable.
    Supports mixed intra/inter/cleanup pipelines with a replayable trace.

    Important note:
    - domain="storage" does NOT modify the tensors directly.
    - storage/offload is handled by the FlexibleCacheLayer because it must write to disk.
    - here the storage step works as a terminal barrier: the pipeline returns immediately.
    """

    def __init__(
        self,
        steps: List[PruningStep],
        similarity_keep_old: bool = False,
    ):
        """
        Initializes the pipeline with the ordered steps and the pair-wise conventions.
        similarity_keep_old controls the shared semantics of pair-based drop/merge.
        The pipeline does not own KV: it only works on the tensors passed to its methods.
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
        Executes the full pipeline on the first layer and produces the official trace.

        If it encounters a storage/offload step:
        - it returns immediately;
        - it does no fallback left-trim;
        - it leaves to the FlexibleCacheLayer the task of offloading the excess.

        Returns:
        - trace_ops
        - mut_k
        - mut_v
        - updated mutable_frame_lengths
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
                    # VERY IMPORTANT:
                    # storage/offload does not modify mut_k inside the pipeline.
                    # If we did progress=True + break, the while loop would restart
                    # forever because mut_k.shape[-2] would stay > target_len.
                    #
                    # So we return immediately. The calling layer will apply
                    # _apply_storage_offload_steps(...) outside the pipeline.
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
        Reapplies in order the trace produced by the first layer on a follower layer.
        Never recomputes scores or selections: it only uses the indices stored in the trace.
        Returns mut_k, mut_v and the updated mutable_frame_lengths on the follower.
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
        Executes a cleanup-only step for frames that are too small.
        The reducer supported here is remove_small_frames with the min_len parameter.
        Returns the generated ops, updated tensors, frame lengths and the changed flag.
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

        # Token-wise branch on value norm.
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

        # Pair-wise intra-frame branch.
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
        Reapplies a single trace op on a follower layer.
        The semantics of the pair reducers stay the global ones of the pipeline.
        Returns mut_k, mut_v and the updated mutable_frame_lengths after the op.
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