from __future__ import annotations

"""
Flexible Cache Layer Implementation.

This version keeps the public surface compatible with the current wrapper:
- update / prepare_frame_append / finalize_frame_append / get_frame_boundaries
- last_similarity_indices shared from layer 0 to follower layers
- sink / mutable / hat regions unchanged from the caller point of view

The main difference is internal organization:
- frame bookkeeping is delegated to FrameState
- intra/inter pruning is delegated to PruningPipeline
- global similarity stays as a lightweight local fallback
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers.cache_utils import CacheLayerMixin

from .frame_state import FrameState
from .pruning_pipeline import PruningPipeline, PruningStep, TraceOp
from .similarity_ops import (
    compute_adjacent_pair_scores_single,
    select_topk_non_overlapping_pairs,
)
from .offload_store import DiskOffloadStore
from .adakv import AdaKVConfig, AdaKVReducer


class FlexibleCacheLayer(CacheLayerMixin):
    """
    Unified cache layer with composable features controlled by configuration flags.
    The class keeps KV state and region logic while delegating pruning policy execution.
    Intra and inter pruning can now coexist inside the same ordered pipeline.
    """

    def __init__(
        self,
        mutable_token_size: int = 4096,
        use_sliding_window: bool = True,
        use_sink_tokens: bool = False,
        sink_sys_prompt: bool = True,
        size_sink_tokens: int = 100,
        use_hat_tokens: bool = False,
        hat_category: str = "tokens",
        hat_num_frames: int = 1,
        HAT_SIZE: int = 0,
        use_similarity: bool = False,
        similarity_merge: bool = False,
        similarity_keep_old: bool = False,
        intra_frame_pruning: bool = False,
        inter_frame_pruning: bool = False,
        is_first_layer: bool = False,
        index_strategy: str = "adiacent_cosine",
        layer_idx: int = 0,
        **kwargs,
    ):
        """
        Stores the public cache configuration used by the rest of the project.
        Builds the default pruning pipeline when frame-aware pruning is enabled.
        Initializes only lightweight state: KV tensors are created lazily on first update.
        """
        super().__init__()

        # Core configuration
        self.mutable_token_size = int(mutable_token_size)
        self.use_sliding_window = bool(use_sliding_window)
        # Opt-in metadata-only frame tracking. This enables the existing
        # visual-only finalize path and dynamic frame-based hat accounting
        # without activating any pruning/merging scorer.
        self.frame_aware_ingestion = bool(kwargs.get("frame_aware_ingestion", False))

        # Sink tokens
        self.use_sink_tokens = bool(use_sink_tokens)
        self.sink_sys_prompt = bool(sink_sys_prompt)
        self.SINK_SIZE = int(size_sink_tokens) if self.use_sink_tokens else 0

        # Hat tokens
        self.use_hat_tokens = bool(use_hat_tokens)
        self.hat_category = str(hat_category)
        self.hat_num_frames = int(hat_num_frames)
        self.HAT_SIZE = int(HAT_SIZE) if self.use_hat_tokens else 0

        # Similarity-based reduction
        self.use_similarity = bool(use_similarity)
        self.similarity_merge = bool(similarity_merge)
        self.similarity_keep_old = bool(similarity_keep_old)
        self.intra_frame_pruning = bool(intra_frame_pruning)
        self.inter_frame_pruning = bool(inter_frame_pruning)
        self.is_first_layer = bool(is_first_layer)
        self.index_strategy = str(index_strategy)

        # Ada-KV is isolated behind an explicit nested configuration.  Every
        # default keeps it disabled, preserving the legacy dispatch exactly.
        self.adakv_config = AdaKVConfig.from_dict(kwargs.get("adakv", None))
        self.adakv_enabled = bool(self.adakv_config.enabled)
        if self.adakv_enabled and (
            self.use_similarity or self.intra_frame_pruning or self.inter_frame_pruning
        ):
            raise ValueError("Ada-KV must be selected as a standalone cache policy")
        self.adakv_reducer = (
            AdaKVReducer(
                config=self.adakv_config,
                mutable_token_size=self.mutable_token_size,
                layer_idx=layer_idx,
            )
            if self.adakv_enabled else None
        )

        # Debug switches
        self.debug_cache_layout = bool(kwargs.get("debug_cache_layout", False))
        self.debug_tracking = bool(kwargs.get("debug_tracking", False))
        self.intra_chunk_size = int(kwargs.get("intra_chunk_size", 64))
        self.inter_chunk_size = int(kwargs.get("inter_chunk_size", 64))
        self.cleanup_min_frame_len = int(kwargs.get("cleanup_min_frame_len", 2))
        self.intra_remove_pct = kwargs.get("intra_remove_pct", None)
        self.intra_min_keep_per_frame = int(kwargs.get("intra_min_keep_per_frame", 1))
        self.sink_guidance_max_tokens = int(kwargs.get("sink_guidance_max_tokens", 0))
        self.sink_guidance_q_chunk_size = int(kwargs.get("sink_guidance_q_chunk_size", 128))
        self.sink_cross_attn_reduction = str(kwargs.get("sink_cross_attn_reduction", "mean"))

        # Disk offloading
        self.layer_idx = int(layer_idx)
        self.enable_disk_offload = bool(kwargs.get("enable_disk_offload", False))
        self.offload_dir = str(kwargs.get("offload_dir", "kv_offload"))
        self.offload_min_tokens = int(kwargs.get("offload_min_tokens", 1))
        self.offload_empty_cache = bool(kwargs.get("offload_empty_cache", True))

        # Quantized disk offloading.
        # Note: this quantization only concerns the offloaded storage.
        # It does not change the live cache used in the forward.
        self.offload_quantize = bool(kwargs.get("offload_quantize", False))
        self.offload_quant_bits = int(kwargs.get("offload_quant_bits", 8))
        self.offload_quant_scheme = str(kwargs.get("offload_quant_scheme", "per_tensor"))

        if self.adakv_enabled and self.enable_disk_offload:
            raise ValueError("Exact Ada-KV is incompatible with the current aligned-head offload store")

        self.offload_store = None
        if self.enable_disk_offload:
            self.offload_store = DiskOffloadStore(
                root_dir=self.offload_dir,
                layer_idx=self.layer_idx,
                quantize=self.offload_quantize,
                quant_bits=self.offload_quant_bits,
                quant_scheme=self.offload_quant_scheme,
            )

        # Internal KV state
        self.past_sequence_length = 0
        self.global_token_counter = 0
        self.is_initialized = False
        self.dtype = None
        self.device = None
        self.keys = None
        self.values = None

        # Shared trace from first layer to follower layers
        self.last_similarity_indices: Optional[Any] = None

        # Frame bookkeeping and ingestion cleanup
        self.frame_state = FrameState()
        self.qa_mode = False
        self.suspend_reduction = False
        self.sink_guidance_k: Optional[torch.Tensor] = None

        # Optional custom pipeline, otherwise build from flags
        custom_steps = kwargs.get("pruning_pipeline", None)
        self.pruning_pipeline = PruningPipeline(
            steps=custom_steps if custom_steps is not None else self._build_default_pipeline_steps(),
            similarity_keep_old=self.similarity_keep_old,
        )

    # ------------------------------------------------------------------
    # SMALL HELPERS / FLAGS
    # ------------------------------------------------------------------

    def _frame_pruning_enabled(self) -> bool:
        """
        Returns whether any frame-aware pruning logic is active.
        This is the single gate used by prepare/finalize and dynamic hat bookkeeping.
        It is true if either intra or inter pruning is enabled.
        """
        return (
            self.frame_aware_ingestion
            or self.intra_frame_pruning
            or self.inter_frame_pruning
            or self._has_storage_offload_step()
            or self.adakv_enabled
        )

    def _storage_steps(self):
        if not hasattr(self, "pruning_pipeline") or self.pruning_pipeline is None:
            return []
        return [
            s for s in self.pruning_pipeline.steps
            if s.enabled and s.domain == "storage" and s.reducer == "offload"
        ]

    def _has_storage_offload_step(self) -> bool:
        return bool(self.enable_disk_offload and len(self._storage_steps()) > 0)

    def capture_sink_guidance(self):
        """
        Capture the sink block after priming.

        Current implementation stores sink K as proxy guidance because the cache only
        receives K/V. For true StreamMem-style Q guidance, the model attention forward
        must pass query_states into cache.update().
        """
        if not self.use_sink_tokens or self.SINK_SIZE <= 0:
            return

        if self.keys is None or self.keys.numel() == 0:
            return

        if self.keys.shape[-2] < self.SINK_SIZE:
            return

        guidance = self.keys[:, :, :self.SINK_SIZE, :].detach().clone()

        if (
            self.sink_guidance_max_tokens > 0
            and guidance.shape[-2] > self.sink_guidance_max_tokens
        ):
            guidance = guidance[:, :, -self.sink_guidance_max_tokens:, :]

        self.sink_guidance_k = guidance


    def _get_sink_guidance_k(self):
        if self.sink_guidance_k is not None and self.sink_guidance_k.numel() > 0:
            return self.sink_guidance_k

        if not self.use_sink_tokens or self.SINK_SIZE <= 0:
            return None

        if (
            self.keys is None
            or self.keys.numel() == 0
            or self.keys.shape[-2] < self.SINK_SIZE
        ):
            return None

        guidance = self.keys[:, :, :self.SINK_SIZE, :].detach()

        if (
            self.sink_guidance_max_tokens > 0
            and guidance.shape[-2] > self.sink_guidance_max_tokens
        ):
            guidance = guidance[:, :, -self.sink_guidance_max_tokens:, :]

        return guidance


    def _split_frame_lengths_from_left(self, frame_lengths, tokens_to_split: int):
        frame_lengths = [int(x) for x in frame_lengths if int(x) > 0]
        tokens_to_split = int(tokens_to_split)

        offloaded = []
        remaining_lengths = []
        left = tokens_to_split

        for fl in frame_lengths:
            fl = int(fl)

            if left <= 0:
                remaining_lengths.append(fl)
                continue

            if left >= fl:
                offloaded.append(fl)
                left -= fl
                continue

            # partial split of the frame, as already happens with the left trim
            offloaded.append(left)
            remaining_lengths.append(fl - left)
            left = 0

        return (
            [int(x) for x in offloaded if int(x) > 0],
            [int(x) for x in remaining_lengths if int(x) > 0],
        )

    def _apply_storage_offload_steps(
        self,
        mut_k: torch.Tensor,
        mut_v: torch.Tensor,
        mutable_frame_lengths,
        target_len: int,
    ):
        if not self._has_storage_offload_step():
            return mut_k, mut_v, mutable_frame_lengths

        if self.offload_store is None:
            return mut_k, mut_v, mutable_frame_lengths

        for step in self._storage_steps():
            if mut_k.shape[-2] <= int(target_len):
                break

            params = step.params or {}
            scope = str(params.get("scope", "left_excess"))

            if scope != "left_excess":
                raise ValueError(f"Unsupported offload scope: {scope}")

            overflow = int(mut_k.shape[-2]) - int(target_len)
            if overflow <= 0:
                continue

            max_tokens = params.get("max_tokens", None)
            tokens_to_offload = overflow
            if max_tokens is not None:
                tokens_to_offload = min(tokens_to_offload, int(max_tokens))

            tokens_to_offload = max(0, int(tokens_to_offload))
            if tokens_to_offload < int(self.offload_min_tokens):
                continue

            off_k = mut_k[..., :tokens_to_offload, :].detach()
            off_v = mut_v[..., :tokens_to_offload, :].detach()

            offloaded_lengths, remaining_lengths = self._split_frame_lengths_from_left(
                mutable_frame_lengths,
                tokens_to_offload,
            )

            quantize = params.get("quantize", None)
            quant_bits = params.get("bits", params.get("quant_bits", None))
            quant_scheme = params.get("scheme", params.get("quant_scheme", None))

            self.offload_store.append(
                keys=off_k,
                values=off_v,
                frame_lengths=offloaded_lengths,
                meta={
                    "step_name": step.name,
                    "layer_idx": self.layer_idx,
                    "tokens_to_offload": tokens_to_offload,
                    "target_len": int(target_len),
                    "live_len_before": int(mut_k.shape[-2]),
                    "quantize": quantize,
                    "quant_bits": quant_bits,
                    "quant_scheme": quant_scheme,
                },
                quantize=quantize,
                quant_bits=quant_bits,
                quant_scheme=quant_scheme,
            )

            mut_k = mut_k[..., tokens_to_offload:, :]
            mut_v = mut_v[..., tokens_to_offload:, :]
            mutable_frame_lengths = remaining_lengths

            if self.offload_empty_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()

        return mut_k, mut_v, mutable_frame_lengths

    def get_offload_stats(self):
        if self.offload_store is None:
            return {
                "offloaded_tokens": 0,
                "offloaded_chunks": 0,
                "offloaded_physical_bytes": 0,
                "offloaded_equivalent_bytes": 0,
                "offloaded_compression_ratio": 0.0,
            }
        return self.offload_store.get_stats()

    def _build_default_pipeline_steps(self) -> List[PruningStep]:
        """
        Builds a deterministic default pipeline from the legacy boolean flags.
        If both intra and inter are enabled, the order is cleanup -> intra -> cleanup -> inter.
        The result is a list of PruningStep objects consumed by PruningPipeline.
        """
        steps: List[PruningStep] = []

        if self._frame_pruning_enabled():
            steps.append(
                PruningStep(
                    name="cleanup_tiny_pre",
                    domain="cleanup",
                    reducer="remove_small_frames",
                    params={"min_len": self.cleanup_min_frame_len},
                )
            )

        if self.intra_frame_pruning:
            intra_params = {
                "chunk_size": self.intra_chunk_size,
                "min_keep_per_frame": self.intra_min_keep_per_frame,
                "q_chunk_size": self.sink_guidance_q_chunk_size,
                "reduction": self.sink_cross_attn_reduction,
            }
            if self.intra_remove_pct is not None:
                intra_params["remove_pct"] = float(self.intra_remove_pct)

            sink_guided_scorers = {
                "sink_cross_attention",
                "sink_guided_attention",
                "sys_prompt_cross_attention",
                "streammem_attention",
            }

            intra_reducer = (
                "drop"
                if str(self.index_strategy).lower() in sink_guided_scorers
                else "merge" if self.similarity_merge else "drop"
            )

            steps.append(
                PruningStep(
                    name="intra_pass",
                    domain="intra",
                    scorer=self.index_strategy,
                    reducer=intra_reducer,
                    params=intra_params,
                )
            )
            steps.append(
                PruningStep(
                    name="cleanup_tiny_after_intra",
                    domain="cleanup",
                    reducer="remove_small_frames",
                    params={"min_len": self.cleanup_min_frame_len},
                )
            )

        if self.inter_frame_pruning:
            steps.append(
                PruningStep(
                    name="inter_pass",
                    domain="inter",
                    scorer=self.index_strategy,
                    reducer="drop",
                    params={"chunk_size": self.inter_chunk_size},
                )
            )
            steps.append(
                PruningStep(
                    name="cleanup_tiny_after_inter",
                    domain="cleanup",
                    reducer="remove_small_frames",
                    params={"min_len": self.cleanup_min_frame_len},
                )
            )

        return steps

    def set_QA(self, enabled: bool = True):
        """
        Enables or disables QA mode.
        In QA mode the layer stops emitting pruning traces and never reduces newly appended KV.
        This is used by the outer pipeline when generation should consume the frozen cache as-is.
        """
        self.qa_mode = bool(enabled)
        if self.qa_mode:
            self.last_similarity_indices = None

    def lazy_initialization(self, key_states: torch.Tensor):
        """
        Creates the internal empty KV tensors matching dtype and device of the first update.
        This keeps construction cheap and avoids having to know the runtime device upfront.
        After this call, keys and values are valid empty tensors ready for concatenation.
        """
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def get_hat_size(self):
        """
        Returns the current hat size used for capacity accounting.
        If hat is frame-based and frame pruning is enabled, the size is computed dynamically.
        Otherwise the static HAT_SIZE configured at construction time is returned.
        """
        return self.frame_state.get_hat_size(
            use_hat_tokens=self.use_hat_tokens,
            hat_category=self.hat_category,
            hat_num_frames=self.hat_num_frames,
            static_hat_size=self.HAT_SIZE,
            frame_pruning_enabled=self._frame_pruning_enabled(),
        )

    def add_frame_tokens(self, frame_token_counts):
        """
        Registers newly appended frame token counts in the tracking state.
        This is used by non-frame-aware ingestion paths that do not go through prepare/finalize.
        The history is stored in temporal order and later split into mutable/hat sections.
        """
        self.frame_state.add_frame_tokens(frame_token_counts)

    # ------------------------------------------------------------------
    # DEBUG / BOOKKEEPING HELPERS
    # ------------------------------------------------------------------

    def _split_frame_lengths_by_hat(self):
        """
        Splits tracked frame lengths into mutable and hat sections.
        The split only depends on current tracking state plus hat configuration.
        Returns two lists: mutable_lengths and hat_lengths.
        """
        return self.frame_state.split_frame_lengths_by_hat(
            use_hat_tokens=self.use_hat_tokens,
            hat_category=self.hat_category,
            hat_num_frames=self.hat_num_frames,
            frame_pruning_enabled=self._frame_pruning_enabled(),
        )

    def _set_frame_lengths(self, mutable_lengths, hat_lengths):
        """
        Replaces the currently tracked frame lengths with the provided state.
        The expected convention is mutable frames followed by hat frames.
        This method is used after truncation and after similarity reduction replay.
        """
        self.frame_state.set_frame_lengths(mutable_lengths, hat_lengths)

    def _trim_frame_lengths_from_left(self, frame_lengths, tokens_to_trim):
        """
        Simulates a temporal trim on frame lengths without touching tensors directly.
        Tokens are removed from the oldest frames first until the requested budget is consumed.
        Returns a new list of frame lengths that mirrors a left trim on mutable KV.
        """
        return self.frame_state.trim_frame_lengths_from_left(frame_lengths, tokens_to_trim)

    def _frame_offsets(self, frame_lengths):
        """
        Converts frame lengths into contiguous [start, end) offsets.
        Offsets are relative to the specific region passed to this function.
        Returns a list of tuples aligned with the order of frame_lengths.
        """
        return self.frame_state.frame_offsets(frame_lengths)

    def get_frame_boundaries(self):
        """
        Builds a user-friendly description of frame boundaries in current cache layout.
        Boundaries are absolute inside the full cache and already account for sink offset.
        Returns a list of dictionaries used by the external debug prints.
        """
        return self.frame_state.get_frame_boundaries(
            sink_size=self.SINK_SIZE,
            use_hat_tokens=self.use_hat_tokens,
            hat_category=self.hat_category,
            hat_num_frames=self.hat_num_frames,
            frame_pruning_enabled=self._frame_pruning_enabled(),
        )

    def _debug_print_frame_layout(self, tag: str):
        """
        Prints a readable snapshot of mutable/hat distribution after a cache event.
        It includes totals, per-frame lengths and explicit boundaries for inspection.
        Output is emitted only when debug_cache_layout is enabled.
        """
        if not self._frame_pruning_enabled() or not self.debug_cache_layout:
            return

        mutable_lengths, hat_lengths = self._split_frame_lengths_by_hat()
        mutable_total = int(sum(mutable_lengths))
        hat_total = int(sum(hat_lengths))
        current_hat_size = int(self.get_hat_size())

        print(
            f"[FlexibleCacheLayer][{tag}] "
            f"sink={self.SINK_SIZE} | mutable_total={mutable_total} | "
            f"hat_total={hat_total} | hat_size={current_hat_size} | "
            f"cache_seq={self.get_seq_length()}",
            flush=True,
        )

        print(
            f"[FlexibleCacheLayer][{tag}] mutable_frames={mutable_lengths}",
            flush=True,
        )

        if len(hat_lengths) > 0:
            print(
                f"[FlexibleCacheLayer][{tag}] hat_frames={hat_lengths}",
                flush=True,
            )

        for b in self.get_frame_boundaries():
            print(
                f"[FlexibleCacheLayer][{tag}] "
                f"frame_idx={b['frame_idx']} | region={b['region']} | "
                f"len={b['length']} | abs=[{b['abs_start']},{b['abs_end']})",
                flush=True,
            )

    def _debug_physical_vs_tracked(self, tag: str):
        """
        Verifies that physical mutable tokens match tracked mutable frame lengths.
        This is useful to catch bugs in replay, cleanup or finalization logic.
        Output is emitted only when debug_tracking is enabled.
        """
        if not self._frame_pruning_enabled() or not self.debug_tracking:
            return
        # Exact Ada-KV has head-specific mutable lengths represented with
        # padding, so one shared frame-length sum is not a valid invariant.
        if self.adakv_enabled:
            return

        hat_size = int(self.get_hat_size())
        physical_mutable = self.get_seq_length() - int(self.SINK_SIZE) - hat_size

        mutable_lengths, hat_lengths = self._split_frame_lengths_by_hat()
        tracked_mutable = int(sum(mutable_lengths))
        tracked_hat = int(sum(hat_lengths))

        print(
            f"[TRACK CHECK][{tag}] "
            f"seq={self.get_seq_length()} | "
            f"sink={self.SINK_SIZE} | "
            f"hat_size={hat_size} | "
            f"physical_mutable={physical_mutable} | "
            f"tracked_mutable={tracked_mutable} | "
            f"tracked_hat={tracked_hat} | "
            f"gap={physical_mutable - tracked_mutable}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # PENDING BATCH CLEANUP FOR FRAME-AWARE INGESTION
    # ------------------------------------------------------------------

    def prepare_frame_append(
        self,
        frame_token_counts,
        visual_pos_mask: torch.Tensor,
        appended_seq_len: int,
    ):
        """
        Stores the metadata needed to keep only visual tokens after an append.
        This method validates the visual mask against the expected number of frame tokens.
        It does not touch KV yet: the actual filtering happens in finalize_frame_append.
        """
        if not self._frame_pruning_enabled():
            return

        self.frame_state.prepare_frame_append(
            frame_token_counts=frame_token_counts,
            visual_pos_mask=visual_pos_mask,
            appended_seq_len=appended_seq_len,
        )
        self.suspend_reduction = True

    def finalize_frame_append(self):
        """
        Keeps only the visual portion of the most recent append in keys and values.
        After filtering it updates tracked frame lengths and re-enables reduction logic.
        This is the hook used by the video ingestion path before capacity enforcement.
        """
        if not self._frame_pruning_enabled():
            return

        pending_indices = self.frame_state.pending_visual_indices
        appended_seq_len = int(self.frame_state.pending_total_seq_len)
        prefix_len = max(0, int(self.keys.shape[-2]) - appended_seq_len)

        self.keys, self.values = self.frame_state.finalize_frame_append_to_visual_only(
            keys=self.keys,
            values=self.values,
        )
        if self.adakv_enabled:
            if pending_indices is None:
                raise RuntimeError("Ada-KV frame finalization requires visual token indices")
            self.adakv_reducer.finalize_visual_append(
                prefix_len=prefix_len,
                keep_indices=pending_indices,
                appended_seq_len=appended_seq_len,
            )
        self.suspend_reduction = False

        self._debug_physical_vs_tracked("after_finalize_frame_append")
        self._debug_print_frame_layout("after_finalize_frame_append")

    # ------------------------------------------------------------------
    # CORE UPDATE
    # ------------------------------------------------------------------

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Appends new KV to the cache and enforces capacity if needed.
        It preserves the current public behavior used by the outer FlexibleCache wrapper.
        When overflow happens it delegates to either global reduction or frame-aware pipeline.
        """
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        if self.adakv_enabled:
            query_states = (cache_kwargs or {}).get("query_states", None)
            self.adakv_reducer.append(key_states=key_states, query_states=query_states)

        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)
        self.past_sequence_length += key_states.shape[-2]

        if self.qa_mode:
            self.last_similarity_indices = None
            return self.keys, self.values

        if self.suspend_reduction:
            self.last_similarity_indices = None
            return self.keys, self.values

        if not self.use_sliding_window:
            return self.keys, self.values

        current_len = self.keys.shape[-2]
        total_capacity = self.mutable_token_size + self.SINK_SIZE + self.get_hat_size()

        if current_len <= total_capacity:
            self.last_similarity_indices = None
            return self.keys, self.values

        if self.adakv_enabled:
            self._apply_adakv_reduction()
        elif self.use_similarity:
            self._apply_similarity_reduction(cache_kwargs)
        else:
            self._apply_simple_truncation()

        self._debug_print_frame_layout("after_reduction")
        return self.keys, self.values

    # ------------------------------------------------------------------
    # ADA-KV (isolated exact path)
    # ------------------------------------------------------------------

    def needs_adakv_reduction(self) -> bool:
        if not self.adakv_enabled:
            return False
        return self.adakv_reducer.needs_reduction(
            sink_size=int(self.SINK_SIZE),
            hat_size=int(self.get_hat_size()),
        )

    def _apply_adakv_reduction(self):
        if not self.adakv_enabled:
            return
        if self.qa_mode or self.suspend_reduction:
            return
        if not self.needs_adakv_reduction():
            return
        self.keys, self.values = self.adakv_reducer.reduce(
            keys=self.keys,
            values=self.values,
            sink_size=int(self.SINK_SIZE),
            hat_size=int(self.get_hat_size()),
        )
        self.last_similarity_indices = None

        every = max(0, int(self.adakv_config.log_every_reductions))
        if self.is_first_layer and (
            self.adakv_reducer.reduction_count == 1
            or (every > 0 and self.adakv_reducer.reduction_count % every == 0)
        ):
            stats = self.adakv_reducer.get_stats()
            b = stats["head_budget"]
            print(
                f"[AdaKV] reductions={stats['reduction_count']} | "
                f"Hkv={stats['num_kv_heads']} | total_mutable={b['sum']} | "
                f"head_budget_mean/min/max={b['mean']:.2f}/{b['min']}/{b['max']} | "
                f"physical_mutable_len={stats['physical_mutable_length']} | "
                f"avg_reduction_sec={stats['average_reduction_time_sec']:.6f}",
                flush=True,
            )

    def get_adakv_valid_mask(self):
        if not self.adakv_enabled or self.adakv_reducer is None:
            return None
        return self.adakv_reducer.valid_mask

    def get_adakv_stats(self):
        if not self.adakv_enabled or self.adakv_reducer is None:
            return None
        return self.adakv_reducer.get_stats()

    # ------------------------------------------------------------------
    # SIMPLE TRUNCATION
    # ------------------------------------------------------------------

    def _apply_simple_truncation(self):
        """
        Applies a pure left trim on mutable KV when similarity is disabled.
        In frame-aware mode it also keeps frame lengths consistent with the trim.
        The sink and hat regions remain untouched by construction.
        """
        sink_k, sink_v, mut_k, mut_v, hat_k, hat_v = self._extract_regions()

        if not self._frame_pruning_enabled():
            if mut_k.shape[-2] > self.mutable_token_size:
                mut_k = mut_k[:, :, -self.mutable_token_size:, :]
                mut_v = mut_v[:, :, -self.mutable_token_size:, :]
            self._reassemble_cache(sink_k, sink_v, mut_k, mut_v, hat_k, hat_v)
            self.last_similarity_indices = None
            return

        mutable_frame_lengths, hat_frame_lengths = self._split_frame_lengths_by_hat()

        if mut_k.shape[-2] > self.mutable_token_size:
            if self._has_storage_offload_step():
                mut_k, mut_v, mutable_frame_lengths = self._apply_storage_offload_steps(
                    mut_k=mut_k,
                    mut_v=mut_v,
                    mutable_frame_lengths=mutable_frame_lengths,
                    target_len=int(self.mutable_token_size),
                )
            else:
                to_trim = mut_k.shape[-2] - self.mutable_token_size
                mut_k = mut_k[:, :, -self.mutable_token_size:, :]
                mut_v = mut_v[:, :, -self.mutable_token_size:, :]
                mutable_frame_lengths = self._trim_frame_lengths_from_left(
                    mutable_frame_lengths,
                    to_trim,
                )

        self._reassemble_cache(sink_k, sink_v, mut_k, mut_v, hat_k, hat_v)
        self._set_frame_lengths(mutable_frame_lengths, hat_frame_lengths)
        self.last_similarity_indices = None
        self._debug_physical_vs_tracked("after_simple_truncation")

    # ------------------------------------------------------------------
    # GLOBAL SIMILARITY
    # ------------------------------------------------------------------

    def _compute_global_pair_indices(self, mut_k: torch.Tensor, to_remove: int) -> torch.Tensor:
        """
        Computes legacy global pair indices on the full mutable region.
        It uses the same pair scorers as before and applies non-overlapping selection.
        The returned indices are pair indices relative to mutable KV.
        """
        to_remove = int(to_remove)
        if to_remove <= 0 or mut_k.shape[-2] <= 1:
            return torch.tensor([], device=mut_k.device, dtype=torch.long)

        pair_scores = compute_adjacent_pair_scores_single(
            k_block=mut_k,
            strategy=self.index_strategy,
            similarity_keep_old=self.similarity_keep_old,
        )
        return select_topk_non_overlapping_pairs(pair_scores, to_remove)

    def _apply_global_similarity_reduction(self, cache_kwargs: Optional[Dict[str, Any]] = None):
        """
        Executes the legacy non-frame-aware similarity reduction on the mutable block.
        First layer computes pair indices and follower layers can reuse the provided indices.
        The resulting cache layout is reassembled back with unchanged sink and hat blocks.
        """
        sink_k, sink_v, mut_k, mut_v, hat_k, hat_v = self._extract_regions()
        target_len = int(self.mutable_token_size)
        to_remove = int(mut_k.shape[-2] - target_len)

        if to_remove <= 0:
            self.last_similarity_indices = None
            return

        if self.is_first_layer:
            indices = self._compute_global_pair_indices(mut_k, to_remove)
            self.last_similarity_indices = indices
        else:
            indices = cache_kwargs.get("similarity_indices") if cache_kwargs else None
            if indices is None:
                indices = self._compute_global_pair_indices(mut_k, to_remove)

        if indices is None or (isinstance(indices, torch.Tensor) and indices.numel() == 0):
            self.last_similarity_indices = indices
            return

        indices = indices.to(mut_k.device)

        if self.similarity_merge:
            if self.similarity_keep_old:
                merge_into = indices
                merge_from = indices + 1
            else:
                merge_into = indices + 1
                merge_from = indices

            mut_v = mut_v.clone()
            mut_v.index_copy_(
                -2,
                merge_into,
                0.5 * (
                    mut_v.index_select(-2, merge_from) +
                    mut_v.index_select(-2, merge_into)
                ),
            )

            keep = torch.ones(mut_k.shape[-2], device=mut_k.device, dtype=torch.bool)
            keep[merge_from] = False
            mut_k = mut_k[..., keep, :]
            mut_v = mut_v[..., keep, :]
        else:
            if self.similarity_keep_old:
                drop_indices = indices + 1
            else:
                drop_indices = indices

            keep = torch.ones(mut_k.shape[-2], device=mut_k.device, dtype=torch.bool)
            keep[drop_indices] = False
            mut_k = mut_k[..., keep, :]
            mut_v = mut_v[..., keep, :]

        self._reassemble_cache(sink_k, sink_v, mut_k, mut_v, hat_k, hat_v)

    # ------------------------------------------------------------------
    # TRACE / PIPELINE HELPERS
    # ------------------------------------------------------------------

    def _normalize_trace_ops(self, ops: Any) -> Optional[List[TraceOp]]:
        """
        Converts the incoming follower-layer trace into a normalized list of TraceOp.
        This keeps replay robust even if upstream passes dataclasses or plain dictionaries.
        Returns None for unsupported legacy tensor traces, which are handled elsewhere.
        """
        if ops is None:
            return None

        if isinstance(ops, torch.Tensor):
            return None

        normalized: List[TraceOp] = []

        for op in ops:
            if isinstance(op, TraceOp):
                normalized.append(op)
                continue

            if isinstance(op, dict):
                pair_indices = op.get("pair_indices", None)
                removed_indices = op.get("removed_indices", None)

                if isinstance(pair_indices, torch.Tensor):
                    pair_indices = pair_indices.detach().cpu()
                if isinstance(removed_indices, torch.Tensor):
                    removed_indices = removed_indices.detach().cpu()

                normalized.append(
                    TraceOp(
                        step_name=op.get("step_name", op.get("type", "legacy_op")),
                        op_type=op.get("op_type", op.get("type", "remove_tokens")),
                        pair_indices=pair_indices,
                        removed_indices=removed_indices,
                        meta=op.get("meta", {}),
                    )
                )
                continue

            raise ValueError(f"Unsupported trace op type: {type(op)}")

        return normalized

    # ------------------------------------------------------------------
    # SIMILARITY REDUCTION ENTRYPOINT
    # ------------------------------------------------------------------

    def _apply_similarity_reduction(self, cache_kwargs: Optional[Dict[str, Any]] = None):
        """
        Main similarity entrypoint used by update and finalize paths.
        Global mode uses the legacy lightweight reducer, frame-aware mode uses PruningPipeline.
        The produced trace is stored in last_similarity_indices for follower layers.
        """
        if not self._frame_pruning_enabled():
            self._apply_global_similarity_reduction(cache_kwargs)
            return

        sink_k, sink_v, mut_k, mut_v, hat_k, hat_v = self._extract_regions()
        target_len = int(self.mutable_token_size)
        to_remove = int(mut_k.shape[-2] - target_len)

        if to_remove <= 0:
            self.last_similarity_indices = None
            return

        mutable_frame_lengths, hat_frame_lengths = self._split_frame_lengths_by_hat()

        if self.is_first_layer:
            trace_ops, mut_k, mut_v, mutable_frame_lengths = self.pruning_pipeline.execute_first_layer(
                mut_k=mut_k,
                mut_v=mut_v,
                hat_k=hat_k,
                mutable_frame_lengths=mutable_frame_lengths,
                hat_frame_lengths=hat_frame_lengths,
                target_len=target_len,
                allow_fallback_trim=not self._has_storage_offload_step(),
                guidance_k=self._get_sink_guidance_k(),
            )
            self.last_similarity_indices = trace_ops
        else:
            ops = cache_kwargs.get("similarity_indices") if cache_kwargs else None
            trace_ops = self._normalize_trace_ops(ops)

            if trace_ops is None:
                trace_ops, mut_k, mut_v, mutable_frame_lengths = self.pruning_pipeline.execute_first_layer(
                    mut_k=mut_k,
                    mut_v=mut_v,
                    hat_k=hat_k,
                    mutable_frame_lengths=mutable_frame_lengths,
                    hat_frame_lengths=hat_frame_lengths,
                    target_len=target_len,
                    allow_fallback_trim=not self._has_storage_offload_step(),
                    guidance_k=self._get_sink_guidance_k(),
                )
            else:
                mut_k, mut_v, mutable_frame_lengths = self.pruning_pipeline.replay_on_follower(
                    mut_k=mut_k,
                    mut_v=mut_v,
                    mutable_frame_lengths=mutable_frame_lengths,
                    trace_ops=trace_ops,
                )

            self.last_similarity_indices = trace_ops
        mut_k, mut_v, mutable_frame_lengths = self._apply_storage_offload_steps(
            mut_k=mut_k,
            mut_v=mut_v,
            mutable_frame_lengths=mutable_frame_lengths,
            target_len=target_len,
        )
        
        self._reassemble_cache(sink_k, sink_v, mut_k, mut_v, hat_k, hat_v)
        self._set_frame_lengths(mutable_frame_lengths, hat_frame_lengths)
        self._debug_physical_vs_tracked("after_similarity_reduction")

    # ------------------------------------------------------------------
    # REGION HELPERS
    # ------------------------------------------------------------------

    def _extract_regions(self):
        """
        Splits the current cache into sink, mutable and hat regions.
        Mutable is always the middle section that pruning is allowed to modify.
        Returns six tensors in the fixed order sink_k, sink_v, mut_k, mut_v, hat_k, hat_v.
        """
        if self.use_sink_tokens:
            sink_k = self.keys[:, :, :self.SINK_SIZE, :]
            sink_v = self.values[:, :, :self.SINK_SIZE, :]
            remaining_start = self.SINK_SIZE
        else:
            sink_k = torch.tensor([], dtype=self.dtype, device=self.device)
            sink_v = torch.tensor([], dtype=self.dtype, device=self.device)
            remaining_start = 0

        current_hat_size = int(self.get_hat_size())
        if self.use_hat_tokens and current_hat_size > 0:
            hat_k = self.keys[:, :, -current_hat_size:, :]
            hat_v = self.values[:, :, -current_hat_size:, :]
            remaining_end = -current_hat_size
        else:
            hat_k = torch.tensor([], dtype=self.dtype, device=self.device)
            hat_v = torch.tensor([], dtype=self.dtype, device=self.device)
            remaining_end = None

        mut_k = self.keys[:, :, remaining_start:remaining_end, :]
        mut_v = self.values[:, :, remaining_start:remaining_end, :]

        return sink_k, sink_v, mut_k, mut_v, hat_k, hat_v

    def _reassemble_cache(
        self,
        sink_k: torch.Tensor,
        sink_v: torch.Tensor,
        mut_k: torch.Tensor,
        mut_v: torch.Tensor,
        hat_k: torch.Tensor,
        hat_v: torch.Tensor,
    ):
        """
        Rebuilds the full cache from sink, mutable and hat parts.
        The concatenation order is always sink then mutable then hat.
        This method updates self.keys and self.values in place.
        """
        parts_k = []
        parts_v = []

        if sink_k.numel() > 0:
            parts_k.append(sink_k)
            parts_v.append(sink_v)

        parts_k.append(mut_k)
        parts_v.append(mut_v)

        if hat_k.numel() > 0:
            parts_k.append(hat_k)
            parts_v.append(hat_v)

        self.keys = torch.cat(parts_k, dim=-2)
        self.values = torch.cat(parts_v, dim=-2)

    # ------------------------------------------------------------------
    # RETRIEVAL HELPERS
    # ------------------------------------------------------------------

    def get_mutable_for_retrieve(self):
        """
        Exposes the current mutable region for the QA-time retrievers.

        Returns:
            mut_k, mut_v, mutable_frame_lengths

        Sink and hat are excluded: they never take part in the retrieval selection.
        """
        if self.adakv_enabled:
            raise RuntimeError(
                "The current ReKV retriever assumes token alignment across heads and "
                "is not compatible with exact head-specific Ada-KV retention."
            )
        _sink_k, _sink_v, mut_k, mut_v, _hat_k, _hat_v = self._extract_regions()
        mutable_lengths, _hat_lengths = self._split_frame_lengths_by_hat()
        return mut_k, mut_v, mutable_lengths

    def rebuild_mutable_for_retrieve(
        self,
        new_mut_k: torch.Tensor,
        new_mut_v: torch.Tensor,
        new_mutable_frame_lengths,
    ) -> int:
        """
        Rebuilds the mutable region after query-aware retrieval.

        Final layout:
            sink + new_mutable + hat

        Sink and hat are preserved regardless.
        """
        if new_mut_k is None or new_mut_v is None:
            return 0

        sink_k, sink_v, _old_mut_k, _old_mut_v, hat_k, hat_v = self._extract_regions()

        if self.device is not None:
            new_mut_k = new_mut_k.to(device=self.device)
            new_mut_v = new_mut_v.to(device=self.device)
        if self.dtype is not None:
            new_mut_k = new_mut_k.to(dtype=self.dtype)
            new_mut_v = new_mut_v.to(dtype=self.dtype)

        _, hat_lengths = self._split_frame_lengths_by_hat()
        new_mutable_frame_lengths = [
            int(x) for x in new_mutable_frame_lengths if int(x) > 0
        ]

        self._reassemble_cache(
            sink_k=sink_k,
            sink_v=sink_v,
            mut_k=new_mut_k,
            mut_v=new_mut_v,
            hat_k=hat_k,
            hat_v=hat_v,
        )
        self._set_frame_lengths(new_mutable_frame_lengths, hat_lengths)

        self.last_similarity_indices = None
        return int(new_mut_k.shape[-2])

    # ------------------------------------------------------------------
    # CACHE API
    # ------------------------------------------------------------------

    def get_seq_length(self) -> int:
        """
        Returns the physical number of KV tokens currently stored in the layer.
        The method is safe before initialization and simply returns zero in that case.
        It is the capacity reference used by the outer wrapper and by debug prints.
        """
        if not self.is_initialized or self.keys.numel() == 0:
            return 0
        return int(self.keys.shape[-2])

    def get_max_cache_shape(self) -> int:
        """
        Returns the maximum cache size the layer is allowed to hold.
        In infinite mode a huge sentinel value is returned to preserve old behavior.
        In sliding-window mode the size includes mutable, sink and dynamic hat.
        """
        if not self.use_sliding_window:
            return 2_000_000_000
        return int(self.mutable_token_size) + int(self.SINK_SIZE) + int(self.get_hat_size())

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """
        Computes mask sizes expected by the caller for current cache plus current query.
        kv_offset remains zero in this implementation because physical cache is contiguous.
        Returns the pair (kv_length, kv_offset) matching the previous public contract.
        """
        kv_offset = 0
        query_length = int(cache_position.shape[0])
        kv_length = int(self.get_seq_length()) + query_length
        return kv_length, kv_offset
