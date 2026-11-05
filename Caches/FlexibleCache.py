"""
Flexible Cache - Multi-layer wrapper for FlexibleCacheLayer

Coordinates similarity computation across layers:
- Layer 0 computes similarity indices
- Follower layers reuse those indices for consistency
"""

import torch
from typing import Any, Optional, Dict
from collections.abc import Iterable
from transformers.cache_utils import Cache
from .FlexibleCacheLayer import FlexibleCacheLayer


class FlexibleCache(Cache):
    """
    Multi-layer cache that coordinates FlexibleCacheLayer instances.
    
    Layer 0 computes similarity indices (if enabled), and all other layers
    reuse the same indices to ensure consistent token reduction across layers.
    """
    
    def __init__(
        self,
        config: Optional[Any] = None,
        offloading: bool = False,
        offload_only_non_sliding: bool = False,
        **cache_config
    ):
        """
        Initialize multi-layer flexible cache.
        
        Args:
            config: Model configuration (to determine number of layers)
            offloading: Enable offloading to CPU
            offload_only_non_sliding: Only offload non-sliding layers
            **cache_config: Cache configuration flags (mutable_token_size, use_sliding_window, etc.)
        """
        layers = []
        
        if config is not None:
            # Try to get decoder config, fallback to config itself
            if hasattr(config, 'get_text_config'):
                decoder_config = config.get_text_config(decoder=True)
            elif hasattr(config, 'text_config'):
                decoder_config = config.text_config
            else:
                decoder_config = config
            
            # Get number of layers
            num_layers = decoder_config.num_hidden_layers
            num_kv_shared = getattr(decoder_config, 'num_kv_shared_layers', 0)
            
            # Create layers
            num_actual_layers = num_layers - num_kv_shared if num_kv_shared > 0 else num_layers
            for idx in range(num_actual_layers):
                # Layer 0 computes similarity, others follow
                layer = FlexibleCacheLayer(
                    is_first_layer=(idx == 0),
                    layer_idx=idx,
                    **cache_config
                )
                layers.append(layer)
        
        super().__init__(
            layers=layers,
            offloading=offloading,
            offload_only_non_sliding=offload_only_non_sliding
        )
        
        # Store config for later access
        self.cache_config = cache_config
    
    def get_offload_stats(self):
        total = {
            "offloaded_tokens": 0,
            "offloaded_chunks": 0,
            "offloaded_physical_bytes": 0,
            "offloaded_equivalent_bytes": 0,
            "offloaded_compression_ratio": 0.0,
            "layers": {},
        }

        for idx, layer in enumerate(self.layers):
            if hasattr(layer, "get_offload_stats"):
                stats = layer.get_offload_stats()
            else:
                stats = {}

            total["layers"][str(idx)] = stats
            total["offloaded_tokens"] += int(stats.get("offloaded_tokens", 0))
            total["offloaded_chunks"] += int(stats.get("offloaded_chunks", 0))
            total["offloaded_physical_bytes"] += int(stats.get("offloaded_physical_bytes", 0))
            total["offloaded_equivalent_bytes"] += int(stats.get("offloaded_equivalent_bytes", 0))

        if total["offloaded_physical_bytes"] > 0:
            total["offloaded_compression_ratio"] = (
                total["offloaded_equivalent_bytes"] / total["offloaded_physical_bytes"]
            )

        return total

    def get_offloaded_token_count(self):
        return int(self.get_offload_stats().get("offloaded_tokens", 0))

    def get_offloaded_memory_size(self):
        return int(self.get_offload_stats().get("offloaded_physical_bytes", 0))


    def get_adakv_valid_mask(self, layer_idx: int = 0):
        if not self.layers:
            return None
        layer = self.layers[int(layer_idx)]
        if hasattr(layer, "get_adakv_valid_mask"):
            return layer.get_adakv_valid_mask()
        return None

    def get_adakv_stats(self):
        enabled_layers = []
        for idx, layer in enumerate(self.layers):
            if getattr(layer, "adakv_enabled", False) and hasattr(layer, "get_adakv_stats"):
                stats = layer.get_adakv_stats()
                if stats is not None:
                    enabled_layers.append((idx, stats))
        if not enabled_layers:
            return None
        total_reductions = sum(int(x[1].get("reduction_count", 0)) for x in enabled_layers)
        total_time = sum(
            float(x[1].get("average_reduction_time_sec", 0.0))
            * int(x[1].get("reduction_count", 0))
            for x in enabled_layers
        )
        base = dict(enabled_layers[0][1])
        budget_summaries = [st.get("head_budget", {}) for _, st in enabled_layers]
        layer_budget_sums = [int(x.get("sum", 0)) for x in budget_summaries]
        layer_budget_means = [float(x.get("mean", 0.0)) for x in budget_summaries]
        base["num_layers"] = len(enabled_layers)
        base["head_budget_across_layers"] = {
            "mean_of_layer_means": (
                sum(layer_budget_means) / len(layer_budget_means)
                if layer_budget_means else 0.0
            ),
            "min": min((int(x.get("min", 0)) for x in budget_summaries), default=0),
            "max": max((int(x.get("max", 0)) for x in budget_summaries), default=0),
            "sum_all_layers": sum(layer_budget_sums),
        }
        base["retained_mutable_entries_all_layers"] = sum(
            int(st.get("retained_mutable_entries", 0)) for _, st in enabled_layers
        )
        base["reduction_count_all_layers"] = total_reductions
        base["average_reduction_time_sec_all_layers"] = (
            total_time / total_reductions if total_reductions > 0 else 0.0
        )
        return base

    def retrieve_for_qa(self, method: str = "rekv", **kwargs):
        """
        Extensible entry point for QA-time retrieval.

        method="rekv":
            ReKV-like internal retrieval on blocks/frames:
            score = cosine(mean_key_block, mean_key_question_proxy)

        In the future you can add other methods in Caches/retrievers.py.
        """
        from . import retrievers

        method = str(method).lower()

        if method == "rekv":
            return retrievers.rekv(self, **kwargs)

        raise ValueError(f"Unsupported retriever method: {method}")


    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache at specific layer index.
        
        Args:
            key_states: New key states
            value_states: New value states
            layer_idx: Index of the layer to update
            cache_kwargs: Optional cache arguments
            
        Returns:
            Updated (keys, values) for this layer
        """
        if cache_kwargs is None:
            cache_kwargs = {}
        
        # Get the layer
        layer = self.layers[layer_idx]
        
        # Update the layer
        keys, values = layer.update(key_states, value_states, cache_kwargs)
        
        # If this is layer 0 and similarity is enabled, store indices for followers
        if layer_idx == 0 and layer.use_similarity and layer.is_first_layer:
            if layer.last_similarity_indices is not None:
                cache_kwargs['similarity_indices'] = layer.last_similarity_indices
        
        return keys, values
    
    def __iter__(self):
        """Iterate over layer caches."""
        for layer in self.layers:
            yield layer.keys, layer.values, getattr(layer, "_sliding_window_tensor", None)
    
    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get sequence length at a specific layer."""
        if not self.layers:
            return 0
        return self.layers[layer_idx].get_seq_length()
    
    def get_max_length(self) -> Optional[int]:
        """Get maximum cache length."""
        if not self.layers:
            return None
        
        # Check if using sliding window
        if self.cache_config.get('use_sliding_window', True):
            return self.layers[0].get_max_cache_shape()
        else:
            return None  # Infinite
    
    def set_sink_size(self, new_sink_size: int):
        """Update sink size for all layers."""
        new_sink_size = int(new_sink_size)
        for layer in self.layers:
            layer.SINK_SIZE = new_sink_size
            
    def capture_sink_guidance(self):
        for layer in self.layers:
            if hasattr(layer, "capture_sink_guidance"):
                layer.capture_sink_guidance()

    def set_hat_size(self, new_hat_size: int):
        """Update sink size for all layers."""
        new_hat_size = int(new_hat_size)
        for layer in self.layers:
            layer.HAT_SIZE = new_hat_size

    def add_frame_tokens(self, frame_token_counts):
        for layer in self.layers:
            if hasattr(layer, "add_frame_tokens"):
                layer.add_frame_tokens(frame_token_counts)

    def set_QA(self, enabled: bool = True):
        enabled = bool(enabled)
        for layer in self.layers:
            if hasattr(layer, "set_QA"):
                layer.set_QA(enabled)
            else:
                layer.qa_mode = enabled
                if enabled and hasattr(layer, "last_similarity_indices"):
                    layer.last_similarity_indices = None

    def get_frame_boundaries(self):
        if not self.layers:
            return []
        layer0 = self.layers[0]
        if hasattr(layer0, "get_frame_boundaries"):
            return layer0.get_frame_boundaries()
        return []
    def add_frame_tokens(self, frame_token_counts):
        for layer in self.layers:
            if hasattr(layer, "add_frame_tokens"):
                layer.add_frame_tokens(frame_token_counts)

    def prepare_frame_batch(self, frame_token_counts, visual_pos_mask, appended_seq_len):
        for layer in self.layers:
            if hasattr(layer, "prepare_frame_append"):
                layer.prepare_frame_append(
                    frame_token_counts=frame_token_counts,
                    visual_pos_mask=visual_pos_mask,
                    appended_seq_len=appended_seq_len,
                )

    def finalize_frame_batch(self):
        if not self.layers:
            return

        for layer in self.layers:
            if hasattr(layer, "finalize_frame_append"):
                layer.finalize_frame_append()

        layer0 = self.layers[0]

        # Ada-KV allocations are layer-specific because each layer has its own
        # Q/K attention statistics.  Do not replay layer-0 indices.
        if getattr(layer0, "adakv_enabled", False):
            for layer in self.layers:
                if not getattr(layer, "adakv_enabled", False):
                    raise RuntimeError("Mixed Ada-KV/non-Ada-KV layers are unsupported")
                if layer.needs_adakv_reduction():
                    layer._apply_adakv_reduction()
            return

        current_len = layer0.get_seq_length()
        total_capacity = layer0.mutable_token_size + layer0.SINK_SIZE + layer0.get_hat_size()

        pending_trace = None

        if (
            not layer0.qa_mode
            and layer0.use_sliding_window
            and current_len > total_capacity
        ):
            if layer0.use_similarity:
                layer0._apply_similarity_reduction()
                pending_trace = layer0.last_similarity_indices
            else:
                layer0._apply_simple_truncation()

        for layer in self.layers[1:]:
            current_len = layer.get_seq_length()
            total_capacity = layer.mutable_token_size + layer.SINK_SIZE + layer.get_hat_size()

            if (
                not layer.qa_mode
                and layer.use_sliding_window
                and current_len > total_capacity
            ):
                if layer.use_similarity:
                    if pending_trace is not None:
                        layer._apply_similarity_reduction(
                            {"similarity_indices": pending_trace}
                        )
                    else:
                        layer._apply_similarity_reduction()
                else:
                    layer._apply_simple_truncation()
    
    def cleanup_offload(self):
        """
        Delete per-cache disk offload artifacts for all layers.

        This is used at end-of-video on Cineca to avoid saturating
        the filesystem with stale KV chunks.
        """
        for layer in self.layers:
            store = getattr(layer, "offload_store", None)
            if store is None:
                continue
            if hasattr(store, "cleanup"):
                store.cleanup()
