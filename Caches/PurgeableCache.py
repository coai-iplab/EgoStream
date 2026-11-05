import torch
from typing import Any, Optional, Iterable
from transformers.cache_utils import CacheLayerMixin, Cache

class PurgeableCacheLayer(CacheLayerMixin): 
    def __init__(self):
        super().__init__()
        self.past_sequence_length = 0 
        
    def lazy_initialization(self, key_states: torch.Tensor):
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, cache_kwargs: Optional[dict[str, Any]] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        # Standard append
        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)
        
        # Increase counter normally
        self.past_sequence_length += key_states.shape[-2]

        return self.keys, self.values

    def drop_range(self, start_index: int, end_index: int):
        """
        Removes K/V pairs in the range [start_index, end_index).
        Adjusts logical position counter only if necessary? 
        Actually, we usually leave past_sequence_length (logical) monotonically increasing 
        to handle RoPE properly for future tokens. We just reduce the PHYSICAL storage.
        """
        if not self.is_initialized:
            return

        # Ensure valid range
        current_len = self.keys.shape[-2]
        if start_index >= current_len:
            return
        
        end_index = min(end_index, current_len)
        if start_index >= end_index:
            return

        # Drop keys and values
        # Shape: [batch, num_heads, seq_len, head_dim]
        # We slice on dim=-2
        
        prefix_keys = self.keys[:, :, :start_index, :]
        suffix_keys = self.keys[:, :, end_index:, :]
        self.keys = torch.cat([prefix_keys, suffix_keys], dim=-2)

        prefix_values = self.values[:, :, :start_index, :]
        suffix_values = self.values[:, :, end_index:, :]
        self.values = torch.cat([prefix_values, suffix_values], dim=-2)

        # IMPORTANT: We do NOT decrement self.past_sequence_length because that variable 
        # is often used by DynamicCache to return the 'seen_tokens' count for RoPE.
        # However, for our manual loop, we might rely on global_token_counter passed from outside.
        # But to be safe and consistent with DynamicCache semantics where get_seq_length() 
        # usually implies correct position, we should be careful.
        # The base CacheLayerMixin implementation of get_seq_length usually returns keys.shape[-2].
        # In this custom implementation:
        # get_seq_length() -> self.keys.shape[-2] (Physical length).
        # We need RoPE to continue from 'logical' position.
        pass
    def get_seq_length(self) -> int:
        if not self.is_initialized or self.keys.numel() == 0:
            return 0
        return int(self.keys.shape[-2])
    
    def get_max_cache_shape(self) -> int:
        # Infinite effective context (limited by VRAM physically)
        return 10000000 
    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        # Used for creating attention masks
        kv_offset = 0 # We generally want to attend to everything in this specific cache type
        query_length = int(cache_position.shape[0])
        kv_length = int(self.get_seq_length()) + query_length
        return kv_length, kv_offset

    
    
    


class PurgeableCache(Cache):
    def __init__(self,
        config: Optional[Any] = None,
        offloading: bool = False,
        offload_only_non_sliding: bool = False):
        
        layers = []
        if config is not None:
            # Replicate the logic to create layers for all attention blocks
            decoder_config = config.get_text_config(decoder=True)
            
            # Use full attention for all layers (no sliding window masking enforced here, we manage it manually)
            layer_types = [
                "full_attention" for _ in range(decoder_config.num_hidden_layers)
            ]
            
            if hasattr(decoder_config, "num_kv_shared_layers"):
                layer_types = layer_types[: -decoder_config.num_kv_shared_layers]

            for _ in layer_types:
                layers.append(PurgeableCacheLayer())
        
        super().__init__(layers=layers, offloading=offloading, offload_only_non_sliding=offload_only_non_sliding)
    
    def __iter__(self):
        for layer in self.layers:
            yield layer.keys, layer.values, getattr(layer, "_sliding_window_tensor", None)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if not self.layers:
            return 0
        return self.layers[layer_idx].get_seq_length()

    def get_max_length(self) -> Optional[int]:
        return None
    
    def drop_range(self, start_index: int, end_index: int):
        """
        Calls drop_range on all layers.
        """
        for layer in self.layers:
            layer.drop_range(start_index, end_index)
