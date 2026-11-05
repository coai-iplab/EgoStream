from __future__ import annotations

import os
import uuid
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


class DiskOffloadStore:
    """
    Minimal store for KV offloaded to disk.

    Supports two modes:
    - raw: saves keys/values in the original dtype
    - quantized int8 per-tensor: saves q_keys/q_values + scale

    Note:
    - this quantization ONLY concerns the offloaded storage
    - it does not change the forward
    - load_chunk/load_chunks are used for QA-time retrieval
    """

    def __init__(
        self,
        root_dir: str,
        layer_idx: int,
        quantize: bool = False,
        quant_bits: int = 8,
        quant_scheme: str = "per_tensor",
    ):
        root_dir = os.path.expandvars(str(root_dir))

        self.root_dir = (
            Path(root_dir)
            / f"pid_{os.getpid()}"
            / f"cache_{uuid.uuid4().hex}"
            / f"layer_{layer_idx}"
        )
        self.root_dir.mkdir(parents=True, exist_ok=True)

        self.layer_idx = int(layer_idx)
        self.next_chunk_id = 0
        self.chunks: List[Dict[str, Any]] = []

        self.quantize = bool(quantize)
        self.quant_bits = int(quant_bits)
        self.quant_scheme = str(quant_scheme)

        if self.quantize:
            self._validate_quant_config(self.quant_bits, self.quant_scheme)

        self.offloaded_tokens = 0
        self.offloaded_physical_bytes = 0
        self.offloaded_equivalent_bytes = 0

    def __deepcopy__(self, memo):
        """
        Caches cloned for QA/tail must not overwrite the master's chunks.

        They share the already existing chunks, but write any new chunks into
        a new dir. The statistics are copied.
        """
        base_root = self.root_dir.parent.parent.parent

        new = DiskOffloadStore(
            root_dir=str(base_root),
            layer_idx=self.layer_idx,
            quantize=self.quantize,
            quant_bits=self.quant_bits,
            quant_scheme=self.quant_scheme,
        )

        new.chunks = copy.deepcopy(self.chunks, memo)
        new.next_chunk_id = 0

        new.offloaded_tokens = int(self.offloaded_tokens)
        new.offloaded_physical_bytes = int(self.offloaded_physical_bytes)
        new.offloaded_equivalent_bytes = int(self.offloaded_equivalent_bytes)

        return new

    @staticmethod
    def _tensor_bytes(x: torch.Tensor) -> int:
        return int(x.numel() * x.element_size())

    @staticmethod
    def _validate_quant_config(bits: int, scheme: str) -> None:
        bits = int(bits)
        scheme = str(scheme)

        if bits != 8:
            raise ValueError(
                f"DiskOffloadStore currently supports only int8 quantization, "
                f"got quant_bits={bits}"
            )
        if scheme != "per_tensor":
            raise ValueError(
                f"DiskOffloadStore currently supports only quant_scheme='per_tensor', "
                f"got quant_scheme={scheme}"
            )

    @staticmethod
    def _quantize_int8_per_tensor(x: torch.Tensor):
        """
        Symmetric int8 per-tensor quantization.
        """
        x_cpu = x.detach().cpu()
        x_float = x_cpu.float()

        max_abs = x_float.abs().max()

        if not torch.isfinite(max_abs) or float(max_abs.item()) == 0.0:
            scale = torch.tensor(1.0, dtype=torch.float32)
            q = torch.zeros_like(x_float, dtype=torch.int8)
            return q, scale

        scale = (max_abs / 127.0).to(torch.float32)
        q = torch.round(x_float / scale).clamp(-128, 127).to(torch.int8)

        return q, scale

    @staticmethod
    def _dequantize_int8_per_tensor(
        q: torch.Tensor,
        scale: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        out = q.float() * scale.float()

        if dtype is not None:
            out = out.to(dtype)

        if device is not None:
            out = out.to(device)

        return out

    def _build_raw_payload(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        frame_lengths,
        meta,
        token_count: int,
    ) -> Dict[str, Any]:
        return {
            "format": "raw",
            "quantized": False,
            "quant_bits": None,
            "quant_scheme": None,

            "keys": keys.detach().cpu(),
            "values": values.detach().cpu(),

            "frame_lengths": frame_lengths,
            "meta": dict(meta or {}),
            "token_count": int(token_count),

            "dtype": str(keys.dtype),
            "shape_k": tuple(keys.shape),
            "shape_v": tuple(values.shape),
        }

    def _build_quantized_payload(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        frame_lengths,
        meta,
        token_count: int,
    ) -> Dict[str, Any]:
        q_keys, scale_k = self._quantize_int8_per_tensor(keys)
        q_values, scale_v = self._quantize_int8_per_tensor(values)

        return {
            "format": "int8_per_tensor",
            "quantized": True,
            "quant_bits": 8,
            "quant_scheme": "per_tensor",

            "q_keys": q_keys,
            "q_values": q_values,
            "scale_k": scale_k,
            "scale_v": scale_v,

            "frame_lengths": frame_lengths,
            "meta": dict(meta or {}),
            "token_count": int(token_count),

            "original_dtype": str(keys.dtype),
            "dtype": str(keys.dtype),
            "shape_k": tuple(keys.shape),
            "shape_v": tuple(values.shape),
        }

    def append(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        frame_lengths,
        meta=None,
        quantize=None,
        quant_bits=None,
        quant_scheme=None,
    ):
        """
        Saves an offloaded chunk to disk.

        The behavior can be controlled globally by the store or
        locally by the pipeline step through quantize/quant_bits/quant_scheme.
        """
        if keys is None or values is None or keys.numel() == 0:
            return None

        frame_lengths = [int(x) for x in frame_lengths if int(x) > 0]
        token_count = int(keys.shape[-2])
        equivalent_bytes = self._tensor_bytes(keys) + self._tensor_bytes(values)

        use_quantize = self.quantize if quantize is None else bool(quantize)
        use_quant_bits = self.quant_bits if quant_bits is None else int(quant_bits)
        use_quant_scheme = self.quant_scheme if quant_scheme is None else str(quant_scheme)

        if use_quantize:
            self._validate_quant_config(use_quant_bits, use_quant_scheme)
            payload = self._build_quantized_payload(
                keys=keys,
                values=values,
                frame_lengths=frame_lengths,
                meta=meta,
                token_count=token_count,
            )
        else:
            payload = self._build_raw_payload(
                keys=keys,
                values=values,
                frame_lengths=frame_lengths,
                meta=meta,
                token_count=token_count,
            )

        tmp_path = self.root_dir / f"chunk_{self.next_chunk_id:08d}.tmp"
        final_path = self.root_dir / f"chunk_{self.next_chunk_id:08d}.pt"

        torch.save(payload, tmp_path)
        os.replace(tmp_path, final_path)

        physical_bytes = int(final_path.stat().st_size)

        record = {
            "path": str(final_path),
            "token_count": int(token_count),
            "physical_bytes": int(physical_bytes),
            "equivalent_bytes": int(equivalent_bytes),
            "frame_lengths": frame_lengths,
            "meta": dict(meta or {}),

            "quantized": bool(use_quantize),
            "quant_bits": int(use_quant_bits) if use_quantize else None,
            "quant_scheme": use_quant_scheme if use_quantize else None,
        }

        self.chunks.append(record)
        self.next_chunk_id += 1

        self.offloaded_tokens += int(token_count)
        self.offloaded_physical_bytes += int(physical_bytes)
        self.offloaded_equivalent_bytes += int(equivalent_bytes)

        return record

    def load_chunk(
        self,
        chunk_idx: int,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ):
        """
        Loads a chunk and dequantizes it if necessary.

        Returns:
            keys, values, payload
        """
        record = self.chunks[int(chunk_idx)]
        payload = torch.load(record["path"], map_location="cpu")

        if bool(payload.get("quantized", False)):
            keys = self._dequantize_int8_per_tensor(
                payload["q_keys"],
                payload["scale_k"],
                dtype=dtype,
                device=device,
            )
            values = self._dequantize_int8_per_tensor(
                payload["q_values"],
                payload["scale_v"],
                dtype=dtype,
                device=device,
            )
        else:
            keys = payload["keys"]
            values = payload["values"]

            if dtype is not None:
                keys = keys.to(dtype)
                values = values.to(dtype)

            if device is not None:
                keys = keys.to(device)
                values = values.to(device)

        return keys, values, payload

    def get_stats(self):
        ratio = 0.0
        if self.offloaded_physical_bytes > 0:
            ratio = float(self.offloaded_equivalent_bytes) / float(self.offloaded_physical_bytes)

        return {
            "offloaded_tokens": int(self.offloaded_tokens),
            "offloaded_chunks": int(len(self.chunks)),

            "offloaded_physical_bytes": int(self.offloaded_physical_bytes),
            "offloaded_equivalent_bytes": int(self.offloaded_equivalent_bytes),
            "offloaded_compression_ratio": ratio,

            "offload_quantized": bool(self.quantize),
            "offload_quant_bits": int(self.quant_bits) if self.quantize else None,
            "offload_quant_scheme": self.quant_scheme if self.quantize else None,

            "offload_dir": str(self.root_dir),
        }
    def cleanup(self):
        import shutil
        root = Path(self.root_dir)
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)