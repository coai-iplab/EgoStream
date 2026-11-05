import os
from pathlib import Path

import torch
import huggingface_hub
from transformers import AutoModelForImageTextToText, AutoProcessor


# Hugging Face cache: honour HF_HOME, otherwise use the standard user cache.
DEFAULT_HF_CACHE = str(Path.home() / ".cache" / "huggingface")
cache_dir = os.environ.get("HF_HOME", DEFAULT_HF_CACHE)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


if not hasattr(huggingface_hub, "is_offline_mode"):
    from huggingface_hub.constants import HF_HUB_OFFLINE

    def is_offline_mode():
        return HF_HUB_OFFLINE

    huggingface_hub.is_offline_mode = is_offline_mode


def _resolve_local_hf_snapshot(model_name: str) -> str | None:
    if Path(model_name).exists():
        return model_name

    try:
        org, repo = model_name.split("/", 1)
    except ValueError:
        return None

    model_rel = f"models--{org}--{repo}"
    cache_root = Path(cache_dir)
    model_root = cache_root / "hub" / model_rel
    if not model_root.exists():
        model_root = cache_root / model_rel
    if not model_root.exists():
        return None

    ref_file = model_root / "refs" / "main"
    snapshot_id = ref_file.read_text(encoding="utf-8").strip() if ref_file.exists() else None
    if snapshot_id:
        snapshot_path = model_root / "snapshots" / snapshot_id
        if snapshot_path.exists():
            return str(snapshot_path)

    snapshots_dir = model_root / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted([p for p in snapshots_dir.iterdir() if p.is_dir()])
        if snapshots:
            return str(snapshots[-1])

    return None


def _build_quantization_config(load_in_4bit=False):
    if not load_in_4bit:
        return None

    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "Hai richiesto il caricamento 4-bit, ma BitsAndBytesConfig non e disponibile. "
            "Aggiorna transformers e installa bitsandbytes."
        ) from exc

    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Hai richiesto il caricamento 4-bit, ma bitsandbytes non e installato. "
            "Installa con: pip install bitsandbytes"
        ) from exc

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def load_molmo2_model(model_name="allenai/Molmo2-8B", device="cuda", load_in_4bit=False):
    model_source = _resolve_local_hf_snapshot(model_name) or model_name
    local_only = Path(model_source).exists() or _env_flag("HF_HUB_OFFLINE")
    quantization_config = _build_quantization_config(load_in_4bit)

    print(f">>> MOLMO2 DEBUG: model_name   = {model_name}", flush=True)
    print(f">>> MOLMO2 DEBUG: model_source = {model_source}", flush=True)
    print(f">>> MOLMO2 DEBUG: cache_dir    = {cache_dir}", flush=True)
    print(f">>> MOLMO2 DEBUG: HF_HOME      = {os.environ.get('HF_HOME')}", flush=True)
    print(f">>> MOLMO2 DEBUG: local_only   = {local_only}", flush=True)
    print(f">>> MOLMO2 DEBUG: load_in_4bit = {load_in_4bit}", flush=True)

    processor = AutoProcessor.from_pretrained(
        model_source,
        trust_remote_code=True,
        local_files_only=local_only,
        cache_dir=cache_dir,
    )

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_source,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            local_files_only=local_only,
            cache_dir=cache_dir,
        )
        print(">>> MOLMO2 DEBUG: modello caricato con Flash Attention 2.", flush=True)
    except Exception as e:
        print(f">>> MOLMO2 WARNING: Flash Attention 2 non disponibile (Errore: {e}). Fallback.", flush=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_source,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map=device,
            trust_remote_code=True,
            local_files_only=local_only,
            cache_dir=cache_dir,
        )

    model.eval()
    return model, processor
