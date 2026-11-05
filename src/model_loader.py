import os
import torch
import huggingface_hub
from pathlib import Path

from transformers import AutoModel, AutoProcessor, Qwen3VLForConditionalGeneration

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = AutoModel

# Hugging Face cache: honour HF_HOME, otherwise use the standard user cache.
DEFAULT_HF_CACHE = str(Path.home() / ".cache" / "huggingface")
cache_dir = os.environ.get("HF_HOME", DEFAULT_HF_CACHE)


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


if not hasattr(huggingface_hub, "is_offline_mode"):
    from huggingface_hub.constants import HF_HUB_OFFLINE

    def is_offline_mode():
        return HF_HUB_OFFLINE

    huggingface_hub.is_offline_mode = is_offline_mode


def _missing_indexed_weight_files(snapshot_path: Path) -> list[str]:
    index_files = [
        snapshot_path / "model.safetensors.index.json",
        snapshot_path / "pytorch_model.bin.index.json",
    ]
    for index_file in index_files:
        if not index_file.exists():
            continue
        try:
            import json

            with index_file.open("r", encoding="utf-8") as f:
                index = json.load(f)
        except Exception:
            return []

        weight_map = index.get("weight_map", {}) or {}
        expected_files = sorted(set(str(x) for x in weight_map.values()))
        return [name for name in expected_files if not (snapshot_path / name).exists()]

    has_weight_file = any(snapshot_path.glob("*.safetensors")) or any(snapshot_path.glob("*.bin"))
    return [] if has_weight_file else ["<no local weight shards found>"]


def _complete_or_downloadable_snapshot(snapshot_path: Path) -> str | None:
    missing = _missing_indexed_weight_files(snapshot_path)
    if not missing:
        return str(snapshot_path)

    message = (
        f"HF snapshot is incomplete: {snapshot_path} is missing "
        f"{len(missing)} weight file(s), for example {missing[:3]}"
    )
    if _env_flag("HF_HUB_OFFLINE"):
        raise FileNotFoundError(message)
    print(f">>> WARNING: {message}. Falling back to Hugging Face download.", flush=True)
    return None


def _resolve_local_hf_snapshot(model_name: str) -> str | None:
    """Return the local snapshot path for a cached HF model, if present."""
    if Path(model_name).exists():
        return model_name

    try:
        org, repo = model_name.split("/", 1)
    except ValueError:
        return None

    cache_root = Path(cache_dir)
    model_rel = f"models--{org}--{repo}"
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
            return _complete_or_downloadable_snapshot(snapshot_path)

    snapshots_dir = model_root / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted([p for p in snapshots_dir.iterdir() if p.is_dir()])
        if snapshots:
            return _complete_or_downloadable_snapshot(snapshots[-1])

    return None


def _is_internvl_model(model_name: str) -> bool:
    return "internvl" in str(model_name).lower()


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


def _load_internvl_model(model_source, device, local_only, load_in_4bit=False):
    quantization_config = _build_quantization_config(load_in_4bit)
    processor = AutoProcessor.from_pretrained(
        model_source,
        trust_remote_code=False,
        local_files_only=local_only,
        cache_dir=cache_dir,
    )

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_source,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=False,
            attn_implementation="flash_attention_2",
            local_files_only=local_only,
            cache_dir=cache_dir,
        )
        print(">>> DEBUG: SUCCESSO! InternVL caricato con Flash Attention 2.", flush=True)
    except Exception as e:
        print(f">>> WARNING: Flash Attention 2 non disponibile per InternVL (Errore: {e}). Fallback.", flush=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_source,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map=device,
            trust_remote_code=False,
            local_files_only=local_only,
            cache_dir=cache_dir,
        )

    model.eval()
    return model, processor


def load_model(model_name="Qwen/Qwen3-VL-4B-Instruct", device="cuda", load_in_4bit=False):
    model_source = _resolve_local_hf_snapshot(model_name) or model_name
    local_only = Path(model_source).exists() or _env_flag("HF_HUB_OFFLINE")

    print(f">>> DEBUG: model_name   = {model_name}", flush=True)
    print(f">>> DEBUG: model_source = {model_source}", flush=True)
    print(f">>> DEBUG: cache_dir    = {cache_dir}", flush=True)
    print(f">>> DEBUG: HF_HOME      = {os.environ.get('HF_HOME')}", flush=True)
    print(f">>> DEBUG: load_in_4bit = {load_in_4bit}", flush=True)

    if _is_internvl_model(model_name):
        return _load_internvl_model(model_source, device, local_only, load_in_4bit=load_in_4bit)

    processor = AutoProcessor.from_pretrained(
        model_source,
        trust_remote_code=False,
        local_files_only=local_only,
        cache_dir=cache_dir,
    )

    try:
        if _env_flag("QWEN_DISABLE_FLASH_ATTN"):
            raise RuntimeError("Flash Attention 2 disabled by QWEN_DISABLE_FLASH_ATTN")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_source,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=False,
            attn_implementation="flash_attention_2",
            local_files_only=local_only,
            cache_dir=cache_dir,
        )
        print(">>> DEBUG: SUCCESSO! Modello caricato con Flash Attention 2.", flush=True)
    except Exception as e:
        print(f">>> WARNING: Flash Attention 2 non disponibile (Errore: {e}). Fallback.", flush=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_source,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=False,
            local_files_only=local_only,
            cache_dir=cache_dir,
        )

    model.eval()
    return model, processor
