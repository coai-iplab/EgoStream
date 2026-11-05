import os
import gc
import sys
import argparse
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")


def _find_project_root(start: Path) -> Path:
    """Find the repo root by looking for a sibling/ancestor directory named 'src'."""
    for candidate in [start] + list(start.parents):
        if (candidate / "src").is_dir():
            return candidate
    raise RuntimeError(
        "Impossibile trovare la root del progetto. Metti questo script dentro il repo "
        "(o in una sua sottocartella) in modo che esista una directory 'src'."
    )


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _find_project_root(THIS_FILE.parent)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.utils import load_config, set_all_seeds
from src.model_loader import load_model
from src.adakv_attention import configure_adakv_attention
from src.video_sampler import SlidingWindowSampler


GLOBAL_SEED = 42
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "single_video" / "qwen3vl_8b_prune_intra_cs.json"


def _is_internvl_model(model_id: str) -> bool:
    return "internvl" in str(model_id).lower()


def _select_model_ops(model_id: str):
    if _is_internvl_model(model_id):
        from src.model_operations_internvl import (
            _prime_cache_sw,
            _append_sampler_to_existing_cache_sw,
            sampling_generation_function_internvl_sw,
        )

        return (
            _prime_cache_sw,
            _append_sampler_to_existing_cache_sw,
            sampling_generation_function_internvl_sw,
        )

    from src.model_operations import (
        _prime_cache_sw,
        _append_sampler_to_existing_cache_sw,
    )

    try:
        from src.generation_sw import sampling_generation_function_sw
    except ImportError:
        # Fallback utile in ambienti locali dove generation_sw non e presente o e stato rinominato.
        from src.model_operations import sampling_generation_function as sampling_generation_function_sw

    return _prime_cache_sw, _append_sampler_to_existing_cache_sw, sampling_generation_function_sw


def _resolve_config_path(config_path: Path, path_value: str) -> str:
    path = Path(os.path.expandvars(str(path_value))).expanduser()
    if path.is_absolute():
        return str(path)
    return str((config_path.resolve().parent / path).resolve())


def _build_question_from_config(config: dict) -> str:
    tester_cfg = config.get("single_video_test", {})
    return tester_cfg.get("question", "Describe this video in detail.")


def _get_window_size(vid_params: dict) -> int:
    # Mantiene compatibilità con la chiave usata nel repo: 'dimention_sw'
    value = vid_params.get("dimention_sw", vid_params.get("dimension_sw", 16))
    return value if int(value) > 0 else 16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path al file di config JSON. Default: conf_sing_vid.json nella stessa cartella dello script.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Override rapido di model.model_id dal config.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Smoke test rapido: InternVL 4-bit, pochi frame e pochi token generati.",
    )
    parser.add_argument(
        "--until-second",
        type=float,
        default=None,
        help="Ferma il sampler dopo N secondi di video.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Override rapido della sliding window.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Override rapido dello stride.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Override rapido di cache.MAX_FRAMES_TO_PROCESS.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override dei token generati.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Riduce le stampe verbose/debug.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config non trovato: {config_path}")

    set_all_seeds(GLOBAL_SEED)

    config = load_config(str(config_path))
    verbose = False if args.quiet or args.fast else config.get("verbose", True)

    paths = config["paths"]
    vid_params = config["video_processing"]
    gen_params = config["generation"]
    model_cfg = config["model"]
    cache_cfg = config["cache"]
    tester_cfg = config.get("single_video_test", {})

    if args.quiet or args.fast:
        cache_cfg = dict(cache_cfg)
        cache_cfg["debug_cache_layout"] = False
        cache_cfg["debug_tracking"] = False

    if cache_cfg.get("type") == "PurgeableCache":
        raise NotImplementedError("Questo tester supporta solo cache sliding-window non Purgeable.")
    if not vid_params.get("use_sliding_window", False):
        raise RuntimeError("Questo tester supporta solo la modalità sliding-window.")

    video_path = _resolve_config_path(config_path, paths["video_path"])
    prompt_path = _resolve_config_path(config_path, paths["prompt_path"])

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video non trovato: {video_path}")
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt non trovato: {prompt_path}")

    model_id = args.model_id or model_cfg.get("model_id")
    if not model_id:
        raise KeyError("Nel config manca model.model_id")

    (
        prime_cache_sw,
        append_sampler_to_existing_cache_sw,
        sampling_generation_fn,
    ) = _select_model_ops(model_id)

    model, processor = load_model(model_id, load_in_4bit=args.fast)
    configure_adakv_attention(model=model, cache_cfg=cache_cfg)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    fps_val = vid_params.get("fps_to_process", 1)
    res_val = tuple(vid_params.get("target_resolution", [896, 512]))
    window_size = args.window_size if args.window_size is not None else _get_window_size(vid_params)
    stride = args.stride if args.stride is not None else int(vid_params.get("stride", 16) or 16)
    until_second = args.until_second if args.until_second is not None else tester_cfg.get("until_second", None)
    max_frames = args.max_frames if args.max_frames is not None else cache_cfg.get("MAX_FRAMES_TO_PROCESS", 100000)

    if args.fast:
        window_size = args.window_size if args.window_size is not None else min(int(window_size), 2)
        stride = args.stride if args.stride is not None else min(int(stride), 2)
        until_second = args.until_second if args.until_second is not None else 2.0
        max_frames = args.max_frames if args.max_frames is not None else 2

    if stride <= 0:
        stride = 16

    if verbose:
        print(f"Project root: {PROJECT_ROOT}")
        print(f"Config: {config_path}")
        print(f"Processing video: {video_path}")
        print(f"Prompt path: {prompt_path}")
        print("Sliding window: True")
        print(
            f"window_size={window_size} | stride={stride} | fps={fps_val} | until_second={until_second}"
        )
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    cache_instance = None
    text_inputs = None
    input_ids = None
    sampler = None

    try:
        cache_instance = prime_cache_sw(
            model=model,
            processor=processor,
            cache_cfg=cache_cfg,
            prompt_path=prompt_path,
            verbose=verbose,
        )

        sampler = SlidingWindowSampler(
            video_path=video_path,
            window_size=window_size,
            stride=stride,
            target_fps=fps_val,
            max_frames=max_frames,
            until_second=until_second,
        )

        append_kwargs = {
            "sampler": sampler,
            "model": model,
            "processor": processor,
            "cache_instance": cache_instance,
            "size": res_val,
            "verbose": verbose,
        }
        if _is_internvl_model(model_id):
            append_kwargs["position_mode"] = "global"

        cache_instance, frame_ingestion_times, cache_size_history = append_sampler_to_existing_cache_sw(
            **append_kwargs,
        )

        avg_frame_time = (
            sum(frame_ingestion_times) / len(frame_ingestion_times)
            if frame_ingestion_times
            else 0.0
        )

        if verbose:
            print(f"Cache pronta. Avg frame time: {avg_frame_time:.3f}s")
            print(f"Cache size history length: {len(cache_size_history)}")
            if cache_size_history:
                print(f"Last cache size: {cache_size_history[-1]}")

            print("LIVE SEQ LEN:", cache_instance.get_seq_length())
            if hasattr(cache_instance, "get_offload_stats"):
                print("OFFLOAD STATS:", cache_instance.get_offload_stats())

        stc_pruner = getattr(cache_instance, "stc_pruner", None)
        if stc_pruner is not None and getattr(stc_pruner, "enabled", False):
            print("STC PRUNER METRICS:", stc_pruner.metrics())

        question = _build_question_from_config(config)
        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": question}],
            }
        ]

        text_inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        input_ids = text_inputs.input_ids.to(model.device)

        retrieve_cfg = cache_cfg.get("retrieve", {})

        if retrieve_cfg.get("enabled", False) and hasattr(cache_instance, "retrieve_for_qa"):
            cache_instance.retrieve_for_qa(
                method=retrieve_cfg.get("method", "rekv"),
                model=model,
                processor=processor,
                question=question,
                top_k_tokens=retrieve_cfg.get(
                    "top_k_tokens",
                    cache_cfg.get("mutable_token_size", 8192),
                ),
                block_size_frames=retrieve_cfg.get("block_size_frames", 1),
                include_live_mutable=retrieve_cfg.get("include_live_mutable", True),
                max_offloaded_chunks=retrieve_cfg.get("max_offloaded_chunks", None),
                dtype=torch.bfloat16,
                device=model.device,
                verbose=verbose,
            )

        cache_instance.set_QA(True)

        generation_kwargs = {
            "model": model,
            "processor": processor,
            "cache": cache_instance,
            "inputs": {"input_ids": input_ids},
            "max_new_tokens": args.max_new_tokens if args.max_new_tokens is not None else (32 if args.fast else 256),
            "temperature": 0.6,
            "top_p": gen_params.get("top_p", 0.9),
            "top_k": 0,
            "repetition_penalty": 1.0,
            "verbose": verbose,
        }
        if _is_internvl_model(model_id):
            generation_kwargs["position_mode"] = "global"
            generation_kwargs["do_sample"] = True

        answer = sampling_generation_fn(**generation_kwargs)

        print("\n=== QUESTION ===")
        print(question)
        print("\n=== MODEL ANSWER ===")
        print(answer)
        print("SEQ LEN: ",cache_instance.get_seq_length())
        if hasattr(cache_instance, "get_offload_stats"):
            print("\n=== OFFLOAD STATS ===")
            print(cache_instance.get_offload_stats())
    finally:
        if sampler is not None:
            try:
                sampler.close()
            except Exception:
                pass
            del sampler

        if text_inputs is not None:
            del text_inputs
        if input_ids is not None:
            del input_ids
        if cache_instance is not None and hasattr(cache_instance, "cleanup_offload"):
            cache_instance.cleanup_offload()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        gc.collect()


if __name__ == "__main__":
    main()
