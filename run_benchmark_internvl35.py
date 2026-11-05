import builtins
import os
import gc
import copy
import json
import time
import random
import argparse
import threading
import hashlib
from pathlib import Path
from collections import defaultdict
import sys
import subprocess
import torch

from src.utils import load_config, save_json, parse_prediction, set_all_seeds
from src.monitor import init_nvml, power_monitor_thread, get_cache_memory_size
from src.model_loader import load_model
from src.model_operations_internvl import (
    _prime_cache_sw,
    _append_sampler_to_existing_cache_sw,
    _append_frame_batches_to_existing_cache_sw,
    _embed_tokens,
)
from src.video_sampler import SlidingWindowSampler, IncrementalSlidingWindowSampler

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

global_seed = 42
EGOLIFE_NUM_RANKS = 8
EGOLIFE_TARGET_FPS = 0.5

QUESTION_TIME_ORDER = [
    "instant_recall",
    "short_term_recall",
    "short_mid_term_recall",
    "mid_term_recall",
    "mid_long_term_recall",
    "long_term_recall",
    "ultra_long_term",
]



def build_result_key(annotation_id, question_time_type):
    return f"{annotation_id}::{question_time_type}"


def day_number(day_name):
    return int(str(day_name).upper().replace("DAY", ""))


def is_egolife_item(item):
    dataset = str(item.get("dataset", "")).lower()
    subfolder = str(item.get("subfolder", "")).lower()
    source_file = str(item.get("source_file", "")).lower()
    return (
        "egolife" in dataset
        or "egolife" in subfolder
        or "egolife" in source_file
    )


def build_task_sort_key(task):
    if task.get("is_egolife", False):
        return (
            1,
            day_number(task["egolife_day"]),
            int(str(task["egolife_clip_id"])),
            int(task.get("annotation_id", -1)),
            QUESTION_TIME_ORDER.index(task["absolute_question_time_type"]),
        )

    return (
        0,
        str(task.get("resolved_video_id", "")),
        int(task.get("annotation_id", -1)),
        QUESTION_TIME_ORDER.index(task["absolute_question_time_type"]),
    )


def expand_annotation_tasks(full_dataset):
    expanded = []

    for item in full_dataset:
        absolute_question_times = item.get("absolute_question_times") or {}
        annotation_id = item.get("annotation_id")
        item_is_egolife = is_egolife_item(item)

        for question_time_type in QUESTION_TIME_ORDER:
            until_second = absolute_question_times.get(question_time_type)
            if until_second is None:
                continue

            task = dict(item)
            task["absolute_question_time_type"] = question_time_type
            task["absolute_question_time_sec"] = float(until_second)
            task["result_key"] = build_result_key(annotation_id, question_time_type)
            task["is_egolife"] = item_is_egolife

            if item_is_egolife:
                video_map = item.get("video_id", {}) or {}
                target_info = video_map.get(question_time_type)
                if not isinstance(target_info, dict):
                    continue
                day = target_info.get("day")
                clip_id = str(target_info.get("video_id"))
                if not day or clip_id in {"", "None"}:
                    continue
                task["egolife_day"] = day
                task["egolife_clip_id"] = clip_id
                task["resolved_video_id"] = f"{day}/{clip_id}"
            else:
                task["resolved_video_id"] = str(item.get("video_id") or item.get("clip_uid") or "")

            expanded.append(task)

    expanded.sort(key=build_task_sort_key)
    return expanded


def stable_bucket(key: str, num_buckets: int, seed: int = 42) -> int:
    s = f"{seed}::{key}".encode("utf-8")
    h = hashlib.md5(s).hexdigest()
    return int(h, 16) % num_buckets


def egolife_day_for_rank(rank):
    """
    Hardcoded EgoLife split for speed:

    rank 0 -> DAY1
    rank 1 -> DAY2
    ...
    rank 7 -> DAY8
    """
    return int(rank) + 1


def egolife_target_day_assigned_to_rank(task, rank):
    return day_number(task["egolife_day"]) == egolife_day_for_rank(rank)


def first_clip_index_for_egolife_rank_day(ordered_clips, rank):
    target_day_num = egolife_day_for_rank(rank)

    for idx, clip_info in enumerate(ordered_clips):
        if day_number(clip_info["day"]) >= target_day_num:
            return idx

    return len(ordered_clips)


def split_tasks_by_rank(expanded_dataset, rank, world_size):
    """
    Hardcoded split:

    - rank 0..7: EgoLife, one day per rank:
        rank 0 -> DAY1
        rank 1 -> DAY2
        ...
        rank 7 -> DAY8

    - rank >= 8: normal videos, assigned by video_id.

    This avoids the old slow setup where rank 0 processed all EgoLife days.
    """
    normal_tasks = [x for x in expanded_dataset if not x.get("is_egolife", False)]
    egolife_tasks = [x for x in expanded_dataset if x.get("is_egolife", False)]

    grouped_by_video = defaultdict(list)
    for task in normal_tasks:
        video_key = str(task.get("resolved_video_id") or task.get("video_id") or task.get("clip_uid") or "")
        grouped_by_video[video_key].append(task)

    ordered_video_keys = sorted(grouped_by_video.keys())

    if world_size <= 1:
        my_normal_tasks = []
        for video_key in ordered_video_keys:
            my_normal_tasks.extend(grouped_by_video[video_key])
        my_egolife_tasks = egolife_tasks
        my_dataset = my_normal_tasks + my_egolife_tasks
        my_dataset.sort(key=build_task_sort_key)
        return my_dataset, my_normal_tasks, my_egolife_tasks, len(ordered_video_keys)

    rank = int(rank)

    # ------------------------------------------------------------
    # EgoLife ranks: 0..7, one day per rank.
    # ------------------------------------------------------------
    if rank < EGOLIFE_NUM_RANKS:
        my_egolife_tasks = [
            task for task in egolife_tasks
            if egolife_target_day_assigned_to_rank(task, rank)
        ]
        my_dataset = my_egolife_tasks
        my_dataset.sort(key=build_task_sort_key)
        return my_dataset, [], my_egolife_tasks, len(ordered_video_keys)

    # ------------------------------------------------------------
    # Normal ranks: rank >= 8.
    # ------------------------------------------------------------
    normal_rank_start = EGOLIFE_NUM_RANKS
    normal_rank_count = max(1, world_size - normal_rank_start)

    my_normal_tasks = []
    for video_key in ordered_video_keys:
        assigned_rank = normal_rank_start + stable_bucket(
            video_key,
            normal_rank_count,
            seed=global_seed,
        )
        if assigned_rank == rank:
            my_normal_tasks.extend(grouped_by_video[video_key])

    my_dataset = my_normal_tasks
    my_dataset.sort(key=build_task_sort_key)
    return my_dataset, my_normal_tasks, [], len(ordered_video_keys)

def resolve_video_path(paths, element, print_fn):

    video_id_raw = element.get("resolved_video_id") or element.get("video_id") or element.get("clip_uid")
    subfolder_raw = element.get("subfolder")
    clip_base_raw = paths.get("clip_base_path", "")

    video_id = str(video_id_raw or "").strip()
    subfolder = str(subfolder_raw or "").strip()
    clip_base = str(clip_base_raw or "").strip()

    video_filename = f"{video_id}.mp4"
    expanded_base = os.path.expandvars(clip_base)

    print_fn(f"DEBUG: video_id raw repr: {video_id_raw!r}")
    print_fn(f"DEBUG: video_id stripped repr: {video_id!r}")
    print_fn(f"DEBUG: subfolder raw repr: {subfolder_raw!r}")
    print_fn(f"DEBUG: subfolder stripped repr: {subfolder!r}")
    print_fn(f"DEBUG: clip_base_path raw repr: {clip_base_raw!r}")
    print_fn(f"DEBUG: clip_base_path stripped repr: {clip_base!r}")
    print_fn(f"DEBUG: expanded_base repr: {expanded_base!r}")

    candidate_paths = []

    if clip_base:
        candidate_paths.extend([
            os.path.join(clip_base, video_filename),
            os.path.join(clip_base, video_id, video_filename),
        ]) 


        if subfolder:
            candidate_paths.extend([
                os.path.join(clip_base, subfolder, video_filename),
                os.path.join(clip_base, subfolder, video_id, video_filename),
            ])

    if subfolder:
        candidate_paths.extend([
            os.path.join(subfolder, video_filename),
            os.path.join(subfolder, video_id, video_filename),
        ])

    # deduplica mantenendo ordine
    seen = set()
    unique_candidate_paths = []
    for p in candidate_paths:
        if p not in seen:
            seen.add(p)
            unique_candidate_paths.append(p)

    for candidate in unique_candidate_paths:
        print_fn(f"DEBUG: Checking existence of: {candidate}")
        if os.path.exists(candidate):
            print_fn(f"DEBUG: Found video at: {candidate}")
            return candidate

    print_fn(f"Skip: {video_id} not found in any known path.")
    return None


def resolve_egolife_base_path(paths, egolife_tasks):
    clip_base_path = os.path.expandvars(str(paths.get("clip_base_path", "")).strip())
    if clip_base_path and Path(clip_base_path).exists():
        return clip_base_path

    for task in egolife_tasks:
        subfolder = os.path.expandvars(str(task.get("subfolder", "")).strip())
        if subfolder and Path(subfolder).exists():
            return subfolder

    raise FileNotFoundError(
        "Unable to resolve EgoLife base path. Set paths.clip_base_path or provide a valid subfolder in annotations."
    )


def index_egolife_clips(clip_base_path):
    base = Path(os.path.expandvars(clip_base_path))
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"EgoLife base path not found: {base}")

    def extract_clip_numeric_id(path: Path) -> str:
        stem = path.stem
        parts = stem.split("_")
        return parts[-1]

    ordered_clips = []
    clip_to_index = {}

    day_dirs = sorted(
        [p for p in base.iterdir() if p.is_dir() and p.name.upper().startswith("DAY")],
        key=lambda p: day_number(p.name),
    )

    for day_dir in day_dirs:
        mp4_files = sorted(
            day_dir.glob("*.mp4"),
            key=lambda p: int(extract_clip_numeric_id(p)),
        )

        for mp4_path in mp4_files:
            clip_id = extract_clip_numeric_id(mp4_path)
            idx = len(ordered_clips)
            entry = {
                "day": day_dir.name,
                "clip_id": clip_id,
                "path": str(mp4_path),
            }
            ordered_clips.append(entry)
            clip_to_index[(day_dir.name, clip_id)] = idx

    return ordered_clips, clip_to_index


def group_egolife_tasks(tasks):
    groups = defaultdict(list)
    for t in tasks:
        groups[(t["egolife_day"], t["egolife_clip_id"])].append(t)
    ordered_targets = sorted(groups.keys(), key=lambda k: (day_number(k[0]), int(str(k[1]))))
    return ordered_targets, groups


def group_tasks_by_video(tasks):
    grouped = defaultdict(list)
    for task in tasks:
        video_key = str(task.get("resolved_video_id") or task.get("video_id") or task.get("clip_uid") or "")
        grouped[video_key].append(task)
    for video_key in grouped:
        grouped[video_key].sort(
            key=lambda x: (
                float(x.get("absolute_question_time_sec") or 0.0),
                int(x.get("annotation_id", -1)),
                QUESTION_TIME_ORDER.index(x["absolute_question_time_type"]),
            )
        )
    return grouped


def group_tasks_by_time(tasks):
    grouped = defaultdict(list)
    for task in tasks:
        grouped[float(task["absolute_question_time_sec"])].append(task)
    ordered_times = sorted(grouped.keys())
    return ordered_times, grouped


def clone_cache_for_question(master_cache):
    q_cache = copy.deepcopy(master_cache)
    if hasattr(master_cache, "global_token_counter"):
        q_cache.global_token_counter = master_cache.global_token_counter
    return q_cache


def build_mcqa_prompt_and_options(element):
    question = element["question"]
    correct_answer = element.get("correct_answer", element.get("answer"))
    options = list(element.get("wrong_answers", [])) + [correct_answer]
    random.shuffle(options)
    options_text = "".join([f"{i+1}) {opt}\n" for i, opt in enumerate(options)])
    prompt_text = f"Question: {question}\nOptions:\n{options_text}Reply with number only."
    return question, correct_answer, options, prompt_text


def build_open_ended_prompt(element):
    return (
        "Answer the following question about the video.\n"
        "Use the video context currently stored in cache and be specific about the action.\n"
        "Keep the answer concise, but include the most relevant detail you can infer.\n"
        "Do not output an option number.\n"
        f"Question: {element['question']}\n"
        "Answer:"
    )


def normalize_text(s):
    if s is None:
        return ""
    return " ".join(str(s).strip().lower().split())


def finalize_open_ended_prediction(output_text, correct_answer):
    predicted_text = str(output_text).strip()
    exact_match = normalize_text(predicted_text) == normalize_text(correct_answer)
    return predicted_text, exact_match


def run_generation_from_cache(model, processor, cache_instance, prompt_text, gen_params, verbose):
    conversation = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    text_inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(model.device)

    from src.model_operations_internvl import sampling_generation_function_internvl_sw

    output_text = sampling_generation_function_internvl_sw(
        model=model,
        processor=processor,
        cache=cache_instance,
        inputs={"input_ids": input_ids},
        max_new_tokens=gen_params.get("max_new_tokens", 16),
        temperature=gen_params.get("temperature", 0.01),
        top_p=gen_params.get("top_p", 0.9),
        top_k=gen_params.get("top_k", 0),
        repetition_penalty=gen_params.get("repetition_penalty", 1.2),
        ban_eos_steps=gen_params.get("ban_eos_steps", 2),
        position_mode="global",
        verbose=verbose,
    )

    return output_text, text_inputs, input_ids


def finalize_prediction(output_text, options, correct_answer):
    pred_idx_str = parse_prediction(output_text)
    pred_answer_text = "N/A"
    is_correct = False

    if pred_idx_str.isdigit():
        idx = int(pred_idx_str) - 1
        if 0 <= idx < len(options):
            pred_answer_text = options[idx]
            is_correct = pred_answer_text == correct_answer

    return pred_idx_str, pred_answer_text, is_correct


def get_hardware_metrics(num_gpus, max_vram_on_loading_per_gpu):
    max_vram_per_gpu = {}
    total_max_vram_gb = 0.0
    if torch.cuda.is_available():
        for gpu_id in range(num_gpus):
            vram_gb = torch.cuda.max_memory_allocated(device=gpu_id) / (1024**3)
            max_vram_per_gpu[gpu_id] = vram_gb
            total_max_vram_gb += vram_gb

    return {
        "gpu_max_vram_on_loading_per_gpu_GB": {str(k): round(v, 3) for k, v in max_vram_on_loading_per_gpu.items()},
        "gpu_max_vram_on_loading_total_GB": round(sum(max_vram_on_loading_per_gpu.values()), 3)
        if max_vram_on_loading_per_gpu
        else 0.0,
        "gpu_max_vram_per_gpu_GB": {str(k): round(v, 3) for k, v in max_vram_per_gpu.items()},
        "gpu_max_vram_total_GB": round(total_max_vram_gb, 3),
    }

def _safe_int_bytes(x):
    if x is None:
        return 0

    if isinstance(x, dict):
        total = 0
        for v in x.values():
            if isinstance(v, dict):
                total += _safe_int_bytes(v)
            elif isinstance(v, (list, tuple)):
                total += sum(_safe_int_bytes(i) for i in v)
            else:
                total += _safe_int_bytes(v)
        return int(total)

    if isinstance(x, (list, tuple)):
        return int(sum(_safe_int_bytes(v) for v in x))

    try:
        return int(x)
    except Exception:
        return 0

def _to_int_bytes(x, default=0):
    """
    Normalizza l'output di get_cache_memory_size(...).

    In alcune versioni ritorna direttamente un int.
    In altre può ritornare un dict con più campi.
    """
    if x is None:
        return int(default)

    if isinstance(x, bool):
        return int(x)

    if isinstance(x, (int, float)):
        return int(x)

    if isinstance(x, dict):
        # Prova prima i nomi più probabili.
        preferred_keys = [
            "total_bytes",
            "cache_memory",
            "cache_memory_bytes",
            "memory_bytes",
            "live_bytes",
            "bytes",
            "total",
        ]

        for key in preferred_keys:
            if key in x and isinstance(x[key], (int, float)):
                return int(x[key])

        # Fallback: sum every scalar numeric value of the dict.
        total = 0
        found = False
        for value in x.values():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                total += int(value)
                found = True

        if found:
            return int(total)

        return int(default)

    return int(default)


def get_cache_metrics(cache):
    raw_live_memory = get_cache_memory_size(cache) if cache else 0
    live_bytes = _to_int_bytes(raw_live_memory)

    offload_stats = {}
    if cache is not None and hasattr(cache, "get_offload_stats"):
        offload_stats = cache.get_offload_stats() or {}

    offloaded_physical_bytes = int(offload_stats.get("offloaded_physical_bytes", 0) or 0)
    offloaded_equivalent_bytes = int(offload_stats.get("offloaded_equivalent_bytes", 0) or 0)
    offloaded_tokens = int(offload_stats.get("offloaded_tokens", 0) or 0)

    live_tokens = (
        int(cache.get_seq_length())
        if cache is not None and hasattr(cache, "get_seq_length")
        else 0
    )

    return {
        "cache_memory_live_bytes": int(live_bytes),
        "cache_memory_live_raw": raw_live_memory,
        "cache_memory_offloaded_physical_bytes": offloaded_physical_bytes,
        "cache_memory_offloaded_equivalent_bytes": offloaded_equivalent_bytes,
        "cache_memory_total_physical_bytes": int(live_bytes + offloaded_physical_bytes),
        "live_tokens": live_tokens,
        "offloaded_tokens": offloaded_tokens,
        "logical_tokens": int(live_tokens + offloaded_tokens),
        "offload_stats": offload_stats,
    }


def _cleanup_cache_artifacts(obj):
    """
    Best-effort cleanup for cache-side temporary resources.
    Safe on caches without disk/offload support.
    """
    if obj is None:
        return
    if hasattr(obj, "cleanup_offload"):
        try:
            obj.cleanup_offload()
        except Exception:
            # Keep benchmark cleanup robust even if a cache backend has no cleanup()
            # method on the underlying store.
            pass



def cleanup_video_cache_artifacts(video_label, verbose=False, *caches):
    """
    End-of-video cleanup for disk/offload caches.

    This must be called when a video group is finished, otherwise the per-video
    offload directories can accumulate and saturate the filesystem on Cineca.
    """
    cleaned = 0
    for cache in caches:
        if cache is None or not hasattr(cache, "cleanup_offload"):
            continue
        try:
            before = {}
            if hasattr(cache, "get_offload_stats"):
                before = cache.get_offload_stats() or {}
            cache.cleanup_offload()
            cleaned += 1
            if verbose:
                off_tokens = int(before.get("offloaded_tokens", 0))
                off_bytes = int(before.get("offloaded_physical_bytes", 0))
                print(
                    f"[OFFLOAD CLEANUP] video={video_label} | "
                    f"removed_cache={cleaned} | "
                    f"offloaded_tokens_before={off_tokens} | "
                    f"physical_bytes_before={off_bytes}",
                    flush=True,
                )
        except Exception as exc:
            if verbose:
                print(
                    f"[OFFLOAD CLEANUP WARNING] video={video_label} | error={exc}",
                    flush=True,
                )
def maybe_retrieve_for_qa(
    cache_instance,
    model,
    processor,
    cache_cfg,
    question,
    verbose=False,
):
    """
    Optional retrieval hook used by offload/reKV-style caches.

    Must run BEFORE set_QA(True), because retrieval may rebuild/extend the live
    cache that the QA forward will consume.
    """
    retrieve_cfg = cache_cfg.get("retrieve", {}) or {}

    if not bool(retrieve_cfg.get("enabled", False)):
        return False

    if cache_instance is None or not hasattr(cache_instance, "retrieve_for_qa"):
        if verbose:
            print(
                "[RETRIEVE] retrieve.enabled=True but cache has no retrieve_for_qa(); skipping.",
                flush=True,
            )
        return False

    if verbose:
        print(
            "[RETRIEVE] Running QA retrieval | "
            f"method={retrieve_cfg.get('method', 'rekv')} | "
            f"top_k_tokens={retrieve_cfg.get('top_k_tokens', cache_cfg.get('mutable_token_size', 8192))} | "
            f"block_size_frames={retrieve_cfg.get('block_size_frames', 1)} | "
            f"include_live_mutable={retrieve_cfg.get('include_live_mutable', True)}",
            flush=True,
        )

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

    return True


def cleanup_locals(*objs):
    for obj in objs:
        try:
            _cleanup_cache_artifacts(obj)
        except Exception:
            pass

        try:
            del obj
        except Exception:
            pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def count_fully_processed_videos(task_list, processed_keys):
    grouped = group_tasks_by_video(task_list)
    done = 0
    for tasks_for_video in grouped.values():
        if all(task["result_key"] in processed_keys for task in tasks_for_video):
            done += 1
    return done, len(grouped)


def result_entry_has_both_modes(item):
    return "model_output_raw" in item and "open_ended_output_raw" in item


def upsert_result(predictions, prediction_index, result_key, result_entry):
    if result_key in prediction_index:
        predictions[prediction_index[result_key]] = result_entry
    else:
        prediction_index[result_key] = len(predictions)
        predictions.append(result_entry)


def run_mcqa_constrained_from_cache(model, processor, cache_instance, prompt_text, num_options, verbose=False):
    conversation = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    text_inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(model.device)

    current_global_step = cache_instance.global_token_counter
    current_seq_len = input_ids.shape[1]
    physical_past_len = cache_instance.get_seq_length()

    attention_mask = torch.ones(
        (1, physical_past_len + current_seq_len),
        dtype=torch.long,
        device=model.device,
    )
    position_ids = torch.arange(
        current_global_step,
        current_global_step + current_seq_len,
        dtype=torch.long,
        device=model.device
    ).unsqueeze(0)
    cache_position = torch.arange(
        physical_past_len,
        physical_past_len + current_seq_len,
        dtype=torch.long,
        device=model.device,
    )
    input_embeds = _embed_tokens(model, input_ids)

    with torch.no_grad():
        out = model.forward(
            inputs_embeds=input_embeds,
            past_key_values=cache_instance,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=True,
        )

    logits = out.logits[:, -1, :]

    allowed = {}
    for i in range(1, num_options + 1):
        for s in [str(i), f" {i}"]:
            ids = processor.tokenizer.encode(s, add_special_tokens=False)
            if len(ids) == 1:
                allowed[i] = ids[0]
                break

    scores = {i: logits[0, tok_id].item() for i, tok_id in allowed.items()}
    pred = max(scores, key=scores.get)
    return str(pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="Configs/Qwen3VL_4B_896x512resolution_1fps.json",
    )
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    args = parser.parse_args()

    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        builtins.print(f"Config file {args.config} not found.")
        return

    verbose = config.get("verbose", False)
    import functools

    if verbose:
        print = functools.partial(builtins.print, flush=True)
    else:
        print = lambda *args, **kwargs: None

    print(">>> START")

    paths = config.get("paths", {})
    vid_params = config.get("video_processing", {})
    gen_params = config.get("generation", {})
    model_cfg = config.get("model", {})
    cache_cfg = config.get("cache", {})

    if not vid_params.get("use_sliding_window", False):
        raise RuntimeError("This script supports only sliding-window mode.")

    fps_val = vid_params.get("fps_to_process", 1)
    res_val = vid_params.get("target_resolution", [360, 240])
    config_name = Path(args.config).stem

    final_output_dir = os.path.join(paths.get("output_dir", "Inference"), config_name)
    os.makedirs(final_output_dir, exist_ok=True)
    output_path = os.path.join(final_output_dir, f"part_{args.rank}.json")
    corrupted_videos_path = os.path.join(final_output_dir, f"corrupted_videos_rank_{args.rank}.txt")
    print(f"[Rank {args.rank}] Output path: {output_path}")

    with open(paths["annotations_file"], "r") as f:
        full_dataset = json.load(f)

    expanded_dataset = expand_annotation_tasks(full_dataset)
    my_dataset, my_normal_tasks_split, my_egolife_tasks_split, total_normal_videos = split_tasks_by_rank(
        expanded_dataset=expanded_dataset,
        rank=args.rank,
        world_size=args.world_size,
    )

    watch_video_ids = {
        "d2b7d331-4617-45db-a339-80afb7e25741",
        "ac556354-22a5-4aee-a678-90aef12e0b4c",
    }

    for item in expanded_dataset:
        vid = str(item.get("resolved_video_id") or item.get("video_id") or item.get("clip_uid") or "")
        vid_clean = vid.strip()
        if vid_clean in watch_video_ids:
            if args.world_size <= 1:
                assigned_rank = 0
            elif item.get("is_egolife", False):
                assigned_rank = day_number(item["egolife_day"]) - 1
            else:
                normal_rank_start = EGOLIFE_NUM_RANKS
                normal_rank_count = max(1, args.world_size - normal_rank_start)
                assigned_rank = normal_rank_start + stable_bucket(
                    vid,
                    normal_rank_count,
                    seed=global_seed,
                )
            builtins.print(
                f"[DEBUG WATCH] annotation_id={item.get('annotation_id')} "
                f"video_id_raw={vid!r} "
                f"video_id_clean={vid_clean!r} "
                f"time_type={item.get('absolute_question_time_type')} "
                f"assigned_rank={assigned_rank}",
                flush=True,
            )

    for item in my_dataset:
        vid = str(item.get("resolved_video_id") or item.get("video_id") or item.get("clip_uid") or "")
        if vid.strip() in watch_video_ids:
            builtins.print(
                f"[RANK {args.rank} HAS WATCH] annotation_id={item.get('annotation_id')} "
                f"video_id={vid!r} time_type={item.get('absolute_question_time_type')}",
                flush=True,
            )

    print(f"[Rank {args.rank}] Assigned {len(my_dataset)} inference tasks.")
    print(
        f"[Rank {args.rank}] Non-EgoLife videos assigned exclusively to this rank: "
        f"{len({str(x.get('resolved_video_id') or x.get('video_id') or x.get('clip_uid') or '') for x in my_normal_tasks_split})} / {total_normal_videos}"
    )
    if args.world_size <= 1:
        print(f"[Rank {args.rank}] EgoLife tasks assigned to this rank: {len(my_egolife_tasks_split)}")
    elif args.rank < EGOLIFE_NUM_RANKS:
        print(
            f"[Rank {args.rank}] Worker type: EgoLife | "
            f"day=DAY{egolife_day_for_rank(args.rank)} | "
            f"target_fps={EGOLIFE_TARGET_FPS} | "
            f"assigned_egolife_tasks={len(my_egolife_tasks_split)}"
        )
    else:
        print(f"[Rank {args.rank}] EgoLife tasks assigned to this rank: 0")

    predictions = []
    prediction_index = {}
    processed_keys = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, "r") as f:
                predictions = json.load(f)
            for idx, item in enumerate(predictions):
                ann_id = item.get("annotation_id")
                qtype = item.get("absolute_question_time_type")
                if ann_id is None or qtype is None:
                    continue
                result_key = build_result_key(ann_id, qtype)
                prediction_index[result_key] = idx
                if result_entry_has_both_modes(item):
                    processed_keys.add(result_key)
        except Exception:
            pass

    watch_annotation_ids = {28, 36, 40, 41, 42, 43, 44, 433}
    watch_result_keys = set()

    for item in my_dataset:
        if item.get("annotation_id") in watch_annotation_ids:
            watch_result_keys.add(item["result_key"])

    print(f"[Rank {args.rank}] predictions loaded: {len(predictions)}")
    print(f"[Rank {args.rank}] processed_keys count: {len(processed_keys)}")

    for idx, item in enumerate(predictions):
        ann_id = item.get("annotation_id")
        qtype = item.get("absolute_question_time_type")
        if ann_id is None or qtype is None:
            continue

        rk = build_result_key(ann_id, qtype)
        if rk in watch_result_keys:
            print(
                f"[RESUME WATCH][Rank {args.rank}] idx={idx} "
                f"result_key={rk} "
                f"video_id={item.get('video_id')!r} "
                f"has_both={result_entry_has_both_modes(item)} "
                f"model_output_raw={item.get('model_output_raw')!r} "
                f"open_ended_output_raw={item.get('open_ended_output_raw')!r}",
                flush=True,
            )

    pending_tasks = [item for item in my_dataset if item["result_key"] not in processed_keys]
    if len(pending_tasks) == 0:
        print("Completed. Exiting.")
        return

    pending_watch = [x["result_key"] for x in pending_tasks if x.get("annotation_id") in watch_annotation_ids]
    print(f"[Rank {args.rank}] pending tasks total: {len(pending_tasks)}")
    print(f"[Rank {args.rank}] pending watch keys: {pending_watch}")

    model, processor = load_model(model_cfg["model_id"])

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    max_vram_on_loading_per_gpu = {}
    if torch.cuda.is_available():
        for gpu_id in range(num_gpus):
            max_vram_on_loading_per_gpu[gpu_id] = (
                torch.cuda.max_memory_allocated(device=gpu_id) / (1024**3)
            )

    use_nvml, nvml_handle = init_nvml()

    window_size = vid_params.get("dimention_sw", 16)
    stride = vid_params.get("stride", 16)
    if window_size <= 0:
        window_size = 16
        print(f"WARNING: Invalid window_size={vid_params.get('dimention_sw')} in config, using default: {window_size}")
    if stride <= 0:
        stride = 16
        print(f"WARNING: Invalid stride={vid_params.get('stride')} in config, using default: {stride}")

    normal_tasks = [x for x in pending_tasks if not x.get("is_egolife", False)]
    egolife_tasks = [x for x in pending_tasks if x.get("is_egolife", False)]

    # ---------------------------------------------------------------
    # NORMAL BRANCH
    # ---------------------------------------------------------------
    if normal_tasks:
        done_videos, total_videos = count_fully_processed_videos(my_normal_tasks_split, processed_keys)
        print(f"[Rank {args.rank}] Non-EgoLife videos already fully processed: {done_videos}/{total_videos}")

        grouped_normal = group_tasks_by_video(normal_tasks)

        for video_id, tasks_for_video in grouped_normal.items():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            example_element = tasks_for_video[0]
            print(f"\n[Rank {args.rank}] >>> PROCESSING VIDEO GROUP: {video_id} | n_tasks={len(tasks_for_video)}")
            video_full_path = resolve_video_path(paths, example_element, print)
            if video_full_path is None:
                continue

            stop_monitor = threading.Event()
            power_readings = []
            monitor_t = None
            if use_nvml:
                monitor_t = threading.Thread(
                    target=power_monitor_thread,
                    args=(nvml_handle, stop_monitor, power_readings),
                )
                monitor_t.start()

            master_cache = None
            query_base_cache = None
            inc_sampler = None
            cumulative_frame_times = []
            cumulative_cache_size_history = []
            cumulative_ingestion_time_sec = 0.0

            try:
                master_cache = _prime_cache_sw(
                    model=model,
                    processor=processor,
                    cache_cfg=cache_cfg,
                    prompt_path=paths["prompt_path"],
                    verbose=verbose,
                )
                print("[DEBUG SARO] VIDEO FULL PATH :::", video_full_path)

                inc_sampler = IncrementalSlidingWindowSampler(
                    video_path=video_full_path,
                    window_size=window_size,
                    stride=stride,
                    target_fps=fps_val,
                    max_frames=cache_cfg.get("MAX_FRAMES_TO_PROCESS", 100000),
                )

                ordered_times, tasks_by_time = group_tasks_by_time(tasks_for_video)
                print(
                    f"[Rank {args.rank}] DEBUG STEP: sampler created | "
                    f"is_open={inc_sampler.is_open()} | "
                    f"native_fps={inc_sampler.native_fps} | "
                    f"skip={inc_sampler.skip}",
                    flush=True,
                )

                for target_time in ordered_times:
                    delta_ingestion_start = time.time()
                    committed_batches, pending_tail = inc_sampler.advance_until(target_time)
                    print(
                        f"[Rank {args.rank}] DEBUG STEP: after advance_until({target_time}) | "
                        f"committed_batches={len(committed_batches)} | pending_tail={len(pending_tail)}",
                        flush=True,
                    )

                    committed_hist = []
                    if len(committed_batches) > 0:
                        print(f"[Rank {args.rank}] DEBUG STEP: before append committed", flush=True)
                        master_cache, committed_frame_times, committed_hist = _append_frame_batches_to_existing_cache_sw(
                            frame_batches=committed_batches,
                            model=model,
                            processor=processor,
                            cache_instance=master_cache,
                            size=tuple(res_val),
                            verbose=verbose,
                        )
                        cumulative_frame_times.extend(committed_frame_times)
                        cumulative_cache_size_history.extend(committed_hist)
                        print(f"[Rank {args.rank}] DEBUG STEP: after append committed", flush=True)

                    delta_ingestion_end = time.time()
                    delta_ingestion_time_sec = delta_ingestion_end - delta_ingestion_start
                    cumulative_ingestion_time_sec += delta_ingestion_time_sec

                    query_base_cache = clone_cache_for_question(master_cache)
                    tail_ingestion_time_sec = 0.0
                    tail_cache_hist = []

                    if len(pending_tail) > 0:
                        tail_start = time.time()
                        query_base_cache, _, tail_cache_hist = _append_frame_batches_to_existing_cache_sw(
                            frame_batches=[pending_tail],
                            model=model,
                            processor=processor,
                            cache_instance=query_base_cache,
                            size=tuple(res_val),
                            verbose=verbose,
                        )
                        tail_end = time.time()
                        tail_ingestion_time_sec = tail_end - tail_start

                    avg_frame_time = (
                        sum(cumulative_frame_times) / len(cumulative_frame_times)
                        if cumulative_frame_times else 0.0
                    )
                    question_cache_history = list(cumulative_cache_size_history) + list(tail_cache_hist)

                    shared_incremental_ingestion_time_sec = delta_ingestion_time_sec + tail_ingestion_time_sec
                    shared_full_ingestion_time_sec = cumulative_ingestion_time_sec + tail_ingestion_time_sec

                    for element in tasks_by_time[target_time]:
                        annotation_id = element.get("annotation_id")
                        question_time_type = element["absolute_question_time_type"]
                        result_key = element["result_key"]
                        if result_key in processed_keys:
                            continue

                        question, correct_answer, options, mcqa_prompt_text = build_mcqa_prompt_and_options(element)
                        open_ended_prompt_text = build_open_ended_prompt(element)

                        mcqa_cache = clone_cache_for_question(query_base_cache)
                        open_ended_cache = clone_cache_for_question(query_base_cache)

                        mcqa_retrieved = maybe_retrieve_for_qa(
                            cache_instance=mcqa_cache,
                            model=model,
                            processor=processor,
                            cache_cfg=cache_cfg,
                            question=question,
                            verbose=verbose,
                        )
                        open_ended_retrieved = maybe_retrieve_for_qa(
                            cache_instance=open_ended_cache,
                            model=model,
                            processor=processor,
                            cache_cfg=cache_cfg,
                            question=question,
                            verbose=verbose,
                        )

                        if hasattr(mcqa_cache, "set_QA"):
                            mcqa_cache.set_QA(True)
                        if hasattr(open_ended_cache, "set_QA"):
                            open_ended_cache.set_QA(True)

                        open_ended_gen_params = dict(gen_params)
                        open_ended_gen_params["max_new_tokens"] = gen_params.get("open_ended_max_new_tokens", 48)
                        open_ended_gen_params["temperature"] = gen_params.get(
                            "open_ended_temperature",
                            gen_params.get("temperature", 0.5)
                        )
                        open_ended_gen_params["top_p"] = gen_params.get(
                            "open_ended_top_p",
                            gen_params.get("top_p", 0.95)
                        )
                        open_ended_gen_params["top_k"] = gen_params.get("open_ended_top_k", 0)
                        open_ended_gen_params["repetition_penalty"] = gen_params.get(
                            "open_ended_repetition_penalty",
                            gen_params.get("repetition_penalty", 1.0)
                        )
                        open_ended_gen_params["ban_eos_steps"] = gen_params.get("open_ended_ban_eos_steps", 2)

                        mcqa_generation_start = time.time()
                        mcqa_output_text = run_mcqa_constrained_from_cache(
                            model=model,
                            processor=processor,
                            cache_instance=mcqa_cache,
                            prompt_text=mcqa_prompt_text,
                            num_options=len(options),
                            verbose=verbose,
                        )
                        mcqa_generation_end = time.time()
                        mcqa_generation_time_sec = mcqa_generation_end - mcqa_generation_start

                        open_ended_generation_start = time.time()
                        open_ended_output_text, open_ended_text_inputs, open_ended_input_ids = run_generation_from_cache(
                            model=model,
                            processor=processor,
                            cache_instance=open_ended_cache,
                            prompt_text=open_ended_prompt_text,
                            gen_params=open_ended_gen_params,
                            verbose=verbose,
                        )
                        open_ended_generation_end = time.time()
                        open_ended_generation_time_sec = open_ended_generation_end - open_ended_generation_start

                        mcqa_total_processing_time_sec = shared_incremental_ingestion_time_sec + mcqa_generation_time_sec
                        open_ended_total_processing_time_sec = shared_incremental_ingestion_time_sec + open_ended_generation_time_sec
                        mcqa_equivalent_full_processing_time_sec = shared_full_ingestion_time_sec + mcqa_generation_time_sec
                        open_ended_equivalent_full_processing_time_sec = shared_full_ingestion_time_sec + open_ended_generation_time_sec

                        avg_watts = sum(power_readings) / len(power_readings) if power_readings else 0.0
                        mcqa_total_joules = avg_watts * mcqa_equivalent_full_processing_time_sec
                        open_ended_total_joules = avg_watts * open_ended_equivalent_full_processing_time_sec

                        pred_idx_str, pred_answer_text, is_correct = finalize_prediction(
                            mcqa_output_text, options, correct_answer
                        )
                        open_ended_predicted_text, open_ended_exact_match = finalize_open_ended_prediction(
                            open_ended_output_text, correct_answer
                        )
                        print(f" -> MCQA Prediction: {pred_idx_str} | Correct: {is_correct}")
                        print(f" -> OPEN Prediction: {open_ended_predicted_text} | Exact match: {open_ended_exact_match}")

                        result_entry = {
                            "annotation_id": annotation_id,
                            "absolute_question_time_type": question_time_type,
                            "absolute_question_time_sec": float(target_time),
                            "video_id": video_id,
                            "question": question,
                            "options": options,
                            "correct_answer": correct_answer,
                            "model_output_raw": mcqa_output_text,
                            "predicted_number": pred_idx_str,
                            "predicted_text": pred_answer_text,
                            "is_correct": is_correct,
                            "open_ended_output_raw": open_ended_output_text,
                            "open_ended_predicted_text": open_ended_predicted_text,
                            "open_ended_exact_match": open_ended_exact_match,
                            "metrics": {
                                "avg_frame_time": round(avg_frame_time, 3),
                                "total_processing_time_sec": round(mcqa_equivalent_full_processing_time_sec, 3),
                                "incremental_processing_time_sec": round(mcqa_total_processing_time_sec, 3),
                                "equivalent_full_processing_time_sec": round(mcqa_equivalent_full_processing_time_sec, 3),
                                "generation_time_sec": round(mcqa_generation_time_sec, 3),
                                "mcqa_generation_time_sec": round(mcqa_generation_time_sec, 3),
                                "open_ended_generation_time_sec": round(open_ended_generation_time_sec, 3),
                                "mcqa_total_processing_time_sec": round(mcqa_total_processing_time_sec, 3),
                                "open_ended_total_processing_time_sec": round(open_ended_total_processing_time_sec, 3),
                                "mcqa_equivalent_full_processing_time_sec": round(mcqa_equivalent_full_processing_time_sec, 3),
                                "open_ended_equivalent_full_processing_time_sec": round(open_ended_equivalent_full_processing_time_sec, 3),
                                "delta_ingestion_time_sec": round(delta_ingestion_time_sec, 3),
                                "tail_ingestion_time_sec": round(tail_ingestion_time_sec, 3),
                                "cumulative_ingestion_time_sec": round(cumulative_ingestion_time_sec, 3),
                                "cache_memory": _to_int_bytes(get_cache_memory_size(master_cache) if master_cache else 0),
                                **get_cache_metrics(master_cache),
                                "qa_cache": {
                                    "retrieve_enabled": bool((cache_cfg.get("retrieve", {}) or {}).get("enabled", False)),
                                    "mcqa_retrieve_applied": bool(mcqa_retrieved),
                                    "open_ended_retrieve_applied": bool(open_ended_retrieved),
                                    "mcqa": get_cache_metrics(mcqa_cache),
                                    "open_ended": get_cache_metrics(open_ended_cache),
                                },
                                "hardware": {
                                    "rank": args.rank,
                                    "num_gpus": num_gpus,
                                    **get_hardware_metrics(num_gpus, max_vram_on_loading_per_gpu),
                                    "gpu_avg_watts": round(avg_watts, 2),
                                    "gpu_energy_joules": round(mcqa_total_joules, 2),
                                    "mcqa_gpu_energy_joules": round(mcqa_total_joules, 2),
                                    "open_ended_gpu_energy_joules": round(open_ended_total_joules, 2),
                                },
                            },
                            "cache_size_history": question_cache_history,
                        }

                        upsert_result(predictions, prediction_index, result_key, result_entry)
                        processed_keys.add(result_key)
                        save_json(predictions, output_path)

                        cleanup_locals(
                            mcqa_cache,
                            mcqa_output_text,
                            open_ended_cache,
                            open_ended_input_ids,
                            open_ended_text_inputs,
                            open_ended_output_text,
                        )
                        print(f"[Rank {args.rank}] Completed annotation_id={annotation_id}, time_type={question_time_type}.")

                    cleanup_locals(query_base_cache)
                    query_base_cache = None

            except Exception:
                import traceback
                traceback.print_exc()
            finally:
                if use_nvml and monitor_t:
                    stop_monitor.set()
                    monitor_t.join()
                if inc_sampler is not None:
                    try:
                        inc_sampler.close()
                    except Exception:
                        pass
                cleanup_video_cache_artifacts(video_id, verbose, master_cache, query_base_cache)
                cleanup_locals(master_cache, query_base_cache)

    # ---------------------------------------------------------------
    # EGOLIFE BRANCH
    # ---------------------------------------------------------------
    if egolife_tasks:
        egolife_base_path = resolve_egolife_base_path(paths, egolife_tasks)
        ordered_clips, clip_to_index = index_egolife_clips(egolife_base_path)
        ordered_targets, egolife_groups = group_egolife_tasks(egolife_tasks)

        print(f"[Rank {args.rank}] EgoLife base path: {egolife_base_path}")
        print(f"[Rank {args.rank}] EgoLife tasks: {len(egolife_tasks)}")
        print(f"[Rank {args.rank}] EgoLife target clips: {len(ordered_targets)}")

        master_cache = None
        if args.world_size > 1 and args.rank < EGOLIFE_NUM_RANKS:
            first_rank_clip_idx = first_clip_index_for_egolife_rank_day(ordered_clips, args.rank)
            current_clip_idx = first_rank_clip_idx - 1
            print(
                f"[Rank {args.rank}] EgoLife ingestion starts from "
                f"DAY{egolife_day_for_rank(args.rank)} | "
                f"first_clip_idx={first_rank_clip_idx}"
            )
        else:
            current_clip_idx = -1
        cumulative_frame_times = []
        cumulative_cache_size_history = []
        cumulative_ingestion_time_sec = 0.0

        stop_monitor = threading.Event()
        power_readings = []
        monitor_t = None
        if use_nvml:
            monitor_t = threading.Thread(
                target=power_monitor_thread,
                args=(nvml_handle, stop_monitor, power_readings),
            )
            monitor_t.start()

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            master_cache = _prime_cache_sw(
                model=model,
                processor=processor,
                cache_cfg=cache_cfg,
                prompt_path=paths["prompt_path"],
                verbose=verbose,
            )

            for target_day, target_clip_id in ordered_targets:
                target_idx = clip_to_index.get((target_day, target_clip_id))
                if target_idx is None:
                    print(f"[Rank {args.rank}] Missing EgoLife target clip: {target_day}/{target_clip_id}")
                    continue

                ingest_start_time = time.time()

                for clip_idx in range(current_clip_idx + 1, target_idx + 1):
                    clip_info = ordered_clips[clip_idx]
                    clip_path = clip_info["path"]
                    print(f"[Rank {args.rank}] EgoLife append clip {clip_info['day']}/{clip_info['clip_id']} | target_fps={EGOLIFE_TARGET_FPS}")

                    sampler = SlidingWindowSampler(
                        video_path=clip_path,
                        window_size=window_size,
                        stride=stride,
                        target_fps=EGOLIFE_TARGET_FPS,
                        max_frames=cache_cfg.get("MAX_FRAMES_TO_PROCESS", 100000),
                        until_second=None,
                    )

                    master_cache, clip_frame_times, clip_cache_hist = _append_sampler_to_existing_cache_sw(
                        sampler=sampler,
                        model=model,
                        processor=processor,
                        cache_instance=master_cache,
                        size=tuple(res_val),
                        verbose=verbose,
                    )
                    cumulative_frame_times.extend(clip_frame_times)
                    cumulative_cache_size_history.extend(clip_cache_hist)

                    del sampler
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

                    current_clip_idx = clip_idx

                ingest_end_time = time.time()
                delta_ingestion_time_sec = ingest_end_time - ingest_start_time
                cumulative_ingestion_time_sec += delta_ingestion_time_sec

                tasks_for_target = egolife_groups[(target_day, target_clip_id)]
                avg_frame_time = (
                    sum(cumulative_frame_times) / len(cumulative_frame_times)
                    if cumulative_frame_times else 0.0
                )

                for element in tasks_for_target:
                    annotation_id = element.get("annotation_id")
                    question_time_type = element.get("absolute_question_time_type")
                    result_key = element["result_key"]
                    if result_key in processed_keys:
                        continue

                    question, correct_answer, options, mcqa_prompt_text = build_mcqa_prompt_and_options(element)
                    open_ended_prompt_text = build_open_ended_prompt(element)
                    until_second = element.get("absolute_question_time_sec")

                    print(
                        f"\n[Rank {args.rank}] >>> EGOLIFE QUESTION annotation_id={annotation_id} | "
                        f"target={target_day}/{target_clip_id} | time_type={question_time_type}"
                    )

                    mcqa_cache = clone_cache_for_question(master_cache)
                    open_ended_cache = clone_cache_for_question(master_cache)

                    mcqa_retrieved = maybe_retrieve_for_qa(
                        cache_instance=mcqa_cache,
                        model=model,
                        processor=processor,
                        cache_cfg=cache_cfg,
                        question=question,
                        verbose=verbose,
                    )
                    open_ended_retrieved = maybe_retrieve_for_qa(
                        cache_instance=open_ended_cache,
                        model=model,
                        processor=processor,
                        cache_cfg=cache_cfg,
                        question=question,
                        verbose=verbose,
                    )

                    if hasattr(mcqa_cache, "set_QA"):
                        mcqa_cache.set_QA(True)
                    if hasattr(open_ended_cache, "set_QA"):
                        open_ended_cache.set_QA(True)

                    open_ended_gen_params = dict(gen_params)
                    open_ended_gen_params["max_new_tokens"] = gen_params.get("open_ended_max_new_tokens", 48)
                    open_ended_gen_params["temperature"] = gen_params.get(
                        "open_ended_temperature",
                        gen_params.get("temperature", 0.5)
                    )
                    open_ended_gen_params["top_p"] = gen_params.get(
                        "open_ended_top_p",
                        gen_params.get("top_p", 0.95)
                    )
                    open_ended_gen_params["top_k"] = gen_params.get("open_ended_top_k", 0)
                    open_ended_gen_params["repetition_penalty"] = gen_params.get(
                        "open_ended_repetition_penalty",
                        gen_params.get("repetition_penalty", 1.0)
                    )
                    open_ended_gen_params["ban_eos_steps"] = gen_params.get("open_ended_ban_eos_steps", 2)

                    mcqa_generation_start = time.time()
                    mcqa_output_text = run_mcqa_constrained_from_cache(
                        model=model,
                        processor=processor,
                        cache_instance=mcqa_cache,
                        prompt_text=mcqa_prompt_text,
                        num_options=len(options),
                        verbose=verbose,
                    )
                    mcqa_generation_end = time.time()
                    mcqa_generation_time_sec = mcqa_generation_end - mcqa_generation_start

                    open_ended_generation_start = time.time()
                    open_ended_output_text, open_ended_text_inputs, open_ended_input_ids = run_generation_from_cache(
                        model=model,
                        processor=processor,
                        cache_instance=open_ended_cache,
                        prompt_text=open_ended_prompt_text,
                        gen_params=open_ended_gen_params,
                        verbose=verbose,
                    )
                    open_ended_generation_end = time.time()
                    open_ended_generation_time_sec = open_ended_generation_end - open_ended_generation_start

                    mcqa_total_processing_time_sec = delta_ingestion_time_sec + mcqa_generation_time_sec
                    open_ended_total_processing_time_sec = delta_ingestion_time_sec + open_ended_generation_time_sec
                    mcqa_equivalent_full_processing_time_sec = cumulative_ingestion_time_sec + mcqa_generation_time_sec
                    open_ended_equivalent_full_processing_time_sec = cumulative_ingestion_time_sec + open_ended_generation_time_sec

                    avg_watts = sum(power_readings) / len(power_readings) if power_readings else 0.0
                    mcqa_total_joules = avg_watts * mcqa_equivalent_full_processing_time_sec
                    open_ended_total_joules = avg_watts * open_ended_equivalent_full_processing_time_sec

                    pred_idx_str, pred_answer_text, is_correct = finalize_prediction(
                        mcqa_output_text, options, correct_answer
                    )
                    open_ended_predicted_text, open_ended_exact_match = finalize_open_ended_prediction(
                        open_ended_output_text, correct_answer
                    )
                    print(f" -> MCQA Prediction: {pred_idx_str} | Correct: {is_correct}")
                    print(f" -> OPEN Prediction: {open_ended_predicted_text} | Exact match: {open_ended_exact_match}")

                    result_entry = {
                        "annotation_id": annotation_id,
                        "absolute_question_time_type": question_time_type,
                        "absolute_question_time_sec": until_second,
                        "video_id": f"{target_day}/{target_clip_id}",
                        "question": question,
                        "options": options,
                        "correct_answer": correct_answer,
                        "model_output_raw": mcqa_output_text,
                        "predicted_number": pred_idx_str,
                        "predicted_text": pred_answer_text,
                        "is_correct": is_correct,
                        "open_ended_output_raw": open_ended_output_text,
                        "open_ended_predicted_text": open_ended_predicted_text,
                        "open_ended_exact_match": open_ended_exact_match,
                        "metrics": {
                            "avg_frame_time": round(avg_frame_time, 3),
                            "total_processing_time_sec": round(mcqa_equivalent_full_processing_time_sec, 3),
                            "incremental_processing_time_sec": round(mcqa_total_processing_time_sec, 3),
                            "equivalent_full_processing_time_sec": round(mcqa_equivalent_full_processing_time_sec, 3),
                            "generation_time_sec": round(mcqa_generation_time_sec, 3),
                            "mcqa_generation_time_sec": round(mcqa_generation_time_sec, 3),
                            "open_ended_generation_time_sec": round(open_ended_generation_time_sec, 3),
                            "mcqa_total_processing_time_sec": round(mcqa_total_processing_time_sec, 3),
                            "open_ended_total_processing_time_sec": round(open_ended_total_processing_time_sec, 3),
                            "mcqa_equivalent_full_processing_time_sec": round(mcqa_equivalent_full_processing_time_sec, 3),
                            "open_ended_equivalent_full_processing_time_sec": round(open_ended_equivalent_full_processing_time_sec, 3),
                            "delta_ingestion_time_sec": round(delta_ingestion_time_sec, 3),
                            "tail_ingestion_time_sec": 0.0,
                            "cumulative_ingestion_time_sec": round(cumulative_ingestion_time_sec, 3),
                            "cache_memory": get_cache_memory_size(master_cache) if master_cache else 0,
                            **get_cache_metrics(master_cache),
                            "qa_cache": {
                                "retrieve_enabled": bool((cache_cfg.get("retrieve", {}) or {}).get("enabled", False)),
                                "mcqa_retrieve_applied": bool(mcqa_retrieved),
                                "open_ended_retrieve_applied": bool(open_ended_retrieved),
                                "mcqa": get_cache_metrics(mcqa_cache),
                                "open_ended": get_cache_metrics(open_ended_cache),
                            },
                            "hardware": {
                                "rank": args.rank,
                                "num_gpus": num_gpus,
                                **get_hardware_metrics(num_gpus, max_vram_on_loading_per_gpu),
                                "gpu_avg_watts": round(avg_watts, 2),
                                "gpu_energy_joules": round(mcqa_total_joules, 2),
                                "mcqa_gpu_energy_joules": round(mcqa_total_joules, 2),
                                "open_ended_gpu_energy_joules": round(open_ended_total_joules, 2),
                            },
                        },
                        "cache_size_history": list(cumulative_cache_size_history),
                    }

                    upsert_result(predictions, prediction_index, result_key, result_entry)
                    processed_keys.add(result_key)
                    save_json(predictions, output_path)

                    cleanup_locals(
                        mcqa_cache,
                        mcqa_output_text,
                        open_ended_cache,
                        open_ended_input_ids,
                        open_ended_text_inputs,
                        open_ended_output_text,
                    )
                    print(f"[Rank {args.rank}] Completed EgoLife annotation_id={annotation_id}, time_type={question_time_type}.")

        finally:
            if monitor_t:
                stop_monitor.set()
                monitor_t.join()
            cleanup_video_cache_artifacts("EgoLife", verbose, master_cache)
            cleanup_locals(master_cache)

    save_json(predictions, output_path)

    if use_nvml:
        try:
            import pynvml
            pynvml.nvmlShutdown()
        except Exception:
            pass

if __name__ == "__main__":
    set_all_seeds(seed=global_seed)
    main()
