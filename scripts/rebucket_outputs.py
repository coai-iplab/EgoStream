#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from run_benchmark_qwen3vl import (  # noqa: E402
    build_result_key,
    expand_annotation_tasks,
    result_entry_has_both_modes,
    split_tasks_by_rank,
)


def load_json_list(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected a JSON list in {path}")
    return data


def write_json_list(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def result_key_from_prediction(item):
    annotation_id = item.get("annotation_id")
    question_type = item.get("absolute_question_time_type")
    if annotation_id is None or question_type is None:
        return None
    return build_result_key(annotation_id, question_type)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Redistribute completed EgoStream part_*.json predictions to a new world_size."
    )
    parser.add_argument("--config", required=True, help="Config used to load annotations.")
    parser.add_argument("--src-dir", required=True, help="Existing inference output directory.")
    parser.add_argument("--dst-dir", required=True, help="Destination inference output directory.")
    parser.add_argument("--src-world-size", type=int, required=True)
    parser.add_argument("--dst-world-size", type=int, required=True)
    parser.add_argument(
        "--clear-dst",
        action="store_true",
        help="Remove destination part_*.json files before writing rebucketed outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    with Path(config["paths"]["annotations_file"]).open("r", encoding="utf-8") as f:
        expanded = expand_annotation_tasks(json.load(f))

    result_to_dst_rank = {}
    result_order = {}
    for rank in range(args.dst_world_size):
        rank_tasks, _, _, _ = split_tasks_by_rank(expanded, rank, args.dst_world_size)
        for idx, task in enumerate(rank_tasks):
            result_to_dst_rank[task["result_key"]] = rank
            result_order[task["result_key"]] = (rank, idx)

    if args.clear_dst and dst_dir.exists():
        for path in dst_dir.glob("part_*.json"):
            path.unlink()

    by_rank = {
        rank: load_json_list(dst_dir / f"part_{rank}.json")
        for rank in range(args.dst_world_size)
    }
    seen = {
        rank: {
            key
            for key in (result_key_from_prediction(item) for item in items)
            if key is not None
        }
        for rank, items in by_rank.items()
    }

    migrated = 0
    skipped_incomplete = 0
    skipped_unknown = 0
    skipped_duplicate = 0

    for src_rank in range(args.src_world_size):
        for item in load_json_list(src_dir / f"part_{src_rank}.json"):
            key = result_key_from_prediction(item)
            if key is None or not result_entry_has_both_modes(item):
                skipped_incomplete += 1
                continue
            dst_rank = result_to_dst_rank.get(key)
            if dst_rank is None:
                skipped_unknown += 1
                continue
            if key in seen[dst_rank]:
                skipped_duplicate += 1
                continue
            by_rank[dst_rank].append(item)
            seen[dst_rank].add(key)
            migrated += 1

    for rank, items in by_rank.items():
        items.sort(key=lambda item: result_order.get(result_key_from_prediction(item), (rank, 10**12)))
        write_json_list(dst_dir / f"part_{rank}.json", items)

    print(f"migrated={migrated}")
    print(f"skipped_incomplete={skipped_incomplete}")
    print(f"skipped_unknown={skipped_unknown}")
    print(f"skipped_duplicate={skipped_duplicate}")
    for rank in range(args.dst_world_size):
        print(f"rank {rank}: {len(by_rank[rank])}")


if __name__ == "__main__":
    main()
