from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter


RECALL_ORDER = [
    "instant_recall",
    "short_term_recall",
    "short_mid_term_recall",
    "mid_term_recall",
    "mid_long_term_recall",
    "long_term_recall",
    "ultra_long_term",
]

RECALL_COLUMN_SUFFIX: Dict[str, str] = {
    "instant_recall": "instant",
    "short_term_recall": "short",
    "short_mid_term_recall": "short-mid",
    "mid_term_recall": "mid",
    "mid_long_term_recall": "mid-long",
    "long_term_recall": "long",
    "ultra_long_term": "ultra-long",
}

# Metrics aggregated in the global sheet.
# Only the avg_ column is created for every numeric metric.
NUMERIC_METRIC_PATHS: Dict[str, Tuple[str, ...]] = {
    "absolute_question_time_sec": ("absolute_question_time_sec",),
    "avg_frame_time_sec": ("metrics", "avg_frame_time"),
    "total_processing_time_sec": ("metrics", "total_processing_time_sec"),
    "incremental_processing_time_sec": ("metrics", "incremental_processing_time_sec"),
    "equivalent_full_processing_time_sec": (
        "metrics",
        "equivalent_full_processing_time_sec",
    ),
    "generation_time_sec": ("metrics", "generation_time_sec"),
    "mcqa_generation_time_sec": ("metrics", "mcqa_generation_time_sec"),
    "open_ended_generation_time_sec": (
        "metrics",
        "open_ended_generation_time_sec",
    ),
    "mcqa_total_processing_time_sec": (
        "metrics",
        "mcqa_total_processing_time_sec",
    ),
    "open_ended_total_processing_time_sec": (
        "metrics",
        "open_ended_total_processing_time_sec",
    ),
    "mcqa_equivalent_full_processing_time_sec": (
        "metrics",
        "mcqa_equivalent_full_processing_time_sec",
    ),
    "open_ended_equivalent_full_processing_time_sec": (
        "metrics",
        "open_ended_equivalent_full_processing_time_sec",
    ),
    "delta_ingestion_time_sec": ("metrics", "delta_ingestion_time_sec"),
    "tail_ingestion_time_sec": ("metrics", "tail_ingestion_time_sec"),
    "cumulative_ingestion_time_sec": ("metrics", "cumulative_ingestion_time_sec"),
    "cache_memory_bytes": ("metrics", "cache_memory", "bytes"),
    "cache_memory_mb": ("metrics", "cache_memory", "MB"),
    "cache_memory_gb": ("metrics", "cache_memory", "GB"),
    "gpu_max_vram_on_loading_total_gb": (
        "metrics",
        "hardware",
        "gpu_max_vram_on_loading_total_GB",
    ),
    "gpu_max_vram_total_gb": (
        "metrics",
        "hardware",
        "gpu_max_vram_total_GB",
    ),
    "gpu_avg_watts": ("metrics", "hardware", "gpu_avg_watts"),
    "gpu_energy_joules": ("metrics", "hardware", "gpu_energy_joules"),
    "mcqa_gpu_energy_joules": (
        "metrics",
        "hardware",
        "mcqa_gpu_energy_joules",
    ),
    "open_ended_gpu_energy_joules": (
        "metrics",
        "hardware",
        "open_ended_gpu_energy_joules",
    ),
}


def load_json_flexible(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()

    if not text:
        return []

    try:
        data = json.loads(text)

        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]

        if isinstance(data, dict):
            return [data]

    except json.JSONDecodeError:
        pass

    fragment = text.rstrip(", \n\r\t")

    if not fragment.startswith("["):
        fragment = f"[{fragment}]"

    fragment = re.sub(r",\s*]", "]", fragment)

    try:
        data = json.loads(fragment)

        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]

    except json.JSONDecodeError:
        pass

    items: List[Dict[str, Any]] = []

    for line in text.splitlines():
        line = line.strip().rstrip(",")

        if not line:
            continue

        try:
            obj = json.loads(line)

            if isinstance(obj, dict):
                items.append(obj)

        except json.JSONDecodeError:
            continue

    if items:
        return items

    raise ValueError(f"Unrecognised JSON format: {path}")


def natural_key(path: Path) -> List[Any]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def load_run_predictions(run_dir: Path) -> List[Dict[str, Any]]:
    """Load the predictions of one experiment folder.

    Uses merged.json when present; otherwise merges every part_<rank>.json written
    by the runners, keeping the last occurrence of each
    (annotation_id, recall regime, video_id) key.
    """
    merged_path = run_dir / "merged.json"
    if merged_path.exists():
        return load_json_flexible(merged_path)

    part_files = sorted(run_dir.glob("part_*.json"), key=natural_key)
    if not part_files:
        raise FileNotFoundError(f"No merged.json or part_*.json found in {run_dir}")

    merged_by_key: Dict[str, Dict[str, Any]] = {}
    for part in part_files:
        for item in load_json_flexible(part):
            key = (
                f"{item.get('annotation_id')}::"
                f"{item.get('absolute_question_time_type')}::"
                f"{item.get('video_id', '')}"
            )
            merged_by_key[key] = item
    return list(merged_by_key.values())


def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)

    return text.casefold()


def infer_is_correct(item: Dict[str, Any]) -> Optional[bool]:
    if isinstance(item.get("is_correct"), bool):
        return item["is_correct"]

    predicted_text = item.get("predicted_text")
    correct_answer = item.get("correct_answer")

    if predicted_text is not None and correct_answer is not None:
        return normalize_text(predicted_text) == normalize_text(correct_answer)

    pred_num = item.get("predicted_number")

    if pred_num is None:
        pred_num = item.get("model_output_raw")

    options = item.get("options")

    if pred_num is not None and isinstance(options, list) and correct_answer is not None:
        try:
            idx = int(str(pred_num).strip()) - 1

            if 0 <= idx < len(options):
                return normalize_text(options[idx]) == normalize_text(correct_answer)

        except Exception:
            pass

    return None


def get_nested(dct: Dict[str, Any], *keys: str) -> Any:
    cur: Any = dct

    for key in keys:
        if not isinstance(cur, dict):
            return None

        cur = cur.get(key)

    return cur


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None

    try:
        return float(value)
    except Exception:
        return None


def safe_mean(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]

    return sum(vals) / len(vals) if vals else None


def folder_size_bytes(folder: Path) -> int:
    total = 0

    for p in folder.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass

    return total


def bytes_to_mb(n: int) -> float:
    return n / (1024**2)


def bytes_to_gb(n: int) -> float:
    return n / (1024**3)


def build_annotation_index(
    annotation_items: List[Dict[str, Any]],
) -> Dict[Any, Dict[str, Any]]:
    ann_by_id: Dict[Any, Dict[str, Any]] = {}
    duplicate_ids = set()

    for ann in annotation_items:
        ann_id = ann.get("annotation_id")

        if ann_id in ann_by_id:
            duplicate_ids.add(ann_id)
            continue

        ann_by_id[ann_id] = ann

    if duplicate_ids:
        print(
            f"[WARN] {len(duplicate_ids)} duplicated annotation_id values "
            f"in the annotation file; the first occurrence is used."
        )

    return ann_by_id


def compute_duplicate_count(items: List[Dict[str, Any]]) -> int:
    seen = set()
    duplicates = 0

    for item in items:
        key = (
            item.get("annotation_id"),
            item.get("absolute_question_time_type"),
            item.get("video_id"),
            normalize_text(item.get("question")),
        )

        if key in seen:
            duplicates += 1
        else:
            seen.add(key)

    return duplicates


def collect_cache_size_history(items: List[Dict[str, Any]]) -> List[float]:
    out: List[float] = []

    for item in items:
        hist = item.get("cache_size_history")

        if isinstance(hist, list):
            for v in hist:
                fv = to_float(v)

                if fv is not None:
                    out.append(fv)

    return out


def accuracy_from_bool_series(values: Iterable[Optional[bool]]) -> Optional[float]:
    scored = [bool(v) for v in values if v is not None]

    if not scored:
        return None

    return sum(scored) / len(scored)


def sorted_unique_strings(values: Iterable[Any]) -> List[str]:
    return sorted(
        {
            str(v).strip()
            for v in values
            if v is not None and str(v).strip() != ""
        }
    )


def excel_column_token(value: Any) -> str:
    text = str(value).strip().casefold()
    text = re.sub(r"[^a-z0-9]+", "-", text)

    return text.strip("-")


def format_sample_count_for_header(counts: Iterable[int]) -> str:
    unique_counts = sorted({int(c) for c in counts})

    if not unique_counts:
        return "n=0"

    if len(unique_counts) == 1:
        return f"n={unique_counts[0]}"

    return f"n={unique_counts[0]}-{unique_counts[-1]}"


def empty_primary_secondary_stats() -> Dict[str, int]:
    return {
        "n_primary": 0,
        "n_correct_primary": 0,
        "n_secondary": 0,
        "n_correct_secondary": 0,
    }


def add_primary_secondary_observation(
    stats_by_label: Dict[str, Dict[str, int]],
    label: str,
    is_correct: bool,
    role: str,
) -> None:
    if label is None or str(label).strip() == "":
        return

    label = str(label).strip()
    stats = stats_by_label.setdefault(label, empty_primary_secondary_stats())

    if role == "primary":
        stats["n_primary"] += 1

        if is_correct:
            stats["n_correct_primary"] += 1

    elif role == "secondary":
        stats["n_secondary"] += 1

        if is_correct:
            stats["n_correct_secondary"] += 1


def evaluate_run(run_dir: Path, ann_by_id: Dict[Any, Dict[str, Any]]) -> Dict[str, Any]:
    raw_items = load_run_predictions(run_dir)

    if not raw_items:
        raise ValueError(f"No prediction records found in {run_dir}")

    # Keep only the predictions that match an annotation.
    items: List[Dict[str, Any]] = []
    missing_annotation_ids = set()

    for item in raw_items:
        ann_id = item.get("annotation_id")

        if ann_id in ann_by_id:
            items.append(item)
        else:
            missing_annotation_ids.add(ann_id)

    if not items:
        raise ValueError(
            f"All records of {run_dir} were discarded: "
            f"no annotation_id matched the annotation file."
        )

    correctness = [infer_is_correct(x) for x in items]
    scored = [c for c in correctness if c is not None]
    n_correct = sum(1 for c in scored if c)

    summary: Dict[str, Any] = {
        "experiment_name": run_dir.name,
        "run_dir": str(run_dir),
        "n_items_raw": len(raw_items),
        "n_items_skipped_no_annotation": len(raw_items) - len(items),
        "n_items": len(items),
        "n_scored": len(scored),
        "n_correct": n_correct,
        "accuracy": (n_correct / len(scored)) if scored else None,
        "n_unique_video_id": len(
            {x.get("video_id") for x in items if x.get("video_id") is not None}
        ),
        "n_unique_annotation_id": len(
            {
                x.get("annotation_id")
                for x in items
                if x.get("annotation_id") is not None
            }
        ),
        "duplicate_records_present": compute_duplicate_count(items),
        "n_missing_annotation_ids": len(
            [x for x in missing_annotation_ids if x is not None]
        ),
    }

    disk_bytes = folder_size_bytes(run_dir)

    summary["disk_size_bytes"] = disk_bytes
    summary["disk_size_mb"] = bytes_to_mb(disk_bytes)
    summary["disk_size_gb"] = bytes_to_gb(disk_bytes)

    cache_hist = collect_cache_size_history(items)
    summary["avg_cache_size_history"] = safe_mean(cache_hist) if cache_hist else None

    num_gpus_values = [
        to_float(get_nested(x, "metrics", "hardware", "num_gpus")) for x in items
    ]

    rank_values = [
        to_float(get_nested(x, "metrics", "hardware", "rank")) for x in items
    ]

    summary["avg_num_gpus"] = safe_mean(num_gpus_values)
    summary["avg_rank"] = safe_mean(rank_values)

    for metric_name, path in NUMERIC_METRIC_PATHS.items():
        values = [to_float(get_nested(item, *path)) for item in items]
        summary[f"avg_{metric_name}"] = safe_mean(values)

    recall_accuracy: Dict[str, Optional[float]] = {name: None for name in RECALL_ORDER}

    for recall_name in RECALL_ORDER:
        recall_values = [
            correctness[i]
            for i, item in enumerate(items)
            if str(item.get("absolute_question_time_type")) == recall_name
        ]

        recall_accuracy[recall_name] = accuracy_from_bool_series(recall_values)

    primary_category_to_values: Dict[str, List[bool]] = {}
    primary_category_no_ultra_to_values: Dict[str, List[bool]] = {}
    primary_category_no_long_ultra_to_values: Dict[str, List[bool]] = {}
    primary_category_recall_to_values: Dict[str, Dict[str, List[bool]]] = {}
    primary_secondary_label_stats: Dict[str, Dict[str, int]] = {}
    dataset_to_values: Dict[str, List[bool]] = {}

    for item, is_correct in zip(items, correctness):
        if is_correct is None:
            continue

        correct_bool = bool(is_correct)
        ann = ann_by_id[item.get("annotation_id")]

        dataset = ann.get("dataset")

        if dataset is not None and str(dataset).strip() != "":
            dataset_to_values.setdefault(str(dataset).strip(), []).append(correct_bool)

        # Sheet primary_category_accuracy:
        # accuracy computed on primaryLabel only.
        primary_label = ann.get("primaryLabel")
        primary_label_str = ""

        if primary_label is not None and str(primary_label).strip() != "":
            primary_label_str = str(primary_label).strip()

            primary_category_to_values.setdefault(primary_label_str, []).append(
                correct_bool
            )

            recall_name = str(item.get("absolute_question_time_type"))

            # Sheet primary_cat_no_ultra:
            # accuracy per primaryLabel over every regime except ultra_long_term.
            if recall_name != "ultra_long_term":
                primary_category_no_ultra_to_values.setdefault(
                    primary_label_str,
                    [],
                ).append(correct_bool)

            # Sheet primary_cat_no_long_ultra:
            # accuracy per primaryLabel over every regime except
            # long_term_recall and ultra_long_term.
            if recall_name not in {"long_term_recall", "ultra_long_term"}:
                primary_category_no_long_ultra_to_values.setdefault(
                    primary_label_str,
                    [],
                ).append(correct_bool)

            # Sheet primary_category_recall_time:
            # accuracy per primaryLabel x recall regime.
            if recall_name in RECALL_ORDER:
                primary_category_recall_to_values.setdefault(
                    primary_label_str,
                    {},
                ).setdefault(
                    recall_name,
                    [],
                ).append(correct_bool)

            add_primary_secondary_observation(
                primary_secondary_label_stats,
                primary_label_str,
                correct_bool,
                role="primary",
            )

        # Sheet primary_secondary_accuracy:
        # accuracy counting both primaryLabel and secondaryLabels.
        secondary_labels = ann.get("secondaryLabels")

        if isinstance(secondary_labels, list):
            for secondary_label in sorted_unique_strings(secondary_labels):
                # Avoid double counting when a label is both primary and
                # secondary.
                if primary_label_str and secondary_label == primary_label_str:
                    continue

                add_primary_secondary_observation(
                    primary_secondary_label_stats,
                    secondary_label,
                    correct_bool,
                    role="secondary",
                )

    primary_category_accuracy = {
        label: (sum(vals) / len(vals) if vals else None)
        for label, vals in primary_category_to_values.items()
    }

    primary_category_no_ultra_accuracy = {
        label: (sum(vals) / len(vals) if vals else None)
        for label, vals in primary_category_no_ultra_to_values.items()
    }

    primary_category_no_long_ultra_accuracy = {
        label: (sum(vals) / len(vals) if vals else None)
        for label, vals in primary_category_no_long_ultra_to_values.items()
    }

    primary_category_recall_accuracy: Dict[str, Dict[str, Optional[float]]] = {}
    primary_category_recall_counts: Dict[str, Dict[str, int]] = {}

    for label, recall_map in primary_category_recall_to_values.items():
        primary_category_recall_accuracy[label] = {}
        primary_category_recall_counts[label] = {}

        for recall_name in RECALL_ORDER:
            values = recall_map.get(recall_name, [])

            primary_category_recall_accuracy[label][recall_name] = (
                accuracy_from_bool_series(values)
            )

            primary_category_recall_counts[label][recall_name] = len(values)

    dataset_accuracy = {
        ds: (sum(vals) / len(vals) if vals else None)
        for ds, vals in dataset_to_values.items()
    }

    return {
        "summary": summary,
        "recall_accuracy": recall_accuracy,
        "primary_category_accuracy": primary_category_accuracy,
        "primary_category_no_ultra_accuracy": primary_category_no_ultra_accuracy,
        "primary_category_no_long_ultra_accuracy": (
            primary_category_no_long_ultra_accuracy
        ),
        "primary_category_recall_accuracy": primary_category_recall_accuracy,
        "primary_category_recall_counts": primary_category_recall_counts,
        "primary_secondary_label_stats": primary_secondary_label_stats,
        "dataset_accuracy": dataset_accuracy,
        "missing_annotation_ids": sorted(
            x for x in missing_annotation_ids if x is not None
        ),
    }


def make_global_sheet(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = [payload["summary"] for payload in results.values()]
    df = pd.DataFrame(rows)

    if df.empty:
        return df

    first_cols = [
        "experiment_name",
        "run_dir",
        "n_items_raw",
        "n_items_skipped_no_annotation",
        "n_items",
        "n_scored",
        "n_correct",
        "accuracy",
        "disk_size_mb",
        "disk_size_gb",
        "avg_cache_size_history",
    ]

    ordered = [c for c in first_cols if c in df.columns] + [
        c for c in df.columns if c not in first_cols
    ]

    return df[ordered].sort_values("experiment_name").reset_index(drop=True)


def make_recall_sheet(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for experiment_name, payload in results.items():
        row = {"experiment_name": experiment_name}

        for recall_name in RECALL_ORDER:
            row[recall_name] = payload["recall_accuracy"].get(recall_name)

        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["experiment_name"] + RECALL_ORDER)

    return pd.DataFrame(rows).sort_values("experiment_name").reset_index(drop=True)


def make_primary_category_sheet(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    all_primary_labels = sorted_unique_strings(
        label
        for payload in results.values()
        for label in payload["primary_category_accuracy"].keys()
    )

    rows = []

    for experiment_name, payload in results.items():
        row = {"experiment_name": experiment_name}

        for label in all_primary_labels:
            row[label] = payload["primary_category_accuracy"].get(label)

        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["experiment_name"] + all_primary_labels)

    return pd.DataFrame(rows).sort_values("experiment_name").reset_index(drop=True)


def make_primary_category_filtered_sheet(
    results: Dict[str, Dict[str, Any]],
    payload_key: str,
) -> pd.DataFrame:
    all_primary_labels = sorted_unique_strings(
        label
        for payload in results.values()
        for label in payload[payload_key].keys()
    )

    rows = []

    for experiment_name, payload in results.items():
        row = {"experiment_name": experiment_name}

        for label in all_primary_labels:
            row[label] = payload[payload_key].get(label)

        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["experiment_name"] + all_primary_labels)

    return pd.DataFrame(rows).sort_values("experiment_name").reset_index(drop=True)


def make_primary_category_recall_time_sheet(
    results: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    all_primary_labels = sorted_unique_strings(
        label
        for payload in results.values()
        for label in payload["primary_category_recall_accuracy"].keys()
    )

    # Internal map:
    # (label, recall_name) -> column name holding the sample count.
    column_name_by_key: Dict[Tuple[str, str], str] = {}

    columns = ["experiment_name"]

    for label in all_primary_labels:
        label_token = excel_column_token(label)

        for recall_name in RECALL_ORDER:
            recall_suffix = RECALL_COLUMN_SUFFIX[recall_name]

            counts_for_column = [
                payload.get("primary_category_recall_counts", {})
                .get(label, {})
                .get(recall_name, 0)
                for payload in results.values()
            ]

            sample_count_text = format_sample_count_for_header(counts_for_column)
            col_name = f"{label_token}-{recall_suffix} ({sample_count_text})"

            column_name_by_key[(label, recall_name)] = col_name
            columns.append(col_name)

    rows = []

    for experiment_name, payload in results.items():
        row = {"experiment_name": experiment_name}
        recall_accuracy_by_label = payload["primary_category_recall_accuracy"]

        for label in all_primary_labels:
            label_recall_map = recall_accuracy_by_label.get(label, {})

            for recall_name in RECALL_ORDER:
                col_name = column_name_by_key[(label, recall_name)]
                row[col_name] = label_recall_map.get(recall_name)

        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=columns)

    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values("experiment_name")
        .reset_index(drop=True)
    )


def make_primary_secondary_sheet(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for experiment_name, payload in results.items():
        stats_by_label = payload["primary_secondary_label_stats"]

        for label in sorted_unique_strings(stats_by_label.keys()):
            stats = stats_by_label[label]

            n_primary = stats.get("n_primary", 0)
            n_correct_primary = stats.get("n_correct_primary", 0)

            n_secondary = stats.get("n_secondary", 0)
            n_correct_secondary = stats.get("n_correct_secondary", 0)

            n_total = n_primary + n_secondary
            n_correct_total = n_correct_primary + n_correct_secondary

            rows.append(
                {
                    "experiment_name": experiment_name,
                    "label": label,
                    "n_primary": n_primary,
                    "n_correct_primary": n_correct_primary,
                    "accuracy_primary": (
                        n_correct_primary / n_primary if n_primary else None
                    ),
                    "n_secondary": n_secondary,
                    "n_correct_secondary": n_correct_secondary,
                    "accuracy_secondary": (
                        n_correct_secondary / n_secondary if n_secondary else None
                    ),
                    "n_total": n_total,
                    "n_correct_total": n_correct_total,
                    "accuracy_total": n_correct_total / n_total if n_total else None,
                }
            )

    columns = [
        "experiment_name",
        "label",
        "n_primary",
        "n_correct_primary",
        "accuracy_primary",
        "n_secondary",
        "n_correct_secondary",
        "accuracy_secondary",
        "n_total",
        "n_correct_total",
        "accuracy_total",
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["experiment_name", "label"])
        .reset_index(drop=True)
    )


def make_dataset_sheet(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    all_datasets = sorted_unique_strings(
        ds
        for payload in results.values()
        for ds in payload["dataset_accuracy"].keys()
    )

    rows = []

    for experiment_name, payload in results.items():
        row = {"experiment_name": experiment_name}

        for ds in all_datasets:
            row[ds] = payload["dataset_accuracy"].get(ds)

        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["experiment_name"] + all_datasets)

    return pd.DataFrame(rows).sort_values("experiment_name").reset_index(drop=True)


def autosize_and_style_excel(path: Path) -> None:
    wb = load_workbook(path)

    header_fill = PatternFill(fill_type="solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)

    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions

        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font

        for col_idx, column_cells in enumerate(ws.columns, start=1):
            max_len = 0

            for cell in column_cells:
                value = "" if cell.value is None else str(cell.value)
                max_len = max(max_len, len(value))

            ws.column_dimensions[get_column_letter(col_idx)].width = min(
                max(max_len + 2, 12),
                40,
            )

        percentage_col_idxs = set()

        if ws.title == "global_metrics":
            for idx, cell in enumerate(ws[1], start=1):
                if cell.value == "accuracy":
                    percentage_col_idxs.add(idx)

        elif ws.title in {
            "recall_accuracy",
            "primary_category_accuracy",
            "primary_cat_no_ultra",
            "primary_cat_no_long_ultra",
            "primary_category_recall_time",
            "dataset_accuracy",
        }:
            non_percentage_cols = {"experiment_name"}

            for idx, cell in enumerate(ws[1], start=1):
                if cell.value not in non_percentage_cols:
                    percentage_col_idxs.add(idx)

        elif ws.title == "primary_secondary_accuracy":
            for idx, cell in enumerate(ws[1], start=1):
                if isinstance(cell.value, str) and cell.value.startswith("accuracy"):
                    percentage_col_idxs.add(idx)

        for row in ws.iter_rows(min_row=2):
            for cell in row:
                if (
                    cell.column in percentage_col_idxs
                    and isinstance(cell.value, (float, int))
                ):
                    cell.number_format = "0.0%"


    wb.save(path)


def find_experiment_dirs(inference_root: Path) -> List[Path]:
    return sorted(
        [
            p
            for p in inference_root.iterdir()
            if p.is_dir() and (any(p.glob("part_*.json")) or (p / "merged.json").exists())
        ],
        key=lambda p: p.name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score every experiment folder under --inference-root that contains "
            "part_*.json predictions and write an Excel summary."
        )
    )

    parser.add_argument(
        "--inference-root",
        required=True,
        help="Folder with one sub-folder per experiment (outputs/).",
    )

    parser.add_argument(
        "--annotations",
        required=True,
        help="EgoStream annotation JSON file.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output Excel path. "
            "Default: <inference_root>/evaluation_summary.xlsx"
        ),
    )

    args = parser.parse_args()

    inference_root = Path(args.inference_root)
    annotations_path = Path(args.annotations)

    output_path = (
        Path(args.output) if args.output else inference_root / "evaluation_summary.xlsx"
    )

    if not inference_root.exists() or not inference_root.is_dir():
        raise ValueError(f"Invalid inference root: {inference_root}")

    if not annotations_path.exists():
        raise ValueError(f"Annotation file not found: {annotations_path}")

    annotation_items = load_json_flexible(annotations_path)
    ann_by_id = build_annotation_index(annotation_items)

    experiment_dirs = find_experiment_dirs(inference_root)

    if not experiment_dirs:
        raise ValueError(
            f"No experiment folder with predictions found in {inference_root}"
        )

    print(f"Annotations loaded : {len(annotation_items)}")
    print(f"Experiments found  : {len(experiment_dirs)}")

    results: Dict[str, Dict[str, Any]] = {}
    skipped: List[Tuple[str, str]] = []

    for run_dir in experiment_dirs:
        try:
            print(f"[RUN] {run_dir.name}")
            results[run_dir.name] = evaluate_run(run_dir, ann_by_id)

        except Exception as exc:
            skipped.append((run_dir.name, str(exc)))
            print(f"[SKIP] {run_dir.name}: {exc}")

    if not results:
        raise ValueError("No experiment could be evaluated.")

    global_df = make_global_sheet(results)
    standard_results = results

    recall_df = make_recall_sheet(standard_results)
    primary_category_df = make_primary_category_sheet(standard_results)

    primary_category_no_ultra_df = make_primary_category_filtered_sheet(
        standard_results,
        "primary_category_no_ultra_accuracy",
    )

    primary_category_no_long_ultra_df = make_primary_category_filtered_sheet(
        standard_results,
        "primary_category_no_long_ultra_accuracy",
    )

    primary_category_recall_time_df = make_primary_category_recall_time_sheet(
        standard_results
    )

    primary_secondary_df = make_primary_secondary_sheet(standard_results)
    dataset_df = make_dataset_sheet(standard_results)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        global_df.to_excel(writer, sheet_name="global_metrics", index=False)

        recall_df.to_excel(
            writer,
            sheet_name="recall_accuracy",
            index=False,
        )

        primary_category_df.to_excel(
            writer,
            sheet_name="primary_category_accuracy",
            index=False,
        )

        primary_category_no_ultra_df.to_excel(
            writer,
            sheet_name="primary_cat_no_ultra",
            index=False,
        )

        primary_category_no_long_ultra_df.to_excel(
            writer,
            sheet_name="primary_cat_no_long_ultra",
            index=False,
        )

        primary_category_recall_time_df.to_excel(
            writer,
            sheet_name="primary_category_recall_time",
            index=False,
        )

        primary_secondary_df.to_excel(
            writer,
            sheet_name="primary_secondary_accuracy",
            index=False,
        )

        dataset_df.to_excel(
            writer,
            sheet_name="dataset_accuracy",
            index=False,
        )


    autosize_and_style_excel(output_path)

    print(f"\nExcel written: {output_path}")
    print(f"Experiments saved: {len(results)}")
    
    if skipped:
        print(f"Experiments skipped: {len(skipped)}")

        for name, reason in skipped:
            print(f" - {name}: {reason}")


if __name__ == "__main__":
    main()