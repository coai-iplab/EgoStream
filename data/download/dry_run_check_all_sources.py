#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import requests


TRIM_RE = re.compile(
    r"^(?P<base>.+)_(?P<start>-?\d+(?:\.\d+)?)_(?P<end>-?\d+(?:\.\d+)?)$"
)

HD_EPIC_ID_RE = re.compile(
    r"^(P\d{2}-\d{8}-\d{6})_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)$"
)

DEFAULT_EGO4D_DATASETS_TO_TRY = [
    "clips",
    "full_scale",
    "video_540ss",
]

FAKE_UID = "00000000-0000-0000-0000-000000000000"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_lines(path: Path, lines: list[str] | set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for line in sorted(lines):
            f.write(str(line) + "\n")


def dataset_norm(x: str) -> str:
    aliases = {
        "ego4d": "Ego4D",
        "egotempo": "EgoTempo",
        "egotempo_trimmed_clips": "EgoTempo",
        "egotempo/trimmed_clips": "EgoTempo",
        "multihop-egoqa": "MultiHop-EgoQA",
        "hd_epic": "HD-EPIC",
        "hd-epic": "HD-EPIC",
        "egolife": "EgoLife",
    }
    return aliases.get(str(x).lower(), str(x))


def parse_trim(video_id: str) -> tuple[str, float, float] | None:
    m = TRIM_RE.match(video_id)
    if not m:
        return None

    base = m.group("base")
    start = max(0.0, float(m.group("start")))
    end = float(m.group("end"))

    if end <= start:
        return None

    return base, start, end


def find_ego4d_exe() -> str:
    exe = shutil.which("ego4d")
    if exe:
        return exe

    possible = Path(sys.executable).parent / "ego4d.exe"
    if possible.exists():
        return str(possible)

    possible = Path(sys.executable).parent / "ego4d"
    if possible.exists():
        return str(possible)

    raise FileNotFoundError(
        "Comando ego4d non trovato. Installa con:\n"
        "  python -m pip install -U ego4d\n"
        "Poi verifica:\n"
        "  ego4d --help"
    )


def extract_ego4d_check_uid(ann: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    """
    Regole:
    - Ego4D / Ego4D_EM_Test: NON trim. Usa video_id così com'è.
    - EgoTempo / MultiHop-EgoQA: se video_id è uid_start_end, usa uid base.
    - EgoLife: skip, viene controllato separatamente su HF.
    - HD-EPIC: skip, viene controllato separatamente su endpoint data.bris.
    """
    dataset = dataset_norm(ann.get("dataset", ""))
    subfolder = str(ann.get("subfolder") or "")
    video_id = ann.get("video_id")

    ref = {
        "dataset": dataset,
        "subfolder": subfolder,
        "annotation_id": ann.get("annotation_id"),
        "video_id": video_id,
        "source_file": ann.get("source_file"),
        "checked_uid": None,
        "rule": None,
        "trim_start": None,
        "trim_end": None,
    }

    if dataset not in {"Ego4D", "EgoTempo", "MultiHop-EgoQA"}:
        ref["rule"] = "skipped_not_ego4d_source"
        return None, ref

    if not isinstance(video_id, str):
        ref["rule"] = "skipped_video_id_not_string"
        return None, ref

    if dataset == "Ego4D":
        ref["checked_uid"] = video_id
        ref["rule"] = "ego4d_or_em_no_trim"
        return video_id, ref

    trim = parse_trim(video_id)
    if trim:
        base, start, end = trim
        ref["checked_uid"] = base
        ref["rule"] = "trimmed_id_base_uid"
        ref["trim_start"] = start
        ref["trim_end"] = end
        return base, ref

    ref["checked_uid"] = video_id
    ref["rule"] = "ego4d_like_non_trimmed"
    return video_id, ref


def collect_ego4d_uids(
    annotations: list[dict[str, Any]],
) -> tuple[set[str], dict[str, list[dict[str, Any]]], Counter]:
    uid_to_refs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rule_counts: Counter = Counter()

    for ann in annotations:
        uid, ref = extract_ego4d_check_uid(ann)
        rule_counts[ref["rule"]] += 1

        if uid:
            uid_to_refs[uid].append(ref)

    return set(uid_to_refs.keys()), uid_to_refs, rule_counts


def parse_missing_uids_from_ego4d_output(text: str) -> set[str] | None:
    if "could not be found in the manifest" not in text:
        return None

    start = text.rfind("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return None

    block = text[start : end + 1]

    try:
        value = ast.literal_eval(block)
    except Exception:
        return None

    if isinstance(value, set):
        return {str(x) for x in value}

    return None


def probe_ego4d_manifest(
    ego4d_exe: str,
    output_dir: Path,
    uid_file: Path,
    dataset_name: str,
    version: str,
    aws_profile: str,
) -> subprocess.CompletedProcess:
    cmd = [
        ego4d_exe,
        "--output_directory",
        str(output_dir),
        "--datasets",
        dataset_name,
        "--video_uid_file",
        str(uid_file),
        "--version",
        version,
        "--yes",
    ]

    if aws_profile:
        cmd.extend(["--aws_profile_name", aws_profile])

    print()
    print("[EGO4D PROBE - NO VIDEO DOWNLOAD]")
    print(" ".join(cmd))

    return subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def check_ego4d_manifests(
    all_uids: set[str],
    download_root: Path,
    version: str,
    aws_profile: str,
    datasets_to_try: list[str],
) -> dict[str, Any]:
    """
    Trucco:
    aggiunge un UID finto alla lista. Così ego4d fallisce SEMPRE prima del download
    e restituisce il set degli UID non presenti nel manifest. Da lì deduciamo i found.
    """
    ego4d_exe = find_ego4d_exe()

    ego4d_output_dir = download_root / "ego4d_raw"
    work_dir = download_root / "_tmp" / "dry_run_ego4d_manifest_check"
    work_dir.mkdir(parents=True, exist_ok=True)

    routing: dict[str, set[str]] = {name: set() for name in datasets_to_try}
    remaining = set(all_uids)

    for dataset_name in datasets_to_try:
        if not remaining:
            break

        probe_uids = set(remaining)
        probe_uids.add(FAKE_UID)

        uid_file = work_dir / f"probe_{dataset_name}.txt"
        write_lines(uid_file, probe_uids)

        proc = probe_ego4d_manifest(
            ego4d_exe=ego4d_exe,
            output_dir=ego4d_output_dir,
            uid_file=uid_file,
            dataset_name=dataset_name,
            version=version,
            aws_profile=aws_profile,
        )

        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        missing = parse_missing_uids_from_ego4d_output(combined)

        if missing is None:
            print(f"[WARN] Impossibile parsare missing UID per dataset={dataset_name}")
            print("[STDOUT tail]")
            print((proc.stdout or "")[-3000:])
            print("[STDERR tail]")
            print((proc.stderr or "")[-3000:])
            continue

        missing.discard(FAKE_UID)

        found = remaining - missing
        routing[dataset_name].update(found)
        remaining = missing

        print(f"[INFO] {dataset_name}: found={len(found)} | remaining_missing={len(remaining)}")

    return {
        "routing": {k: sorted(v) for k, v in routing.items() if v},
        "missing": sorted(remaining),
    }


def collect_hd_epic_targets(
    annotations: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    targets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    malformed: list[dict[str, Any]] = []

    for ann in annotations:
        dataset = dataset_norm(ann.get("dataset", ""))
        if dataset != "HD-EPIC":
            continue

        video_id = ann.get("video_id")

        ref = {
            "dataset": dataset,
            "annotation_id": ann.get("annotation_id"),
            "video_id": video_id,
            "source_file": ann.get("source_file"),
            "subfolder": ann.get("subfolder"),
        }

        if not isinstance(video_id, str):
            ref["reason"] = "video_id_not_string"
            malformed.append(ref)
            continue

        m = HD_EPIC_ID_RE.match(video_id)
        if not m:
            ref["reason"] = "not_matching_hd_epic_trim_pattern"
            malformed.append(ref)
            continue

        base = m.group(1)
        start = max(0.0, float(m.group(2)))
        end = float(m.group(3))

        if end <= start:
            ref["reason"] = "invalid_trim_range"
            ref["base"] = base
            ref["start"] = start
            ref["end"] = end
            malformed.append(ref)
            continue

        ref["base"] = base
        ref["start"] = start
        ref["end"] = end
        targets[base].append(ref)

    return targets, malformed


def ckan_package_search(query: str) -> list[dict[str, Any]]:
    url = "https://data.bris.ac.uk/data/api/3/action/package_search"
    r = requests.get(url, params={"q": query, "rows": 20}, timeout=60)
    r.raise_for_status()
    payload = r.json()

    if not payload.get("success"):
        return []

    return payload.get("result", {}).get("results", [])


def ckan_find_hd_epic_url(base_video_id: str) -> str | None:
    participant = base_video_id.split("-")[0]
    wanted = f"{base_video_id}.mp4"

    queries = [
        f"{wanted}",
        f"HD-EPIC {wanted}",
        f"HD-EPIC/Videos/{participant}",
        f"HD-EPIC {participant}",
    ]

    for q in queries:
        try:
            packages = ckan_package_search(q)
        except Exception as e:
            print(f"[WARN] CKAN search failed for {q}: {e}")
            continue

        for pkg in packages:
            for res in pkg.get("resources", []):
                name = res.get("name") or ""
                url = res.get("url") or ""

                if name == wanted or wanted in name or wanted in url:
                    return url

    return None


def check_hd_epic(
    annotations: list[dict[str, Any]],
    skip_remote: bool,
) -> dict[str, Any]:
    targets, malformed = collect_hd_epic_targets(annotations)

    found = {}
    missing = {}

    print()
    print(f"[HD-EPIC] base videos to check: {len(targets)}")
    print(f"[HD-EPIC] malformed annotation ids: {len(malformed)}")

    if skip_remote:
        return {
            "base_video_count": len(targets),
            "clip_count": sum(len(v) for v in targets.values()),
            "found": {},
            "missing": sorted(targets.keys()),
            "malformed": malformed,
            "skipped_remote_check": True,
        }

    for i, base in enumerate(sorted(targets), start=1):
        print(f"[HD-EPIC CHECK] {i}/{len(targets)} {base}")
        url = ckan_find_hd_epic_url(base)

        if url:
            found[base] = {
                "url": url,
                "refs": targets[base],
            }
        else:
            missing[base] = {
                "refs": targets[base],
                "reason": "url_not_found",
            }

    return {
        "base_video_count": len(targets),
        "clip_count": sum(len(v) for v in targets.values()),
        "found_count": len(found),
        "missing_count": len(missing),
        "malformed_count": len(malformed),
        "found": found,
        "missing": missing,
        "malformed": malformed,
        "skipped_remote_check": False,
    }


def collect_egolife_targets(
    annotations: list[dict[str, Any]],
) -> tuple[set[tuple[str, str]], list[dict[str, Any]]]:
    targets: set[tuple[str, str]] = set()
    malformed: list[dict[str, Any]] = []

    for ann in annotations:
        dataset = dataset_norm(ann.get("dataset", ""))
        if dataset != "EgoLife":
            continue

        video_id = ann.get("video_id")

        if not isinstance(video_id, dict):
            malformed.append(
                {
                    "dataset": dataset,
                    "annotation_id": ann.get("annotation_id"),
                    "video_id": video_id,
                    "reason": "egolife_video_id_not_dict",
                }
            )
            continue

        for recall_name, payload in video_id.items():
            if not isinstance(payload, dict):
                continue

            day = payload.get("day")
            clip_id = payload.get("video_id")

            if day and clip_id:
                targets.add((str(day), str(clip_id)))

    return targets, malformed


def check_egolife_hf(
    annotations: list[dict[str, Any]],
    repo_id: str,
    skip_remote: bool,
) -> dict[str, Any]:
    targets, malformed = collect_egolife_targets(annotations)

    print()
    print(f"[EgoLife] target clips to check: {len(targets)}")
    print(f"[EgoLife] malformed annotation ids: {len(malformed)}")

    if skip_remote:
        return {
            "target_count": len(targets),
            "found": [],
            "missing": sorted([{"day": d, "video_id": v} for d, v in targets], key=lambda x: (x["day"], x["video_id"])),
            "malformed": malformed,
            "skipped_remote_check": True,
        }

    try:
        from huggingface_hub import HfApi
    except ImportError:
        return {
            "target_count": len(targets),
            "found": [],
            "missing": sorted([{"day": d, "video_id": v} for d, v in targets], key=lambda x: (x["day"], x["video_id"])),
            "malformed": malformed,
            "error": "huggingface_hub not installed. Install with: python -m pip install -U huggingface_hub",
            "skipped_remote_check": False,
        }

    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

    found = []
    missing = []

    for day, clip_id in sorted(targets):
        matches = [
            f for f in files
            if f.lower().endswith(".mp4")
            and day in f
            and clip_id in Path(f).name
        ]

        if matches:
            found.append(
                {
                    "day": day,
                    "video_id": clip_id,
                    "matches": matches[:10],
                    "match_count": len(matches),
                }
            )
        else:
            missing.append(
                {
                    "day": day,
                    "video_id": clip_id,
                    "reason": "no_matching_hf_file",
                }
            )

    return {
        "target_count": len(targets),
        "found_count": len(found),
        "missing_count": len(missing),
        "found": found,
        "missing": missing,
        "malformed": malformed,
        "skipped_remote_check": False,
        "repo_id": repo_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dry-run checker: verify whether all EgoStream source videos/clips can be found without downloading videos."
    )

    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)

    parser.add_argument("--aws-profile", default="ego4d")
    parser.add_argument("--ego4d-version", default="v2")
    parser.add_argument(
        "--ego4d-datasets-to-try",
        nargs="+",
        default=DEFAULT_EGO4D_DATASETS_TO_TRY,
        help="Default: clips full_scale video_540ss",
    )

    parser.add_argument("--skip-ego4d", action="store_true")
    parser.add_argument("--skip-hd-epic", action="store_true")
    parser.add_argument("--skip-egolife", action="store_true")

    parser.add_argument(
        "--skip-hd-epic-remote",
        action="store_true",
        help="Do not query data.bris for HD-EPIC URLs.",
    )
    parser.add_argument(
        "--skip-egolife-remote",
        action="store_true",
        help="Do not query Hugging Face for EgoLife files.",
    )
    parser.add_argument(
        "--egolife-repo",
        default="lmms-lab/EgoLife",
        help="Hugging Face dataset repo for EgoLife.",
    )

    args = parser.parse_args()

    annotations_path = args.annotations.expanduser().resolve()
    report_root = args.report_root.expanduser().resolve()
    report_root.mkdir(parents=True, exist_ok=True)

    annotations = load_json(annotations_path)
    if not isinstance(annotations, list):
        raise ValueError("Annotations JSON must be a list")

    dataset_counts = Counter(dataset_norm(a.get("dataset", "UNKNOWN")) for a in annotations)

    print("[DATASET COUNTS]")
    for dataset, count in dataset_counts.most_common():
        print(f"  {dataset}: {count}")

    final_report: dict[str, Any] = {
        "annotation_file": str(annotations_path),
        "annotation_count": len(annotations),
        "dataset_counts": dict(dataset_counts),
    }

    if not args.skip_ego4d:
        ego4d_uids, ego4d_uid_refs, ego4d_rule_counts = collect_ego4d_uids(annotations)

        print()
        print(f"[EGO4D-SOURCE] unique UIDs to check: {len(ego4d_uids)}")
        print("[EGO4D-SOURCE] extraction rules:")
        for rule, count in ego4d_rule_counts.most_common():
            print(f"  {rule}: {count}")

        write_lines(report_root / "ego4d_all_checked_uids.txt", ego4d_uids)
        write_json(report_root / "ego4d_uid_refs.json", ego4d_uid_refs)

        ego4d_result = check_ego4d_manifests(
            all_uids=ego4d_uids,
            download_root=report_root,
            version=args.ego4d_version,
            aws_profile=args.aws_profile,
            datasets_to_try=args.ego4d_datasets_to_try,
        )

        write_json(report_root / "ego4d_report.json", ego4d_result)
        write_lines(report_root / "ego4d_missing_uids.txt", set(ego4d_result["missing"]))

        for dataset_name, uids in ego4d_result["routing"].items():
            write_lines(report_root / f"ego4d_found_{dataset_name}.txt", set(uids))

        final_report["ego4d"] = {
            "unique_uid_count": len(ego4d_uids),
            "rule_counts": dict(ego4d_rule_counts),
            "routing_counts": {k: len(v) for k, v in ego4d_result["routing"].items()},
            "missing_count": len(ego4d_result["missing"]),
        }

    if not args.skip_hd_epic:
        hd_report = check_hd_epic(
            annotations=annotations,
            skip_remote=args.skip_hd_epic_remote,
        )
        write_json(report_root / "hd_epic_report.json", hd_report)
        write_lines(report_root / "hd_epic_missing_base_videos.txt", set(hd_report.get("missing", {}).keys()))
        final_report["hd_epic"] = {
            "base_video_count": hd_report.get("base_video_count"),
            "clip_count": hd_report.get("clip_count"),
            "found_count": hd_report.get("found_count"),
            "missing_count": hd_report.get("missing_count"),
            "malformed_count": hd_report.get("malformed_count"),
            "skipped_remote_check": hd_report.get("skipped_remote_check"),
        }

    if not args.skip_egolife:
        egolife_report = check_egolife_hf(
            annotations=annotations,
            repo_id=args.egolife_repo,
            skip_remote=args.skip_egolife_remote,
        )
        write_json(report_root / "egolife_report.json", egolife_report)

        missing_egolife = [
            f"{x.get('day')}/{x.get('video_id')}"
            for x in egolife_report.get("missing", [])
        ]
        write_lines(report_root / "egolife_missing_clips.txt", set(missing_egolife))

        final_report["egolife"] = {
            "target_count": egolife_report.get("target_count"),
            "found_count": egolife_report.get("found_count"),
            "missing_count": egolife_report.get("missing_count"),
            "malformed_count": len(egolife_report.get("malformed", [])),
            "skipped_remote_check": egolife_report.get("skipped_remote_check"),
            "repo_id": egolife_report.get("repo_id"),
            "error": egolife_report.get("error"),
        }

    write_json(report_root / "dry_run_summary.json", final_report)

    print()
    print("[SUMMARY]")
    print(json.dumps(final_report, indent=2, ensure_ascii=False))
    print()
    print(f"[WROTE] {report_root / 'dry_run_summary.json'}")
    print(f"[REPORT DIR] {report_root}")


if __name__ == "__main__":
    main()