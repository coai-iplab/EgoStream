#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

TRIM_RE = re.compile(
    r"^(?P<base>.+)_(?P<start>-?\d+(?:\.\d+)?)_(?P<end>-?\d+(?:\.\d+)?)$"
)

HD_EPIC_ID_RE = re.compile(
    r"^(P\d{2}-\d{8}-\d{6})_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)$"
)

IGNORE_DIRS = {
    ".git",
    ".cache",
    "__pycache__",
    ".ipynb_checkpoints",
    "#recycle",
    "_tmp",
}

EGO4D_LIKE_DATASETS = {
    "Ego4D",
    "EgoTempo",
    "EGOTEMPO",
    "MultiHop-EgoQA",
}

DEFAULT_EGO4D_DATASETS_TO_TRY = [
    "clips",
    "full_scale",
    "video_540ss",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print("[CMD]", " ".join(str(x) for x in cmd))
    return subprocess.run(cmd, check=check)


def dataset_norm(x: str) -> str:
    aliases = {
        "egotempo": "EgoTempo",
        "egotempo_trimmed_clips": "EgoTempo",
        "hd_epic": "HD-EPIC",
        "hd-epic": "HD-EPIC",
        "egolife": "EgoLife",
        "ego4d": "Ego4D",
        "multihop-egoqa": "MultiHop-EgoQA",
    }
    return aliases.get(str(x).lower(), str(x))


def safe_name(x: str) -> str:
    return str(x).replace("/", "_").replace("\\", "_").replace(":", "_")


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


def should_skip(p: Path) -> bool:
    return any(part in IGNORE_DIRS for part in p.parts)


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
        "ego4d command not found. Install it with:\n"
        "  python -m pip install -U ego4d\n"
        "Then check:\n"
        "  ego4d --help"
    )


def write_uid_file(path: Path, uids: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for uid in sorted(uids):
            f.write(uid + "\n")


def collect_ego4d_like_uids(annotations: list[dict[str, Any]]) -> set[str]:
    out = set()

    for ann in annotations:
        ds = dataset_norm(ann.get("dataset", ""))
        if ds not in EGO4D_LIKE_DATASETS:
            continue

        vid = ann.get("video_id")
        if not isinstance(vid, str):
            continue

        trim = parse_trim(vid)
        if trim:
            base, _, _ = trim
            out.add(base)
        else:
            out.add(vid)

    return out


def collect_hd_epic_targets(annotations: list[dict[str, Any]]) -> dict[str, list[tuple[str, float, float]]]:
    targets = defaultdict(list)

    for ann in annotations:
        ds = dataset_norm(ann.get("dataset", ""))
        if ds != "HD-EPIC":
            continue

        vid = ann.get("video_id")
        if not isinstance(vid, str):
            continue

        m = HD_EPIC_ID_RE.match(vid)
        if not m:
            print(f"[WARN] HD-EPIC video_id non in formato trim: {vid}")
            continue

        base = m.group(1)
        start = max(0.0, float(m.group(2)))
        end = float(m.group(3))

        if end <= start:
            print(f"[WARN] HD-EPIC trim invalido: {vid}")
            continue

        targets[base].append((vid, start, end))

    return targets


def collect_egolife_targets(annotations: list[dict[str, Any]]) -> set[tuple[str, str]]:
    out = set()

    for ann in annotations:
        ds = dataset_norm(ann.get("dataset", ""))
        if ds != "EgoLife":
            continue

        vid = ann.get("video_id")
        if not isinstance(vid, dict):
            continue

        for _, payload in vid.items():
            if not isinstance(payload, dict):
                continue

            day = payload.get("day")
            clip_id = payload.get("video_id")

            if day and clip_id:
                out.add((str(day), str(clip_id)))

    return out


def find_file_by_stem_or_contains(root: Path, video_id: str) -> Path | None:
    if not root.exists():
        return None

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if should_skip(p):
            continue
        if p.suffix.lower() not in VIDEO_EXTS:
            continue
        if p.stem == video_id or video_id in p.name:
            return p

    return None


def ffmpeg_trim(src: Path, dst: Path, start: float, end: float, reencode: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and dst.stat().st_size > 0:
        print(f"[SKIP] exists: {dst}")
        return

    duration = end - start

    if reencode:
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", str(src),
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "18",
            "-c:a", "aac",
            "-movflags", "+faststart",
            str(dst),
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", str(src),
            "-t", str(duration),
            "-c", "copy",
            str(dst),
        ]

    run(cmd)


def copy_or_link(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and dst.stat().st_size > 0:
        return "exists"

    if mode == "copy":
        shutil.copy2(src, dst)
        return "copied"

    if mode == "hardlink":
        try:
            os.link(src, dst)
            return "hardlinked"
        except OSError:
            shutil.copy2(src, dst)
            return "copied_fallback"

    if mode == "symlink":
        dst.symlink_to(src.resolve())
        return "symlinked"

    raise ValueError(mode)


def run_ego4d_probe(
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
    print("[PROBE]", " ".join(cmd))

    return subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


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


def classify_uids_across_ego4d_datasets(
    all_uids: set[str],
    datasets_to_try: list[str],
    work_dir: Path,
    ego4d_output_dir: Path,
    version: str,
    aws_profile: str,
) -> dict[str, Any]:
    ego4d_exe = find_ego4d_exe()

    routing: dict[str, set[str]] = {d: set() for d in datasets_to_try}
    remaining = set(all_uids)

    work_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in datasets_to_try:
        if not remaining:
            break

        uid_file = work_dir / f"probe_{dataset_name}.txt"
        write_uid_file(uid_file, remaining)

        proc = run_ego4d_probe(
            ego4d_exe=ego4d_exe,
            output_dir=ego4d_output_dir,
            uid_file=uid_file,
            dataset_name=dataset_name,
            version=version,
            aws_profile=aws_profile,
        )

        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")

        if proc.returncode == 0:
            print(f"[OK] All remaining {len(remaining)} UIDs are present in {dataset_name}")
            routing[dataset_name].update(remaining)
            remaining.clear()
            break

        missing = parse_missing_uids_from_ego4d_output(combined)

        if missing is None:
            print(f"[WARN] Could not parse missing UID set for dataset={dataset_name}")
            print("[STDOUT tail]")
            print((proc.stdout or "")[-3000:])
            print("[STDERR tail]")
            print((proc.stderr or "")[-3000:])
            continue

        found = remaining - missing

        print(f"[INFO] Dataset {dataset_name}: found={len(found)} missing={len(missing)}")

        routing[dataset_name].update(found)
        remaining = missing

    return {
        "routing": {k: sorted(v) for k, v in routing.items() if v},
        "missing": sorted(remaining),
    }


def download_routed_ego4d_uids(
    routing: dict[str, list[str]],
    work_dir: Path,
    ego4d_output_dir: Path,
    version: str,
    aws_profile: str,
    dry_run: bool,
) -> None:
    ego4d_exe = find_ego4d_exe()

    for dataset_name, uids in routing.items():
        if not uids:
            continue

        uid_file = work_dir / f"download_{dataset_name}.txt"
        write_uid_file(uid_file, set(uids))

        cmd = [
            ego4d_exe,
            "--output_directory",
            str(ego4d_output_dir),
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
        print(f"[DOWNLOAD] Ego4D {dataset_name}: {len(uids)} UIDs")
        print("[CMD]", " ".join(cmd))

        if not dry_run:
            subprocess.run(cmd, check=True)


def auto_download_ego4d_uids(
    uids: set[str],
    download_root: Path,
    version: str,
    aws_profile: str,
    datasets_to_try: list[str],
    dry_run: bool,
) -> tuple[Path, dict[str, Any]]:
    ego4d_output_dir = download_root / "ego4d_raw"
    work_dir = download_root / "_tmp" / "ego4d_auto"

    print(f"[INFO] Ego4D-like UID count: {len(uids)}")
    print(f"[INFO] Ego4D datasets to try: {datasets_to_try}")

    if not uids:
        return ego4d_output_dir, {"routing": {}, "missing": []}

    write_uid_file(download_root / "ego4d_all_base_uids.txt", uids)

    result = classify_uids_across_ego4d_datasets(
        all_uids=uids,
        datasets_to_try=datasets_to_try,
        work_dir=work_dir,
        ego4d_output_dir=ego4d_output_dir,
        version=version,
        aws_profile=aws_profile,
    )

    routing = result["routing"]
    missing = result["missing"]

    report = {
        "total_unique_base_uids": len(uids),
        "datasets_to_try": datasets_to_try,
        "routing_counts": {k: len(v) for k, v in routing.items()},
        "missing_count": len(missing),
        "routing": routing,
        "missing": missing,
    }

    write_json(download_root / "ego4d_uid_routing.json", report)
    write_uid_file(download_root / "ego4d_missing_uids.txt", set(missing))

    for dataset_name, found_uids in routing.items():
        write_uid_file(download_root / f"ego4d_found_{dataset_name}.txt", set(found_uids))

    print()
    print("[EGO4D ROUTING REPORT]")
    print(f"  total:   {len(uids)}")
    for dataset_name, found_uids in routing.items():
        print(f"  {dataset_name}: {len(found_uids)}")
    print(f"  missing: {len(missing)}")
    print(f"  wrote:   {download_root / 'ego4d_uid_routing.json'}")

    download_routed_ego4d_uids(
        routing=routing,
        work_dir=work_dir,
        ego4d_output_dir=ego4d_output_dir,
        version=version,
        aws_profile=aws_profile,
        dry_run=dry_run,
    )

    return ego4d_output_dir, report


def materialize_ego4d_like(
    annotations: list[dict[str, Any]],
    ego4d_raw_root: Path,
    output_root: Path,
    mode: str,
    reencode: bool,
    dry_run: bool,
    manifest: list[dict[str, Any]],
    missing: list[dict[str, Any]],
) -> None:
    search_roots = [
        ego4d_raw_root,
        ego4d_raw_root / "v2",
        ego4d_raw_root / "v2" / "clips",
        ego4d_raw_root / "v2" / "full_scale",
        ego4d_raw_root / "v2" / "video_540ss",
        ego4d_raw_root / "v1",
        ego4d_raw_root / "v1" / "clips",
        ego4d_raw_root / "v1" / "full_scale",
        ego4d_raw_root / "v1" / "video_540ss",
    ]

    already_done = set()

    for ann in annotations:
        ds = dataset_norm(ann.get("dataset", ""))
        if ds not in EGO4D_LIKE_DATASETS:
            continue

        vid = ann.get("video_id")
        if not isinstance(vid, str):
            continue

        trim = parse_trim(vid)

        if trim:
            base, start, end = trim
            key = (ds, vid, start, end)
            if key in already_done:
                continue
            already_done.add(key)

            src = None
            for root in search_roots:
                src = find_file_by_stem_or_contains(root, base)
                if src:
                    break

            dst = output_root / ds / "clips" / f"{safe_name(vid)}.mp4"

            if src is None:
                print(f"[MISS] Ego4D-like source for trim not found: {vid}")
                missing.append({
                    "dataset": ds,
                    "video_id": vid,
                    "base": base,
                    "reason": "ego4d_source_for_trim_not_found",
                })
                continue

            if dry_run:
                print(f"[DRY] trim Ego4D-like: {src} [{start}, {end}] -> {dst}")
            else:
                ffmpeg_trim(src, dst, start, end, reencode=reencode)
                manifest.append({
                    "dataset": ds,
                    "type": "trim",
                    "source": str(src),
                    "target": str(dst.relative_to(output_root)),
                    "video_id": vid,
                    "start": start,
                    "end": end,
                })

        else:
            key = (ds, vid)
            if key in already_done:
                continue
            already_done.add(key)

            src = None
            for root in search_roots:
                src = find_file_by_stem_or_contains(root, vid)
                if src:
                    break

            dst = output_root / ds / "videos" / f"{safe_name(vid)}.mp4"

            if src is None:
                print(f"[MISS] Ego4D-like video not found: {vid}")
                missing.append({
                    "dataset": ds,
                    "video_id": vid,
                    "reason": "ego4d_video_not_found",
                })
                continue

            if dry_run:
                print(f"[DRY] link/copy Ego4D-like: {src} -> {dst}")
            else:
                action = copy_or_link(src, dst, mode)
                manifest.append({
                    "dataset": ds,
                    "type": "video",
                    "source": str(src),
                    "target": str(dst.relative_to(output_root)),
                    "video_id": vid,
                    "action": action,
                })


def hf_download_egolife(
    targets: set[tuple[str, str]],
    download_root: Path,
    dry_run: bool,
) -> Path:
    hf_root = download_root / "egolife_hf"
    allow_patterns: list[str] = []

    for day, clip_id in sorted(targets):
        allow_patterns.extend([
            f"**/{day}/**/*{clip_id}*.mp4",
            f"**/*{day}*{clip_id}*.mp4",
            f"**/{day}/**/*{clip_id}*.metadata",
        ])

    allow_patterns = sorted(set(allow_patterns))

    print(f"[INFO] EgoLife targets: {len(targets)}")
    print(f"[INFO] EgoLife HF allow_patterns: {len(allow_patterns)}")

    if dry_run:
        for p in allow_patterns[:50]:
            print("[DRY] HF allow:", p)
        if len(allow_patterns) > 50:
            print(f"[DRY] ... {len(allow_patterns) - 50} more patterns")
        return hf_root

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="lmms-lab/EgoLife",
        repo_type="dataset",
        local_dir=str(hf_root),
        allow_patterns=allow_patterns,
        resume_download=True,
    )

    return hf_root


def materialize_egolife(
    annotations: list[dict[str, Any]],
    hf_root: Path,
    output_root: Path,
    mode: str,
    dry_run: bool,
    manifest: list[dict[str, Any]],
    missing: list[dict[str, Any]],
) -> None:
    used = set()

    for ann in annotations:
        ds = dataset_norm(ann.get("dataset", ""))
        if ds != "EgoLife":
            continue

        vid = ann.get("video_id")
        if not isinstance(vid, dict):
            continue

        for recall_name, payload in vid.items():
            if not isinstance(payload, dict):
                continue

            day = payload.get("day")
            clip_id = payload.get("video_id")

            if not day or not clip_id:
                continue

            key = (str(day), str(clip_id))
            if key in used:
                continue
            used.add(key)

            src = None
            for p in hf_root.rglob("*.mp4"):
                if str(day) in p.name and str(clip_id) in p.name:
                    src = p
                    break
                if str(day) in str(p) and str(clip_id) in p.name:
                    src = p
                    break

            if src is None:
                print(f"[MISS] EgoLife file not found after HF download: {day}/{clip_id}")
                missing.append({
                    "dataset": "EgoLife",
                    "day": day,
                    "video_id": clip_id,
                    "reason": "egolife_file_not_found_after_hf_download",
                })
                continue

            dst = output_root / "EgoLife" / str(day) / src.name

            if dry_run:
                print(f"[DRY] link/copy EgoLife: {src} -> {dst}")
            else:
                action = copy_or_link(src, dst, mode)
                manifest.append({
                    "dataset": "EgoLife",
                    "type": "video",
                    "source": str(src),
                    "target": str(dst.relative_to(output_root)),
                    "day": day,
                    "video_id": clip_id,
                    "action": action,
                })


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
    queries = [
        f"HD-EPIC/Videos/{participant}",
        f"{base_video_id}.mp4",
        f"HD-EPIC {participant}",
    ]

    wanted = f"{base_video_id}.mp4"

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


def download_file_stream(url: str, dst: Path, sleep: float = 0.0) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and dst.stat().st_size > 0:
        print(f"[SKIP] exists: {dst}")
        return

    tmp = dst.with_suffix(dst.suffix + ".part")

    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with tmp.open("wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=dst.name,
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))

    tmp.rename(dst)

    if sleep:
        time.sleep(sleep)


def download_and_trim_hd_epic(
    targets: dict[str, list[tuple[str, float, float]]],
    download_root: Path,
    output_root: Path,
    reencode: bool,
    dry_run: bool,
    manifest: list[dict[str, Any]],
    missing: list[dict[str, Any]],
) -> None:
    raw_dir = download_root / "hd_epic_raw"

    for base_id, clips in sorted(targets.items()):
        raw_mp4 = raw_dir / f"{base_id}.mp4"

        if not raw_mp4.exists():
            url = ckan_find_hd_epic_url(base_id)

            if url is None:
                print(f"[MISS] HD-EPIC URL not found for {base_id}")
                missing.append({
                    "dataset": "HD-EPIC",
                    "video_id": base_id,
                    "reason": "hd_epic_ckan_url_not_found",
                })
                continue

            if dry_run:
                print(f"[DRY] download HD-EPIC: {url} -> {raw_mp4}")
            else:
                print(f"[INFO] HD-EPIC URL {base_id}: {url}")
                download_file_stream(url, raw_mp4)

        for full_id, start, end in clips:
            dst = output_root / "HD-EPIC" / "clips" / f"{safe_name(full_id)}.mp4"

            if dry_run:
                print(f"[DRY] trim HD-EPIC: {raw_mp4} [{start}, {end}] -> {dst}")
            else:
                if not raw_mp4.exists():
                    missing.append({
                        "dataset": "HD-EPIC",
                        "video_id": full_id,
                        "base": base_id,
                        "reason": "hd_epic_raw_missing_after_download",
                    })
                    continue

                ffmpeg_trim(raw_mp4, dst, start, end, reencode=reencode)
                manifest.append({
                    "dataset": "HD-EPIC",
                    "type": "trim",
                    "source": str(raw_mp4),
                    "target": str(dst.relative_to(output_root)),
                    "video_id": full_id,
                    "base": base_id,
                    "start": start,
                    "end": end,
                })


def write_annotation_subsets(annotations: list[dict[str, Any]], output_root: Path) -> None:
    write_json(output_root / "all_annotations.json", annotations)

    grouped = defaultdict(list)
    for ann in annotations:
        ds = dataset_norm(ann.get("dataset", "UNKNOWN"))
        grouped[ds].append(ann)

    for ds, items in grouped.items():
        write_json(output_root / ds / "annotations" / "annotations.json", items)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)

    parser.add_argument("--mode", choices=["copy", "hardlink", "symlink"], default="hardlink")
    parser.add_argument("--reencode-trim", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--skip-egolife", action="store_true")
    parser.add_argument("--skip-ego4d", action="store_true")
    parser.add_argument("--skip-hd-epic", action="store_true")

    parser.add_argument("--ego4d-version", default="v2")
    parser.add_argument("--aws-profile", default="ego4d")
    parser.add_argument(
        "--ego4d-datasets-to-try",
        nargs="+",
        default=DEFAULT_EGO4D_DATASETS_TO_TRY,
        help="Ego4D dataset manifests to try automatically. Default: clips full_scale video_540ss",
    )

    args = parser.parse_args()

    annotations_path = args.annotations.expanduser().resolve()
    download_root = args.download_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    annotations = load_json(annotations_path)
    if not isinstance(annotations, list):
        raise ValueError("Annotations JSON must be a list")

    download_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    if not args.dry_run:
        write_annotation_subsets(annotations, output_root)

    if not args.skip_ego4d:
        ego4d_uids = collect_ego4d_like_uids(annotations)

        ego4d_raw_root, ego4d_report = auto_download_ego4d_uids(
            uids=ego4d_uids,
            download_root=download_root,
            version=args.ego4d_version,
            aws_profile=args.aws_profile,
            datasets_to_try=args.ego4d_datasets_to_try,
            dry_run=args.dry_run,
        )

        for uid in ego4d_report.get("missing", []):
            missing.append({
                "dataset": "Ego4D-like",
                "video_id": uid,
                "reason": "uid_not_found_in_any_ego4d_manifest",
            })

        materialize_ego4d_like(
            annotations=annotations,
            ego4d_raw_root=ego4d_raw_root,
            output_root=output_root,
            mode=args.mode,
            reencode=args.reencode_trim,
            dry_run=args.dry_run,
            manifest=manifest,
            missing=missing,
        )

    if not args.skip_egolife:
        egolife_targets = collect_egolife_targets(annotations)

        hf_root = hf_download_egolife(
            targets=egolife_targets,
            download_root=download_root,
            dry_run=args.dry_run,
        )

        materialize_egolife(
            annotations=annotations,
            hf_root=hf_root,
            output_root=output_root,
            mode=args.mode,
            dry_run=args.dry_run,
            manifest=manifest,
            missing=missing,
        )

    if not args.skip_hd_epic:
        hd_targets = collect_hd_epic_targets(annotations)

        print(f"[INFO] HD-EPIC base videos: {len(hd_targets)}")
        print(f"[INFO] HD-EPIC clips: {sum(len(v) for v in hd_targets.values())}")

        download_and_trim_hd_epic(
            targets=hd_targets,
            download_root=download_root,
            output_root=output_root,
            reencode=args.reencode_trim,
            dry_run=args.dry_run,
            manifest=manifest,
            missing=missing,
        )

    if not args.dry_run:
        write_json(output_root / "manifest.json", manifest)
        write_json(output_root / "missing_files.json", missing)

        readme = output_root / "README.md"
        readme.write_text(
            """# Anonymous Review Dataset

This release was generated from the annotation file.

## Sources

- EgoLife files are downloaded from Hugging Face.
- Ego4D-like files are downloaded through the Ego4D CLI by automatically checking multiple Ego4D manifests.
- HD-EPIC videos are downloaded from University of Bristol data.bris and trimmed according to annotation IDs.

## Ego4D routing

The script automatically checks Ego4D-like video UIDs against:

- clips
- full_scale
- video_540ss

It writes routing information under the download root:

- ego4d_uid_routing.json
- ego4d_all_base_uids.txt
- ego4d_missing_uids.txt
- ego4d_found_<dataset>.txt

## Trim convention

A video id like:

`P08-20240618-171546_1130.87_1184.533`

is interpreted as:

- source video: `P08-20240618-171546.mp4`
- start: `1130.87`
- end: `1184.533`

Negative start times are clamped to `0.0`.
""",
            encoding="utf-8",
        )

    print()
    print(f"[DONE] output: {output_root}")
    print(f"[DONE] manifest items: {len(manifest)}")
    print(f"[DONE] missing items: {len(missing)}")

    if missing:
        print(f"[CHECK] missing file report: {output_root / 'missing_files.json'}")


if __name__ == "__main__":
    main()