# Data

EgoStream does **not** redistribute any video. Each question references a video (or a
trimmed clip) of one of five public egocentric datasets, which you must obtain under
their own licenses and access agreements:

| Source dataset | How the videos are obtained | Where to request access |
|---|---|---|
| Ego4D (Episodic Memory VQA) | Ego4D CLI (AWS credentials) | https://ego4d-data.org |
| EgoTempo | Ego4D CLI, then trimmed to the clip encoded in `video_id` | https://ego4d-data.org |
| Multi-Hop EgoQA | Ego4D CLI, then trimmed to the clip encoded in `video_id` | https://ego4d-data.org |
| HD-EPIC | University of Bristol data endpoint, then trimmed | https://hd-epic.github.io |
| EgoLife | Hugging Face (`lmms-lab/EgoLife`) | https://huggingface.co/datasets/lmms-lab/EgoLife |

The scripts in `data/download/` read the EgoStream annotation file, download only the
videos that are needed, trim them where required and write the final layout expected by
the benchmark scripts (`clip_base_path` in the run configs):

```text
data/
├── annotations/egostream_annotations.json   # released separately (see data/annotations/README.md)
├── download/global_download_and_trim.py     # download + trim everything
├── download/dry_run_check_all_sources.py    # check availability without downloading
├── source_data/                             # raw downloads (git-ignored)
└── trimmed_clips/                           # final clips + all_annotations.json (git-ignored)
```

After the download finishes, point the run configs to the reconstructed data:

```json
"paths": {
  "annotations_file": "data/trimmed_clips/all_annotations.json",
  "clip_base_path": "data/trimmed_clips"
}
```

The rest of this page is the step-by-step setup guide for the download scripts.

---

This section explains how to configure the external dataset access tools and how to run the download/reconstruction scripts on Linux.

The pipeline reconstructs the dataset from the final annotation file:

```text
data/annotations/egostream_annotations.json
```

It handles the sources as follows:

```text
Ego4D / Ego4D_EM_Test:
  downloaded through the Ego4D CLI

EgoTempo:
  downloaded through the Ego4D CLI, then trimmed when video_id encodes start/end

MultiHop-EgoQA:
  downloaded through the Ego4D CLI, then trimmed when video_id encodes start/end

HD-EPIC:
  downloaded from the HD-EPIC endpoint, then trimmed according to video_id

EgoLife:
  downloaded from Hugging Face
```

---

## 1. Python environment

From the project root:

```bash
cd EgoStream
python3 -m venv .venv
source .venv/bin/activate
```

Install the required Python packages:

```bash
python -m pip install -U pip
python -m pip install -U requests tqdm huggingface_hub ego4d boto3 botocore
```

---

## 2. FFmpeg

FFmpeg is required for trimming clips.

Check if FFmpeg is available:

```bash
ffmpeg -version
```

If it is not available, install it:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

Then check again:

```bash
ffmpeg -version
```

---

## 3. AWS CLI setup for Ego4D

The Ego4D CLI uses AWS credentials to access the Ego4D buckets.

Install AWS CLI if needed:

```bash
sudo apt-get update
sudo apt-get install -y awscli
```

Check that it works:

```bash
aws --version
```

Configure an AWS profile named `ego4d`:

```bash
aws configure --profile ego4d
```

When prompted, enter the Ego4D credentials:

```text
AWS Access Key ID [None]: <YOUR_EGO4D_ACCESS_KEY>
AWS Secret Access Key [None]: <YOUR_EGO4D_SECRET_KEY>
Default region name [None]: us-east-1
Default output format [None]: json
```

Verify the profile:

```bash
aws configure list --profile ego4d
```

The credentials are stored locally under:

```text
~/.aws/credentials
~/.aws/config
```

Example `~/.aws/credentials`:

```ini
[ego4d]
aws_access_key_id = AKIA...
aws_secret_access_key = ...
```

Example `~/.aws/config`:

```ini
[profile ego4d]
region = us-east-1
output = json
```

Do not commit AWS keys to git.

---

## 4. Ego4D CLI check

After installing the `ego4d` Python package, check that the CLI is available:

```bash
ego4d --help
```

If the shell cannot find it, try the virtual environment executable directly:

```bash
./.venv/bin/ego4d --help
```

The scripts automatically try to locate `ego4d` inside the active virtual environment.

---

## 5. Hugging Face CLI setup for EgoLife

EgoLife is downloaded from Hugging Face.

Install the Hugging Face Hub package:

```bash
python -m pip install -U huggingface_hub
```

Login with a Hugging Face token:

```bash
huggingface-cli login
```

Paste your Hugging Face access token when prompted.

To create a token:

```text
Hugging Face account → Settings → Access Tokens → New token
```

Use a token with permission to read the EgoLife dataset.

Check login:

```bash
huggingface-cli whoami
```

If the EgoLife repository is gated or private, make sure your account has access before running the script.

---

## 6. Dry-run: check whether files exist

Before downloading, run the dry-run checker.

This checks whether the required source videos/clips can be found, without downloading the full videos.

Recommended folders:

```text
data/dry_run_reports/
```

### Check only Ego4D-source datasets

This checks Ego4D, EgoTempo, and MultiHop-EgoQA against the Ego4D manifests:

```bash
python data/download/dry_run_check_all_sources.py \
  --annotations data/annotations/egostream_annotations.json \
  --report-root data/dry_run_reports \
  --aws-profile ego4d \
  --skip-hd-epic \
  --skip-egolife
```

The checker automatically tests the Ego4D manifests:

```text
clips
full_scale
video_540ss
```

Output files:

```text
data/dry_run_reports/dry_run_summary.json
data/dry_run_reports/ego4d_report.json
data/dry_run_reports/ego4d_missing_uids.txt
data/dry_run_reports/ego4d_found_clips.txt
data/dry_run_reports/ego4d_found_full_scale.txt
data/dry_run_reports/ego4d_found_video_540ss.txt
```

Inspect the summary:

```bash
cat data/dry_run_reports/dry_run_summary.json
```

Inspect missing Ego4D UIDs:

```bash
head -100 data/dry_run_reports/ego4d_missing_uids.txt
```

If `ego4d_missing_uids.txt` is empty, all Ego4D-source videos were found.

### Full dry-run for all sources

This also checks HD-EPIC and EgoLife remote availability:

```bash
python data/download/dry_run_check_all_sources.py \
  --annotations data/annotations/egostream_annotations.json \
  --report-root data/dry_run_reports \
  --aws-profile ego4d
```

To avoid remote checks for HD-EPIC and EgoLife:

```bash
python data/download/dry_run_check_all_sources.py \
  --annotations data/annotations/egostream_annotations.json \
  --report-root data/dry_run_reports \
  --aws-profile ego4d \
  --skip-hd-epic-remote \
  --skip-egolife-remote
```

---

## 7. Run the full download/reconstruction script

The main script downloads the required sources and creates the final reconstructed dataset.

Recommended folders:

```text
data/source_data/
  raw downloaded source data

data/trimmed_clips/
  final reconstructed dataset with selected videos/clips and annotations
```

Run the full pipeline:

```bash
python data/download/global_download_and_trim.py \
  --annotations data/annotations/egostream_annotations.json \
  --download-root data/source_data \
  --output-root data/trimmed_clips \
  --aws-profile ego4d
```

Run only Ego4D-source datasets:

```bash
python data/download/global_download_and_trim.py \
  --annotations data/annotations/egostream_annotations.json \
  --download-root data/source_data \
  --output-root data/trimmed_clips \
  --aws-profile ego4d \
  --skip-hd-epic \
  --skip-egolife
```

Run only EgoLife:

```bash
python data/download/global_download_and_trim.py \
  --annotations data/annotations/egostream_annotations.json \
  --download-root data/source_data \
  --output-root data/trimmed_clips \
  --skip-ego4d \
  --skip-hd-epic
```

Run only HD-EPIC:

```bash
python data/download/global_download_and_trim.py \
  --annotations data/annotations/egostream_annotations.json \
  --download-root data/source_data \
  --output-root data/trimmed_clips \
  --skip-ego4d \
  --skip-egolife
```

---

## 8. Trimming behavior

For IDs of the form:

```text
<base_video_id>_<start_second>_<end_second>
```

example:

```text
P08-20240618-171546_1130.87_1184.533
```

the script interprets:

```text
base video: P08-20240618-171546
start:      1130.87
end:        1184.533
```

Negative starts are clamped to `0.0`.

By default, trimming uses fast stream copy:

```text
ffmpeg -ss <start> -i <source> -t <duration> -c copy <output>
```

For more accurate but slower trimming, add:

```bash
--reencode-trim
```

Example:

```bash
python data/download/global_download_and_trim.py \
  --annotations data/annotations/egostream_annotations.json \
  --download-root data/source_data \
  --output-root data/trimmed_clips \
  --aws-profile ego4d \
  --reencode-trim
```

---

## 9. Resume behavior

The scripts are designed to be rerunnable.

Existing source folders are reused:

```text
data/source_data/ego4d_raw/
data/source_data/egolife_hf/
data/source_data/hd_epic_raw/
```

Existing final outputs are reused:

```text
data/trimmed_clips/
```

For HD-EPIC and generated clips, files that already exist and have non-zero size are skipped.

For EgoLife, Hugging Face handles resume/caching.

For Ego4D, the Ego4D CLI is called with grouped UID files according to the detected manifest routing. Existing downloads under `source_data/ego4d_raw/` are reused by the CLI where possible.

---

## 10. Final outputs

After a successful run, inspect:

```text
data/trimmed_clips/all_annotations.json
data/trimmed_clips/manifest.json
data/trimmed_clips/missing_files.json
data/trimmed_clips/README.md
```

Check missing files:

```bash
cat data/trimmed_clips/missing_files.json
```

Check generated manifest:

```bash
head -200 data/trimmed_clips/manifest.json
```

Check disk usage:

```bash
du -sh data/source_data
du -sh data/trimmed_clips
```

