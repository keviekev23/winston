# Flywheel Setup Guide

> **Parent:** `docs/phases/phase_a_perception.md`
> **Prerequisite:** Phase A Layer 1-2 running (perception service collecting data)

One-time setup for the cloud fine-tuning pipeline. Takes about 15 minutes.

---

## Overview

```
Mac (local)                      Google Drive                    Google Colab
───────────────                  ─────────────────               ──────────────
data/collection/audio/
  *.wav + *.json
       │
       │  rclone copy
       ▼
                        winston-flywheel/
                          data/audio/       ←── notebook reads from here
                          models/merged/    ──► notebook saves here
                               │
                               │  rclone sync
                               ▼
                        data/adapters/cycle-1/hf/
                               │
                               │  mlx_whisper.convert
                               ▼
                        data/adapters/cycle-1/mlx/  ← perception service loads this
```

---

## Step 1: Install rclone

```bash
brew install rclone
```

---

## Step 2: Configure rclone for Google Drive

```bash
rclone config
```

At the prompt, follow these steps:

```
n) New remote
name> gdrive          ← type exactly "gdrive" (matches config/default.yaml)

Storage> drive        ← type "drive" for Google Drive

client_id>            ← press Enter (leave blank, use rclone's default)
client_secret>        ← press Enter (leave blank)

scope> 1              ← "Full access to all files"

root_folder_id>       ← press Enter (leave blank)
service_account_file> ← press Enter (leave blank)

Edit advanced config? n

Use auto config? y    ← opens browser for OAuth

                      ← log in with your Google account in the browser

Configure as a Shared Drive? n

y) Yes this is OK
```

Verify it works:
```bash
rclone lsd gdrive:
```
You should see your Google Drive folders listed.

---

## Step 3: Create the Drive folder structure

```bash
rclone mkdir gdrive:winston-flywheel/data/audio
rclone mkdir gdrive:winston-flywheel/models/merged
rclone mkdir gdrive:winston-flywheel/models/checkpoints
```

---

## Step 4: Do a dry-run upload to verify everything connects

```bash
python scripts/upload_training_data.py --dry-run
```

Expected output:
```
Winston Flywheel — Upload Training Data
  [DRY RUN — nothing will be uploaded]

  Collection dir:        .../data/collection/audio
  Total recordings:      47
  Low-confidence (<0.7): 12
  High-confidence:       35 total, 7 sampled (20%)
  Uploading:             19 files

  rclone copy → gdrive:winston-flywheel/data/audio

Dry run complete — no files were transferred.
```

If the dry run looks right, run without `--dry-run` to upload for real.

---

## Step 5: Set up Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Upload notebook**
3. Upload `notebooks/whisper_finetune.ipynb` from this repo
4. Click **Runtime → Change runtime type → T4 GPU** (required)
5. Click **Runtime → Run all** — or step through cells manually (recommended first time)

The notebook will:
- Ask for Google Drive permission (first cell — click through the OAuth prompt)
- Install dependencies (~2 min)
- Prompt you to confirm the config cell looks right
- Run for 30–90 minutes depending on dataset size

---

## Step 6: Download the fine-tuned model

After the Colab notebook completes:

```bash
# List what's available on Drive
python scripts/download_adapter.py --list

# Download and convert to MLX (takes ~5 min)
python scripts/download_adapter.py --cycle cycle-1
```

This will:
1. Sync the merged HuggingFace model from Drive to `data/adapters/cycle-1/hf/`
2. Convert it to MLX format at `data/adapters/cycle-1/mlx/`
3. Update `config/default.yaml` → `stt.model` to point to the new model

Restart the perception service to use the updated model:
```bash
python -m src.perception.service
```

---

## Step 7: Measure improvement

```bash
# Run evaluation with the new model
python scripts/evaluate_whisper.py --label cycle-1

# Compare to your baseline
python scripts/evaluate_whisper.py --compare \
    data/benchmark/evals/<timestamp>_base.json \
    data/benchmark/evals/<timestamp>_cycle-1.json
```

---

## Flywheel cycle cadence

| Step | When |
|------|------|
| Collect data | Ongoing — perception service runs automatically |
| Upload | Weekly, or when you have 100+ new utterances |
| Train (Colab) | After each upload batch |
| Download + evaluate | After training |
| Increment CYCLE_NAME | Each training run |

---

## Troubleshooting

**rclone: `gdrive` remote not found**
Re-run `rclone config` and verify the remote name is exactly `gdrive`.

**Colab: No GPU available**
Go to Runtime → Change runtime type → Hardware accelerator → T4 GPU.
If T4 is unavailable, wait and retry (free tier has quotas). Colab Pro removes this issue.

**Colab: Out of memory during training**
Reduce `BATCH_SIZE` from 8 to 4 in the config cell. Also try reducing `LORA_R` from 8 to 4.

**download_adapter.py: mlx_whisper conversion fails**
Make sure mlx-whisper is installed: `pip install mlx-whisper`. The conversion requires MLX,
which only runs on Apple Silicon. Don't run this on an Intel Mac.

**WER doesn't improve after fine-tuning**
Common causes:
- Too few training examples (<50) — collect more data before running again
- Confidence threshold miscalibrated — run `evaluate_whisper.py` to check calibration
- Training overfit (val WER increased while train WER decreased) — reduce `TRAIN_EPOCHS` to 5
