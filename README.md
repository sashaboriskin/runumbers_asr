# runumbers_asr

Russian spoken-numbers ASR for the Kaggle [ASR-2026 Spoken Numbers Recognition Challenge](https://www.kaggle.com/competitions/asr-2026-spoken-numbers-recognition-challenge/overview).

Team: `kuzya_lakomkin` — Boriskin Aleksandr, Fedor Avilov.

**Best public LB score: 1.718** (harmonic mean CER).

## Approach

- **Target text:** `digits_to_words(int)` via `num2words(lang='ru')`. A character-level CTC model emits Russian number words; a closed-vocabulary parser decodes them back to digits.
- **Model:** compact Conformer-CTC (Conv2D subsampling 4× → MHSA + depthwise conv + macaron FFN → linear head over 34 chars).

| Config | d_model | n_blocks | n_heads | Parameters |
|--------|---------|----------|---------|------------|
| v1 (`configs/base.yaml`) | 144 | 8 | 4 | 3.97M |
| **v2** (`configs/v2.yaml`) | **160** | **8** | **4** | **4.89M** |

- **Input:** log-mel (80 bins, 16 kHz, 25 ms / 10 ms).
- **Augmentations (v2):** speed perturb (0.9/1.0/1.1), gain jitter (±6 dB), MUSAN additive noise (SNR 5–25 dB), SpecAugment (2× freq masks of 27, 2× time masks of 40).
- **Decoding:** CTC beam search (width=20) with number-constrained hypothesis selection and word-level Levenshtein correction.
- **Sampler:** speaker-balanced `WeightedRandomSampler` (spk_E is 45% of raw train).
- **Label filter:** drop train rows with `transcription < 1000` (12 outliers).

## Results

| Experiment | Model | Decoding | inD CER | ooD CER | Hmean (dev) | Public LB |
|:---|:---|:---|---:|---:|---:|---:|
| v1 | 3.97M | greedy | 1.947% | 15.691% | 3.464% | 3.861 |
| v1 + beam | 3.97M | beam=20 | 1.422% | 12.271% | 2.549% | 2.172 |
| **v2 + beam** | **4.89M** | **beam=20** | **0.858%** | **4.952%** | **1.463%** | **1.718** |

## Layout
```
src/asr/        library code (text, data, model, train, infer, beam_search)
scripts/        runnable entrypoints + smoke + EDA
configs/        YAML configs (base.yaml = v1, v2.yaml = v2, smoke.yaml = local)
data/           competition data (CSVs tracked; audio git-ignored)
notebooks/      Kaggle submission notebook
runs/           TensorBoard logs + checkpoints (git-ignored)
reports/        EDA outputs, report
```

## Setup
```bash
uv sync
uv run python scripts/test_text.py       # text module round-trip tests
uv run python scripts/smoke_pipeline.py  # end-to-end pipeline sanity-check
```

## EDA
```bash
uv run python scripts/eda.py
uv run python scripts/investigate_outliers.py
```
Key findings are in `reports/eda/summary.json`.

## Training (H100)
```bash
# pre-cache audio to 16 kHz .npy (one-time, ~2 min)
uv run python scripts/cache_audio.py --workers 32

# optional: download MUSAN noise for augmentation (~10 GB)
bash scripts/prepare_musan_noise.sh
# then set  noise_dir: data/noise/flat  in the config

# launch training
uv run python scripts/train.py --config configs/v2.yaml

# monitor
uv run tensorboard --logdir runs
```

## Evaluation
```bash
uv run python scripts/evaluate_dev.py --ckpt runs/v2/ckpts/best.pt --beam-width 20
```
Prints per-speaker digit CER and the harmonic mean of inD / ooD CER.

## Kaggle submission
1. Push this repo to GitHub as a public repo.
2. Create a release and attach `runs/v2/ckpts/best.pt` as an asset.
3. Open `notebooks/kaggle_submission.ipynb` on Kaggle, update `WEIGHTS_URL` if needed, and "Save & Run All (Commit)" → Submit.

## Constraints (from the task)
- Model ≤ 5M parameters, trained **from scratch** (no pretrained weights).
- Input: 16 kHz mono audio.
- Output: integer in `[1000, 999999]`.
- Metric: harmonic mean of CER over in-domain and out-of-domain speakers.
