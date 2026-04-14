# runumbers_asr

Russian spoken-numbers ASR for the Kaggle [ASR-2026 Spoken Numbers Recognition Challenge](https://www.kaggle.com/competitions/asr-2026-spoken-numbers-recognition-challenge/overview).

Team: `kuzya_lakomkin`.

## Approach (v1)
- **Target text:** `digits_to_words(int)` via `num2words(lang='ru')`. Train a character-level CTC model to emit Russian number words; decode back to digits with a closed-vocabulary parser. Hint from the task about "тысяча" is handled naturally by parsing.
- **Model:** compact Conformer-CTC. Default `d_model=144`, `n_blocks=8`, `n_heads=4` → **3.97M params** (< 5M). Conv subsampling 4× → MHSA + depth-wise conv + macaron FFN → linear head over 34 chars (32 cyrillic + space + CTC blank).
- **Input:** log-mel (80 bins, 16 kHz, 25 ms / 10 ms), audio resampled to 16 kHz at load-time.
- **Augmentations:** speed perturb (0.9/1.0/1.1), gain jitter (±6 dB), additive noise from MUSAN with SNR 5–25 dB, SpecAugment (2× freq masks of 20, 2× time masks of 30).
- **Sampler:** speaker-balanced `WeightedRandomSampler` (spk_E is 45% of raw train).
- **Label filter:** drop train rows with `transcription < 1000` (12 outliers).

## Layout
```
src/asr/        library code (text, data, model, train, infer)
scripts/        runnable entrypoints + smoke + EDA
configs/        YAML configs (base.yaml = H100; smoke.yaml = local)
data/           competition data (CSVs tracked; audio ignored)
notebooks/      Kaggle submission notebook
runs/           TensorBoard logs (git-ignored)
reports/        EDA outputs, report drafts
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
# optional: download MUSAN noise for augmentation (~10 GB)
bash scripts/prepare_musan_noise.sh
# then set  data.noise_dir: data/noise/flat  in configs/base.yaml

# launch training with TensorBoard logs under runs/base/tb
uv run python scripts/train.py --config configs/base.yaml

# watch:
uv run tensorboard --logdir runs
```

## Evaluation
```bash
uv run python scripts/evaluate_dev.py --ckpt runs/base/ckpts/best.pt
```
Prints per-speaker digit CER and the harmonic mean of inD / ooD CER
(the competition metric is computed the same way on test).

## Kaggle submission
1. Push this repo to GitHub as a public repo (e.g. `sashaboriskin/runumbers_asr`).
2. Create a release (e.g. `v0.1.0`) and attach `runs/base/ckpts/best.pt` as an asset.
3. Open `notebooks/kaggle_submission.ipynb` on Kaggle, update `REPO_URL`/`WEIGHTS_URL` if needed, and "Save & Run All (Commit)" → Submit.

## Constraints (from the task)
- Model ≤ 5M parameters, trained **from scratch** (no pretrained weights).
- Input: 16 kHz mono audio.
- Output: integer in `[1000, 999999]`.
- Metric: harmonic mean of CER over in-domain and out-of-domain speakers.
