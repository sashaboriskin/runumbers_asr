"""EDA for the ASR-2026 Russian spoken numbers dataset.

Writes a summary to reports/eda.md and a few plots to reports/eda/.
Run locally on Mac:
    uv run python scripts/eda.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports" / "eda"
REPORTS.mkdir(parents=True, exist_ok=True)


def audio_duration(path: Path) -> tuple[float, int]:
    info = sf.info(str(path))
    return info.duration, info.samplerate


def scan_split(split: str) -> pd.DataFrame:
    csv_path = DATA / split / f"{split}.csv"
    df = pd.read_csv(csv_path)
    durations, srs = [], []
    for fn in tqdm(df["filename"].tolist(), desc=f"scan {split}"):
        # filename in csv is like "train/xxxx.wav" — prefix with split dir
        p = DATA / split / fn
        try:
            d, sr = audio_duration(p)
        except Exception:
            d, sr = np.nan, -1
        durations.append(d)
        srs.append(sr)
    df["duration_s"] = durations
    df["sr_actual"] = srs
    df["transcription_len"] = df["transcription"].astype(str).str.len()
    return df


def summarize(df: pd.DataFrame, split: str) -> dict:
    out = {
        "split": split,
        "n": int(len(df)),
        "speakers": sorted(df["spk_id"].unique().tolist()),
        "n_speakers": int(df["spk_id"].nunique()),
        "per_speaker_count": df["spk_id"].value_counts().sort_index().to_dict(),
        "per_gender_count": df["gender"].value_counts().to_dict(),
        "ext_count": df["ext"].value_counts().to_dict(),
        "sr_count": df["samplerate"].value_counts().to_dict(),
        "sr_actual_count": df["sr_actual"].value_counts().to_dict(),
        "duration_s": {
            "min": float(np.nanmin(df["duration_s"])),
            "max": float(np.nanmax(df["duration_s"])),
            "mean": float(np.nanmean(df["duration_s"])),
            "p50": float(np.nanpercentile(df["duration_s"], 50)),
            "p95": float(np.nanpercentile(df["duration_s"], 95)),
            "p99": float(np.nanpercentile(df["duration_s"], 99)),
        },
        "transcription_len_count": df["transcription_len"].value_counts().sort_index().to_dict(),
        "transcription_min": int(df["transcription"].min()),
        "transcription_max": int(df["transcription"].max()),
    }
    return out


def main() -> None:
    results = {}
    for split in ("train", "dev"):
        df = scan_split(split)
        df.to_csv(REPORTS / f"{split}_scan.csv", index=False)
        results[split] = summarize(df, split)
    (REPORTS / "summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
