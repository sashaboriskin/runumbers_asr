"""Look at suspicious samples: short-label in train, long-duration in dev."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "eda"


def main() -> None:
    train = pd.read_csv(REPORTS / "train_scan.csv")
    dev = pd.read_csv(REPORTS / "dev_scan.csv")

    print("=== train: short transcriptions (<4 digits) ===")
    short = train[train["transcription_len"] < 4].sort_values("transcription_len")
    print(short.to_string())
    print()
    print("=== train: 4-digit transcriptions (sample) — durations ===")
    four = train[train["transcription_len"] == 4]
    print(four["duration_s"].describe())
    print()
    print("=== dev: long audio (>10s) ===")
    longd = dev[dev["duration_s"] > 10]
    print(longd.to_string())
    print()
    print("=== dev durations by speaker ===")
    print(dev.groupby("spk_id")["duration_s"].describe())
    print()
    print("=== train durations by speaker ===")
    print(train.groupby("spk_id")["duration_s"].describe())


if __name__ == "__main__":
    main()
