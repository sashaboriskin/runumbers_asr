"""One-time preprocessing: decode + resample all train/dev audio to 16 kHz mono
float16 and dump as .npy files. Subsequent training reads these via mmap, which
is ~10x faster than decode+resample in each DataLoader worker.

    uv run python scripts/cache_audio.py --data-root data --cache-dir data/cache_16k --workers 16
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# lazy: import heavy deps inside worker for cleaner fork
def _preprocess_one(args):
    src_path, out_path, target_sr = args
    if out_path.exists():
        return True
    # import inside worker
    import soundfile as sf
    import torchaudio.functional as AF
    import torch
    try:
        data, sr = sf.read(str(src_path), dtype="float32", always_2d=False)
    except Exception:
        import librosa
        data, sr = librosa.load(str(src_path), sr=None, mono=True)
        data = np.asarray(data, dtype=np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        t = torch.from_numpy(np.ascontiguousarray(data))
        t = AF.resample(t, sr, target_sr)
        data = t.numpy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Save as float16 to halve file size; precision loss is negligible for ASR.
    np.save(str(out_path), data.astype(np.float16), allow_pickle=False)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--cache-dir", type=Path, default=Path("data/cache_16k"))
    ap.add_argument("--target-sr", type=int, default=16_000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--splits", nargs="+", default=["train", "dev"])
    args = ap.parse_args()

    tasks: list[tuple[Path, Path, int]] = []
    for split in args.splits:
        csv = args.data_root / split / f"{split}.csv"
        df = pd.read_csv(csv)
        for fn in df["filename"].tolist():
            src = args.data_root / split / fn
            base = Path(fn).name.rsplit(".", 1)[0] + ".npy"
            dst = args.cache_dir / split / base
            tasks.append((src, dst, args.target_sr))
    print(f"[cache] {len(tasks)} files across splits={args.splits}, workers={args.workers}")

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_preprocess_one, t) for t in tasks]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="cache"):
            pass
    print(f"[cache] done. dir={args.cache_dir}")


if __name__ == "__main__":
    main()
