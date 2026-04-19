"""Inference / decoding for ConformerCTC model.

Usage (local):
    uv run python scripts/infer.py --ckpt runs/base/ckpts/best.pt \
        --csv data/dev/dev.csv --audio-root data/dev --out submission.csv

For Kaggle, the notebook does essentially the same but with kaggle paths.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .beam_search import decode_beams
from .data import MelExtractor, DataConfig, _load_audio
from .model import ConformerCTC, ModelConfig
from .text import ID2CHAR, BLANK_ID, ctc_greedy_decode, words_to_digits_safe


@dataclass
class InferConfig:
    ckpt: Path
    csv: Path
    audio_root: Path
    out: Path
    batch_size: int = 8
    device: str = "cuda"
    fallback: int = 100000  # reasonable mid-range number if decoding fails
    beam_width: int = 1     # 1 = greedy, >1 = beam search


def load_model(ckpt_path: Path, device: torch.device) -> tuple[ConformerCTC, ModelConfig]:
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    mcfg = ModelConfig(**blob["cfg_model"])
    model = ConformerCTC(mcfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, mcfg


@torch.no_grad()
def decode_file(
    model: ConformerCTC,
    mel_ext: MelExtractor,
    path: Path,
    device: torch.device,
    fallback: int,
    beam_width: int = 1,
) -> int:
    wav = _load_audio(path, target_sr=16_000).to(device)
    if wav.numel() < 400:
        return fallback
    mel = mel_ext(wav.unsqueeze(0))  # [1, n_mels, T]
    mel_lens = torch.tensor([mel.size(-1)], device=device)
    logits, out_lens = model(mel, mel_lens)
    L = int(out_lens[0])

    if beam_width > 1:
        log_probs = F.log_softmax(logits[0, :L].float(), dim=-1).cpu().numpy()
        n, _ = decode_beams(log_probs, beam_width=beam_width, fallback=fallback)
        return n

    pred = logits.argmax(-1)[0, :L].cpu().tolist()
    hyp = ctc_greedy_decode(pred)
    return words_to_digits_safe(hyp, fallback=fallback)


def run_on_csv(cfg: InferConfig) -> pd.DataFrame:
    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model, mcfg = load_model(cfg.ckpt, device)
    # mel extractor must match training: use same DataConfig defaults
    mel_ext = MelExtractor(DataConfig(data_root=Path("."))).to(device)

    df = pd.read_csv(cfg.csv)
    preds: list[int] = []
    for fn in tqdm(df["filename"].tolist(), desc="infer"):
        p = cfg.audio_root / fn
        try:
            digit = decode_file(model, mel_ext, p, device, cfg.fallback, cfg.beam_width)
        except Exception:
            digit = cfg.fallback
        # clamp into valid range
        digit = int(max(1000, min(999_999, digit)))
        preds.append(digit)
    df["transcription"] = preds
    out = df[["filename", "transcription"]]
    out.to_csv(cfg.out, index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--audio-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fallback", type=int, default=100000)
    ap.add_argument("--beam-width", type=int, default=1)
    args = ap.parse_args()
    run_on_csv(InferConfig(**vars(args)))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
