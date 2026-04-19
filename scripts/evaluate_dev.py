"""Run full dev evaluation with a trained checkpoint, print per-speaker CER
and the Kaggle-style harmonic mean on dev (inD = A-F seen in train, ooD = H/I/J/K).

    uv run python scripts/evaluate_dev.py --ckpt runs/base/ckpts/best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from jiwer import cer
from tqdm import tqdm

from asr.beam_search import decode_beams
from asr.data import DataConfig, MelExtractor, _load_audio
from asr.infer import load_model
from asr.text import ctc_greedy_decode, digits_to_words, words_to_digits_safe


IND_SPK = {"spk_A", "spk_B", "spk_C", "spk_D", "spk_E", "spk_F"}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--data-root", default=Path("data"), type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--beam-width", type=int, default=1, help="1=greedy, >1=beam search")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model, _ = load_model(args.ckpt, device)
    mel_ext = MelExtractor(DataConfig(data_root=args.data_root)).to(device)

    df = pd.read_csv(args.data_root / "dev" / "dev.csv")
    beam_width = args.beam_width
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        path = args.data_root / "dev" / row["filename"]
        wav = _load_audio(path, 16_000).to(device)
        mel = mel_ext(wav.unsqueeze(0))
        mel_lens = torch.tensor([mel.size(-1)], device=device)
        logits, out_lens = model(mel, mel_lens)
        L = int(out_lens[0])

        if beam_width > 1:
            log_probs = F.log_softmax(logits[0, :L].float(), dim=-1).cpu().numpy()
            hyp_digits, hyp_text = decode_beams(log_probs, beam_width=beam_width)
        else:
            pred = logits.argmax(-1)[0, :L].cpu().tolist()
            hyp_text = ctc_greedy_decode(pred)
            hyp_digits = words_to_digits_safe(hyp_text, fallback=100_000)
        hyp_digits = int(max(1000, min(999_999, hyp_digits)))
        ref_digits = int(row["transcription"])
        ref_text = digits_to_words(ref_digits) if ref_digits >= 0 else ""
        results.append({
            "filename": row["filename"],
            "spk_id": row["spk_id"],
            "ref_digits": ref_digits,
            "hyp_digits": hyp_digits,
            "hyp_text": hyp_text,
            "text_cer": cer(ref_text, hyp_text) if ref_text else float("nan"),
            "digit_cer": cer(str(ref_digits), str(hyp_digits)),
        })
    rdf = pd.DataFrame(results)
    print("\n=== per-speaker digit CER ===")
    print(rdf.groupby("spk_id")["digit_cer"].agg(["mean", "count"]).to_string())
    ind_mask = rdf["spk_id"].isin(IND_SPK)
    ind_cer = rdf.loc[ind_mask, "digit_cer"].mean()
    ood_cer = rdf.loc[~ind_mask, "digit_cer"].mean()
    hmean = 2 * ind_cer * ood_cer / (ind_cer + ood_cer + 1e-12) * 100
    print(f"\ninD digit CER: {ind_cer*100:.3f}%")
    print(f"ooD digit CER: {ood_cer*100:.3f}%")
    print(f"harmonic mean: {hmean:.3f}%")
    out_path = args.ckpt.parent / "dev_eval.csv"
    rdf.to_csv(out_path, index=False)
    print(f"wrote per-sample details to {out_path}")


if __name__ == "__main__":
    main()
