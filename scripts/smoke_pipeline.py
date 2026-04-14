"""End-to-end smoke test with 8 train / 8 dev samples and 3 update steps.

Validates: data loading, mel extraction, forward pass, CTC backward, evaluate.
Run:
    uv run python scripts/smoke_pipeline.py
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from asr.data import DataConfig, MelExtractor, NumbersASRDataset, SpecAugment, collate
from asr.model import ConformerCTC, ModelConfig, count_parameters
from asr.text import BLANK_ID
from asr.train import evaluate


def main():
    data_cfg = DataConfig(data_root=Path("data"))
    train_full = NumbersASRDataset(data_cfg, "train", augment=True)
    dev_full = NumbersASRDataset(data_cfg, "dev", augment=False)
    train_ds = Subset(train_full, list(range(8)))
    dev_ds = Subset(dev_full, list(range(8)))
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_ds, batch_size=4, shuffle=False, collate_fn=collate)

    device = torch.device("cpu")
    mel_ext = MelExtractor(data_cfg).to(device)
    spec_aug = SpecAugment(data_cfg).to(device)
    model = ConformerCTC(ModelConfig(n_blocks=2, d_model=96, n_heads=4, ff_mult=2)).to(device)
    print(f"smoke model params: {count_parameters(model)/1e6:.3f}M")
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ctc = torch.nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)

    model.train()
    for step, batch in enumerate(train_loader):
        wav = batch["wav"].to(device)
        wav_lens = batch["wav_lens"].to(device)
        target = batch["target"].to(device)
        target_lens = batch["target_lens"].to(device)
        mel = mel_ext(wav)
        mel_lens = (wav_lens // mel_ext.mel.hop_length + 1).clamp_max(mel.size(-1))
        mel = spec_aug(mel)
        logits, out_lens = model(mel, mel_lens)
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
        loss = ctc(log_probs, target, out_lens, target_lens)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        print(f"step {step}: loss={loss.item():.3f} out_lens={out_lens.tolist()}")
        if step >= 2:
            break

    m = evaluate(model, mel_ext, dev_loader, device)
    print("eval metrics:", m)
    print("SMOKE OK")


if __name__ == "__main__":
    main()
