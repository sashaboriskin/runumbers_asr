"""Training loop for ConformerCTC on the ASR-2026 number-recognition challenge.

Usage:
    uv run python scripts/train.py --config configs/base.yaml

TensorBoard logs are written to <out_dir>/tb and checkpoints to <out_dir>/ckpts.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from jiwer import cer
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from .data import DataConfig, MelExtractor, NumbersASRDataset, SpecAugment, collate
from .model import ConformerCTC, ModelConfig, count_parameters
from .text import BLANK_ID, ctc_greedy_decode, words_to_digits_safe


@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=lambda: DataConfig(data_root=Path("data")))
    model: ModelConfig = field(default_factory=ModelConfig)
    epochs: int = 60
    batch_size: int = 32
    num_workers: int = 4
    lr: float = 3e-3
    weight_decay: float = 1e-2
    warmup_steps: int = 1000
    grad_clip: float = 5.0
    seed: int = 42
    out_dir: Path = Path("runs/base")
    log_every: int = 50
    val_every_epoch: int = 1
    use_balanced_sampler: bool = True
    amp: bool = True
    save_every_epoch: bool = False
    # early stopping
    patience: int = 15


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> TrainConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    dc = raw.pop("data", {})
    mc = raw.pop("model", {})
    if "data_root" in dc:
        dc["data_root"] = Path(dc["data_root"])
    if "noise_dir" in dc and dc["noise_dir"]:
        dc["noise_dir"] = Path(dc["noise_dir"])
    cfg = TrainConfig(
        data=DataConfig(**dc),
        model=ModelConfig(**mc),
        **raw,
    )
    cfg.out_dir = Path(cfg.out_dir)
    return cfg


def warmup_cosine(step: int, warmup: int, total: int, base_lr: float, min_ratio: float = 0.05) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(max(progress, 0.0), 1.0)
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * (min_ratio + (1 - min_ratio) * cos)


def evaluate(
    model: ConformerCTC,
    mel_ext: MelExtractor,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    total_cer = 0.0
    n = 0
    per_spk_cer: dict[str, list[float]] = {}
    digit_cer_sum = 0.0
    digit_n = 0
    bad_decode = 0
    with torch.no_grad():
        for batch in loader:
            wav = batch["wav"].to(device, non_blocking=True)
            wav_lens = batch["wav_lens"].to(device)
            mel = mel_ext(wav)
            # mel frames length
            mel_lens = (wav_lens // mel_ext.mel.hop_length + 1).clamp_max(mel.size(-1))
            logits, out_lens = model(mel, mel_lens)
            preds = logits.argmax(-1).cpu().tolist()
            for i, (p, L) in enumerate(zip(preds, out_lens.cpu().tolist())):
                ref_text = batch["text"][i]
                hyp_text = ctc_greedy_decode(p[:L])
                c = cer(ref_text, hyp_text) if ref_text else 0.0
                total_cer += c
                n += 1
                spk = batch["spk_id"][i]
                per_spk_cer.setdefault(spk, []).append(c)
                # digit-level CER (what the Kaggle metric measures)
                ref_digits = str(_parse_ref_digits(ref_text))
                hyp_digits = str(words_to_digits_safe(hyp_text, fallback=0))
                if hyp_digits == "0":
                    bad_decode += 1
                digit_cer_sum += cer(ref_digits, hyp_digits)
                digit_n += 1
    model.train()
    mean_text_cer = total_cer / max(1, n)
    mean_digit_cer = digit_cer_sum / max(1, digit_n)
    per_spk = {k: float(np.mean(v)) for k, v in per_spk_cer.items()}
    return {
        "text_cer": mean_text_cer,
        "digit_cer": mean_digit_cer,
        "per_spk_text_cer": per_spk,
        "bad_decode_frac": bad_decode / max(1, n),
    }


def _parse_ref_digits(ref_text: str) -> int:
    from .text import words_to_digits
    return words_to_digits(ref_text)


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "tb").mkdir(exist_ok=True)
    (cfg.out_dir / "ckpts").mkdir(exist_ok=True)
    (cfg.out_dir / "config.json").write_text(
        json.dumps({
            "data": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg.data).items()},
            "model": asdict(cfg.model),
            "train": {k: (str(v) if isinstance(v, Path) else v)
                      for k, v in asdict(cfg).items() if k not in {"data", "model"}},
        }, indent=2, ensure_ascii=False)
    )

    train_ds = NumbersASRDataset(cfg.data, split="train", augment=True)
    dev_ds = NumbersASRDataset(cfg.data, split="dev", augment=False)

    sampler = None
    shuffle = True
    if cfg.use_balanced_sampler:
        sampler = WeightedRandomSampler(train_ds.speaker_weights, num_samples=len(train_ds), replacement=True)
        shuffle = False

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=cfg.num_workers, collate_fn=collate, pin_memory=True, drop_last=True,
    )
    # Sort dev by audio duration ascending so each mini-batch has similar lengths
    # (the 101s outlier in spk_I ends up alone in its batch, avoiding huge pad).
    from torch.utils.data import SequentialSampler, BatchSampler
    dev_order = dev_ds.df.assign(_i=range(len(dev_ds.df))).copy()
    # Cheap length proxy: use reported duration if we computed it; otherwise filename hash.
    # Here we just use transcription length (correlates a bit) + idx for determinism.
    dev_order = dev_order.sort_values(["transcription"], kind="stable")["_i"].tolist()
    class _IdxSampler(torch.utils.data.Sampler):
        def __init__(self, order): self.order = order
        def __iter__(self): return iter(self.order)
        def __len__(self): return len(self.order)
    dev_loader = DataLoader(
        dev_ds,
        batch_size=max(1, cfg.batch_size // 2),
        sampler=_IdxSampler(dev_order),
        num_workers=cfg.num_workers, collate_fn=collate, pin_memory=True,
    )

    mel_ext = MelExtractor(cfg.data).to(device)
    spec_aug = SpecAugment(cfg.data).to(device)
    model = ConformerCTC(cfg.model).to(device)
    print(f"[train] params={count_parameters(model)/1e6:.2f}M")
    assert count_parameters(model) < 5_000_000, "model exceeds 5M param budget"

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.98))
    total_steps = cfg.epochs * len(train_loader)
    scaler = torch.amp.GradScaler(enabled=cfg.amp)
    ctc = torch.nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)

    tb = SummaryWriter(log_dir=str(cfg.out_dir / "tb"))
    best_dev = float("inf")
    best_epoch = -1
    step = 0
    for epoch in range(cfg.epochs):
        t0 = time.time()
        model.train()
        for batch in train_loader:
            step += 1
            lr = warmup_cosine(step, cfg.warmup_steps, total_steps, cfg.lr)
            for g in optim.param_groups:
                g["lr"] = lr

            wav = batch["wav"].to(device, non_blocking=True)
            wav_lens = batch["wav_lens"].to(device)
            target = batch["target"].to(device, non_blocking=True)
            target_lens = batch["target_lens"].to(device)

            with torch.amp.autocast("cuda", enabled=cfg.amp, dtype=torch.float16):
                mel = mel_ext(wav)
                mel_lens = (wav_lens // mel_ext.mel.hop_length + 1).clamp_max(mel.size(-1))
                mel = spec_aug(mel)
                logits, out_lens = model(mel, mel_lens)
                log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)  # [T, B, V]
                loss = ctc(log_probs, target, out_lens, target_lens)

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optim)
            scaler.update()

            if step % cfg.log_every == 0:
                tb.add_scalar("train/loss", loss.item(), step)
                tb.add_scalar("train/lr", lr, step)
                print(f"ep{epoch} step{step}/{total_steps} loss={loss.item():.3f} lr={lr:.2e}")

        # eval
        if (epoch + 1) % cfg.val_every_epoch == 0:
            metrics = evaluate(model, mel_ext, dev_loader, device)
            text_cer = metrics["text_cer"]
            digit_cer = metrics["digit_cer"]
            tb.add_scalar("dev/text_cer", text_cer, step)
            tb.add_scalar("dev/digit_cer", digit_cer, step)
            tb.add_scalar("dev/bad_decode_frac", metrics["bad_decode_frac"], step)
            # split ind vs ood
            ind_spk = {"spk_A", "spk_B", "spk_C", "spk_D", "spk_E", "spk_F"}
            ind_vals, ood_vals = [], []
            for spk, c in metrics["per_spk_text_cer"].items():
                tb.add_scalar(f"dev/spk_text_cer/{spk}", c, step)
                if spk in ind_spk:
                    ind_vals.append(c)
                else:
                    ood_vals.append(c)
            if ind_vals:
                tb.add_scalar("dev/ind_text_cer", float(np.mean(ind_vals)), step)
            if ood_vals:
                tb.add_scalar("dev/ood_text_cer", float(np.mean(ood_vals)), step)
            dt = time.time() - t0
            print(
                f"[eval] ep{epoch} text_cer={text_cer:.4f} digit_cer={digit_cer:.4f} "
                f"bad={metrics['bad_decode_frac']:.3f} time={dt:.1f}s"
            )
            # checkpoint
            is_best = digit_cer < best_dev
            if is_best:
                best_dev = digit_cer
                best_epoch = epoch
                torch.save(
                    {
                        "model": model.state_dict(),
                        "cfg_model": asdict(cfg.model),
                        "epoch": epoch,
                        "metrics": metrics,
                    },
                    cfg.out_dir / "ckpts" / "best.pt",
                )
            if cfg.save_every_epoch:
                torch.save({"model": model.state_dict(), "epoch": epoch}, cfg.out_dir / "ckpts" / f"ep{epoch:03d}.pt")
            if epoch - best_epoch > cfg.patience:
                print(f"[train] early stop: no improvement for {cfg.patience} epochs")
                break

    tb.close()
    print(f"[train] done. best dev digit_cer={best_dev:.4f} at epoch {best_epoch}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
