"""Audio dataset + augmentations for ASR-2026 Russian spoken numbers.

Design notes:
- We resample every audio to 16 kHz on load (torchaudio.functional.resample).
- Training target = digits_to_words(int(transcription)) encoded as char ids.
- Features = log-mel spectrogram, 80 bins, 16k/400/160 (25ms / 10ms).
- Augmentations (train only):
    * speed perturb (random resample to 0.9x or 1.1x)
    * SpecAugment (freq + time masks)
    * additive noise from a noise_dir (e.g. MUSAN)
    * gain jitter
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as AF
from torch.utils.data import Dataset

from .text import CHAR2ID, digits_to_words, encode

TARGET_SR = 16_000


def _load_audio(path: Path, target_sr: int = TARGET_SR) -> torch.Tensor:
    """Load mono audio and resample to target_sr. Returns float32 tensor [T]."""
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception:
        # fallback for formats soundfile cannot decode (older libsndfile, etc.)
        import librosa
        data, sr = librosa.load(str(path), sr=None, mono=True)
        data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    wav = torch.from_numpy(np.ascontiguousarray(data))
    if sr != target_sr:
        wav = AF.resample(wav, sr, target_sr)
    return wav.contiguous()


@dataclass
class DataConfig:
    data_root: Path
    train_csv: str = "train/train.csv"
    dev_csv: str = "dev/dev.csv"
    target_sr: int = TARGET_SR
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    min_label: int = 1000        # filter train rows below this
    max_label: int = 999_999
    # If set, we read pre-resampled 16 kHz mono audio from <cache_dir>/<split>/<basename>.npy
    # instead of the raw dataset. Produced by scripts/cache_audio.py.
    cache_dir: Path | None = None
    # training aug
    speed_perturb: tuple[float, ...] = (0.9, 1.0, 1.0, 1.0, 1.1)
    gain_db_range: tuple[float, float] = (-6.0, 6.0)
    noise_dir: Path | None = None
    noise_snr_db: tuple[float, float] = (5.0, 25.0)
    noise_prob: float = 0.5
    # SpecAugment
    freq_mask: int = 20
    freq_mask_n: int = 2
    time_mask: int = 30
    time_mask_n: int = 2


class MelExtractor(torch.nn.Module):
    def __init__(self, cfg: DataConfig):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.target_sr,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            n_mels=cfg.n_mels,
            power=2.0,
            f_min=20.0,
            f_max=cfg.target_sr // 2,
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: [B, T] or [T]
        m = self.mel(wav)
        m = torch.log(m.clamp_min(1e-10))
        return m  # [..., n_mels, T']


class SpecAugment(torch.nn.Module):
    def __init__(self, cfg: DataConfig):
        super().__init__()
        self.freq_mask = cfg.freq_mask
        self.freq_mask_n = cfg.freq_mask_n
        self.time_mask = cfg.time_mask
        self.time_mask_n = cfg.time_mask_n

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: [B, n_mels, T]
        B, F, T = mel.shape
        for _ in range(self.freq_mask_n):
            f = random.randint(0, self.freq_mask)
            f0 = random.randint(0, max(0, F - f))
            mel[:, f0 : f0 + f, :] = 0.0
        for _ in range(self.time_mask_n):
            t = random.randint(0, min(self.time_mask, max(1, T // 2)))
            t0 = random.randint(0, max(0, T - t))
            mel[:, :, t0 : t0 + t] = 0.0
        return mel


class NoiseBank:
    """Loads noise files lazily from a directory; MUSAN-style flat layout ok."""

    def __init__(self, noise_dir: Path, target_sr: int = TARGET_SR):
        self.files: list[Path] = sorted(
            [p for p in noise_dir.rglob("*") if p.suffix.lower() in {".wav", ".mp3", ".flac"}]
        )
        self.target_sr = target_sr
        if not self.files:
            raise RuntimeError(f"no noise files under {noise_dir}")

    def sample(self, length: int, rng: random.Random) -> torch.Tensor:
        p = self.files[rng.randrange(len(self.files))]
        wav = _load_audio(p, self.target_sr)
        if wav.numel() < length:
            reps = math.ceil(length / max(1, wav.numel()))
            wav = wav.repeat(reps)
        start = rng.randint(0, wav.numel() - length)
        return wav[start : start + length]


def _add_noise(wav: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    sig_p = wav.pow(2).mean().clamp_min(1e-10)
    nse_p = noise.pow(2).mean().clamp_min(1e-10)
    target_nse_p = sig_p / (10 ** (snr_db / 10))
    noise = noise * (target_nse_p / nse_p).sqrt()
    return wav + noise


class NumbersASRDataset(Dataset):
    def __init__(
        self,
        cfg: DataConfig,
        split: str,
        *,
        augment: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.augment = augment
        csv_path = cfg.data_root / (cfg.train_csv if split == "train" else cfg.dev_csv)
        df = pd.read_csv(csv_path)
        if split == "train":
            before = len(df)
            df = df[(df["transcription"] >= cfg.min_label) & (df["transcription"] <= cfg.max_label)]
            df = df.reset_index(drop=True)
            print(f"[data] train filter {cfg.min_label}<=x<={cfg.max_label}: {before} -> {len(df)}")
        self.df = df
        # precompute text targets
        self._targets: list[list[int]] = []
        self._text: list[str] = []
        for n in df["transcription"].astype(int).tolist():
            txt = digits_to_words(n)
            self._text.append(txt)
            self._targets.append(encode(txt))
        # speaker weights
        counts = df["spk_id"].value_counts().to_dict()
        self.speaker_weights = torch.tensor(
            [1.0 / counts[s] for s in df["spk_id"].tolist()], dtype=torch.double
        )
        self._rng = random.Random(1234 + (0 if split == "train" else 1))
        self.noise_bank: NoiseBank | None = None
        if augment and cfg.noise_dir is not None:
            self.noise_bank = NoiseBank(cfg.noise_dir, cfg.target_sr)

    def __len__(self) -> int:
        return len(self.df)

    def _read_wav(self, filename: str) -> torch.Tensor:
        """Read audio. If cache_dir is set, read the pre-resampled 16 kHz npy file.
        Otherwise read the raw source and resample on the fly (slow)."""
        if self.cfg.cache_dir is not None:
            base = Path(filename).with_suffix(".npy").name
            p = self.cfg.cache_dir / self.split / base
            # np.load with mmap_mode is fast and avoids CPU-side copy until used
            arr = np.load(p, mmap_mode="r")
            return torch.from_numpy(np.ascontiguousarray(arr))
        return _load_audio(self.cfg.data_root / self.split / filename, self.cfg.target_sr)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        wav = self._read_wav(row["filename"])
        if self.augment:
            wav = self._augment(wav)
        return {
            "wav": wav,
            "wav_len": wav.numel(),
            "target": torch.tensor(self._targets[idx], dtype=torch.long),
            "target_len": len(self._targets[idx]),
            "text": self._text[idx],
            "spk_id": row["spk_id"],
            "idx": idx,
        }

    def _augment(self, wav: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        # speed perturb: skip if only {1.0} in the list to avoid expensive CPU resample
        if cfg.speed_perturb and set(cfg.speed_perturb) != {1.0}:
            factor = self._rng.choice(cfg.speed_perturb)
            if factor != 1.0:
                new_sr = int(cfg.target_sr * factor)
                wav = AF.resample(wav.unsqueeze(0), cfg.target_sr, new_sr).squeeze(0)
        # gain jitter
        g = self._rng.uniform(*cfg.gain_db_range)
        wav = wav * (10 ** (g / 20))
        # additive noise
        if self.noise_bank is not None and self._rng.random() < cfg.noise_prob:
            snr = self._rng.uniform(*cfg.noise_snr_db)
            noise = self.noise_bank.sample(wav.numel(), self._rng)
            wav = _add_noise(wav, noise, snr)
        return wav


def collate(batch: Sequence[dict]) -> dict:
    wavs = [b["wav"] for b in batch]
    targets = [b["target"] for b in batch]
    wav_lens = torch.tensor([w.numel() for w in wavs], dtype=torch.long)
    target_lens = torch.tensor([t.numel() for t in targets], dtype=torch.long)
    max_w = int(wav_lens.max().item())
    max_t = int(target_lens.max().item())
    wav_pad = torch.zeros(len(batch), max_w, dtype=torch.float32)
    tgt_pad = torch.zeros(len(batch), max_t, dtype=torch.long)
    for i, (w, t) in enumerate(zip(wavs, targets)):
        wav_pad[i, : w.numel()] = w
        tgt_pad[i, : t.numel()] = t
    return {
        "wav": wav_pad,
        "wav_lens": wav_lens,
        "target": tgt_pad,
        "target_lens": target_lens,
        "text": [b["text"] for b in batch],
        "spk_id": [b["spk_id"] for b in batch],
    }


__all__ = [
    "DataConfig",
    "MelExtractor",
    "SpecAugment",
    "NumbersASRDataset",
    "collate",
]
