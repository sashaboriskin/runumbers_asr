# ASR-2026: Russian Spoken Numbers Recognition

**Team:** `kuzya_lakomkin` — Boriskin Aleksandr, Fedor Avilov  
**Repository:** [github.com/sashaboriskin/runumbers_asr](https://github.com/sashaboriskin/runumbers_asr)  
**Final Public LB Score:** 1.718

---

## 1. Task Overview

The goal is to build a compact ASR system that recognizes Russian spoken numbers in the range [1,000--999,999]. Key constraints:

- Model size $\leq$ 5M parameters, trained from scratch (no pretrained weights).
- Input: 16 kHz mono audio.
- Evaluation metric: harmonic mean of Character Error Rate (CER) over in-domain (inD) and out-of-domain (ooD) speaker subsets. Lower is better.

The harmonic mean penalizes imbalance between inD and ooD performance, so the system must generalize to unseen speakers rather than overfitting to training voices.

## 2. Data Analysis

### Training set
- **12,553 samples** from **6 speakers** (spk_A--spk_F).
- Severe speaker imbalance: spk_E alone accounts for 45% (5,686 samples), while spk_D and spk_F have fewer than 1,000 each.
- Mixed sample rates: 24 kHz (83%) and 22.05 kHz (17%), all WAV format.
- Duration: 1.1--5.7 s (mean 3.1 s).
- 12 label outliers with transcription < 1,000 (outside the official range).

### Development set
- **2,265 samples** from **10 speakers**: 6 in-domain (spk_A--F, 100 each) and 4 out-of-domain (spk_H: 633, spk_I: 497, spk_J: 344, spk_K: 191).
- All 16 kHz; mixed WAV/MP3.
- One anomalous 101-second recording from spk_I (typical samples are 1--6 s).

### Key data challenges
1. **Speaker imbalance** in training --- naive sampling would bias toward spk_E.
2. **Out-of-domain speakers** in dev/test --- the model must generalize from 6 training voices to 14 test voices.
3. **Format heterogeneity** --- different sample rates and codecs require normalization.
4. **Noisy test conditions** --- the task description hints at "real-world signal degradations."

## 3. Approach

### 3.1. Text Normalization

We convert numeric labels to their spoken Russian form using `num2words`:

$$139473 \rightarrow \text{"сто тридцать девять тысяч четыреста семьдесят три"}$$

The CTC model is trained on these word-level character sequences. At inference, we parse Russian number words back to digits using a rule-based parser that:
- Splits on thousand markers ("тысяча" / "тысячи" / "тысяч"),
- Parses each group (hundreds + tens + units) via lookup tables.

This approach naturally handles the "тысяча" hint from the task: e.g., "одна тысяча пять" $\rightarrow$ 1,005.

**Vocabulary:** 34 tokens --- CTC blank + 32 lowercase Cyrillic letters (а--я, no ё) + space.

### 3.2. Model Architecture: Conformer-CTC

The model follows the Conformer architecture adapted for a compact parameter budget:

1. **Conv2D subsampling** (2 layers, stride 2 each) $\rightarrow$ 4$\times$ time reduction.
2. **Sinusoidal positional encoding.**
3. **$N$ Conformer blocks**, each containing:
   - Macaron-style Feed-Forward ($\times 0.5$)
   - Multi-Head Self-Attention (MHSA)
   - Convolution Module (pointwise $\rightarrow$ GLU $\rightarrow$ depthwise $\rightarrow$ BN $\rightarrow$ SiLU $\rightarrow$ pointwise)
   - Feed-Forward ($\times 0.5$)
   - LayerNorm
4. **Linear CTC head** (d_model $\rightarrow$ 34).

**Input features:** 80-bin log-mel spectrogram (16 kHz, 25 ms window, 10 ms hop).

| Config | d_model | n_blocks | n_heads | Parameters |
|--------|---------|----------|---------|------------|
| v1     | 144     | 8        | 4       | 3.97M      |
| v2     | 160     | 8        | 4       | 4.89M      |

### 3.3. Training Pipeline

**Optimizer:** AdamW ($\beta_1{=}0.9$, $\beta_2{=}0.98$), weight decay $10^{-2}$.  
**LR schedule:** linear warmup (500 steps) $\rightarrow$ cosine decay to 5% of peak LR.  
**Batch size:** 256, mixed precision (AMP, float16).  
**Early stopping:** patience = 15 epochs on dev digit CER.

**Speaker-balanced sampling:** `WeightedRandomSampler` with weights inversely proportional to per-speaker counts --- ensures equal expected contribution from each speaker per epoch despite the 45% spk_E dominance.

**Audio caching:** all training audio is pre-resampled to 16 kHz mono and stored as float16 `.npy` files. DataLoader workers read these via `numpy.mmap`, which is ~10$\times$ faster than decoding + resampling WAV/MP3 on the fly. This was critical for GPU utilization on H100.

### 3.4. Data Augmentation

| Augmentation | v1 | v2 |
|---|---|---|
| Speed perturbation | off (`[1.0]`) | `[0.9, 1.0, 1.0, 1.0, 1.1]` |
| Gain jitter | $\pm$6 dB | $\pm$6 dB |
| MUSAN additive noise | yes, SNR 5--25 dB | yes, SNR 5--25 dB |
| SpecAugment freq masks | $2 \times 20$ | $2 \times 27$ |
| SpecAugment time masks | $2 \times 30$ | $2 \times 40$ |

Speed perturbation was initially disabled to maximize I/O throughput during the first baseline run, then enabled for v2. MUSAN noise (~900 clips of ambient noise and music) was enabled for the final evaluation runs of both v1 and v2.

### 3.5. CTC Beam Search with Number-Constrained Decoding

Standard CTC greedy decoding produces a single best-path hypothesis. This is suboptimal because:
- A single character error can invalidate an entire word, causing the number parser to drop it.
- The model does not "know" that the output must be a valid Russian number.

Our beam search decoder:

1. **CTC prefix beam search** (beam width = 20, top-$k$ = 10 characters per frame) produces $N$ hypotheses ranked by CTC log-probability.
2. **Exact parse:** try `words_to_digits()` on each hypothesis best-first; accept the first that yields a valid number in [1,000; 999,999].
3. **Word-level correction:** if no hypothesis parses exactly, apply Levenshtein-based spelling correction (max edit distance = 2) against the closed vocabulary of ~30 valid Russian number words, then re-parse.
4. **Safe fallback:** if all else fails, drop unrecognized words and parse what remains.

This cascading strategy recovers from most CTC character-level errors by leveraging the fact that the output space is highly constrained.

## 4. Experiments and Results

### 4.1. Summary

| Experiment | Model | Decoding | inD CER | ooD CER | Hmean (dev) | Public LB |
|:---|:---|:---|---:|---:|---:|---:|
| v1 | 3.97M | greedy | 1.947% | 15.691% | 3.464% | 3.861 |
| v1 + beam | 3.97M | beam=20 | 1.422% | 12.271% | 2.549% | 2.172 |
| **v2 + beam** | **4.89M** | **beam=20** | **0.858%** | **4.952%** | **1.463%** | **1.718** |

### 4.2. Submission History on Kaggle

| # | Date | Description | Public LB |
|---|------|-------------|-----------|
| 1 | Apr 2026 | v1, greedy decode | 3.861 |
| 2 | Apr 2026 | v1, beam search (width=20) | 2.172 |
| 3 | Apr 2026 | v2, beam search (width=20) | **1.718** |

### 4.3. Per-Speaker Breakdown (Dev Set)

**v1 (greedy):**

| Speaker | Digit CER | Count | Domain |
|---------|----------:|------:|--------|
| spk_A   | 1.67%     | 100   | inD    |
| spk_B   | 1.33%     | 100   | inD    |
| spk_C   | 2.03%     | 100   | inD    |
| spk_D   | 2.25%     | 100   | inD    |
| spk_E   | 0.87%     | 100   | inD    |
| spk_F   | 3.53%     | 100   | inD    |
| spk_H   | 1.58%     | 633   | ooD    |
| spk_I   | 38.51%    | 497   | ooD    |
| spk_J   | 4.17%     | 344   | ooD    |
| spk_K   | 23.83%    | 191   | ooD    |

**v2 + beam search:**

| Speaker | Digit CER | Count | Domain |
|---------|----------:|------:|--------|
| spk_A   | 0.67%     | 100   | inD    |
| spk_B   | 0.67%     | 100   | inD    |
| spk_C   | 0.67%     | 100   | inD    |
| spk_D   | 1.58%     | 100   | inD    |
| spk_E   | 0.17%     | 100   | inD    |
| spk_F   | 1.40%     | 100   | inD    |
| spk_H   | 1.57%     | 633   | ooD    |
| spk_I   | 8.36%     | 497   | ooD    |
| spk_J   | 1.55%     | 344   | ooD    |
| spk_K   | 13.44%    | 191   | ooD    |

The most dramatic improvement is on the hardest ooD speakers:
- **spk_I:** 38.51% $\rightarrow$ 8.36% ($4.6\times$ reduction)
- **spk_K:** 23.83% $\rightarrow$ 13.44% ($1.8\times$ reduction)

## 5. Analysis of Key Techniques

### 5.1. Impact of Beam Search (v1 greedy $\rightarrow$ v1 beam)

Beam search alone (no retraining) reduced the harmonic mean from 3.464% to 2.549% on dev and from 3.861 to 2.172 on the public leaderboard. This is a **44% relative improvement** from decoding alone.

The improvement is especially pronounced on ooD speakers (15.7% $\rightarrow$ 12.3%), where CTC character errors are more frequent. The number-constrained decoding recovers many cases where greedy decoding produces a slightly garbled word that fails to parse, but a lower-ranked beam hypothesis is correct.

### 5.2. Impact of Model and Augmentation (v1 beam $\rightarrow$ v2 beam)

The v2 model with augmentations further reduced the harmonic mean from 2.549% to 1.463% on dev (1.718 on LB). Key changes and their expected contributions:

- **Speed perturbation** introduces variation in speaking rate, helping the model generalize to speakers with different tempos. This is especially important for ooD speakers whose speaking rate may differ from training data.
- **More aggressive SpecAugment** (freq mask 20$\rightarrow$27, time mask 30$\rightarrow$40) acts as a stronger regularizer, forcing the model to rely on redundant spectral information.
- **Larger model** (3.97M $\rightarrow$ 4.89M) provides additional capacity, particularly in the attention and feed-forward layers (d_model 144$\rightarrow$160), enabling richer representations of speaker-invariant features.
- **MUSAN noise augmentation** (SNR 5--25 dB) improves robustness to background noise, which the task description indicates is present in test data.

### 5.3. Speaker-Balanced Sampling

Without balancing, spk_E (45% of training data) would dominate gradient updates. The `WeightedRandomSampler` ensures each speaker contributes equally in expectation, which directly improves inD CER uniformity and prevents the model from specializing to one speaker's characteristics.

### 5.4. Audio Caching for Training Speed

Initial training on H100 was extremely slow (~1 hour for 100 steps) despite 100% GPU utilization. The bottleneck was CPU-bound audio decoding and resampling in DataLoader workers. Pre-caching all audio as 16 kHz float16 `.npy` files with memory-mapped reads eliminated this bottleneck, enabling batch size 256 with full GPU utilization.

## 6. Repository Structure

```
src/asr/
    text.py         # digits <-> Russian words normalization
    data.py         # dataset, augmentations, mel extraction
    model.py        # Conformer-CTC architecture
    train.py        # training loop with TensorBoard logging
    infer.py        # inference with greedy/beam decoding
    beam_search.py  # CTC beam search + word correction
scripts/
    train.py        # CLI entrypoint for training
    evaluate_dev.py # per-speaker CER evaluation
    cache_audio.py  # pre-resample audio to 16 kHz .npy
    smoke_pipeline.py  # end-to-end sanity check
configs/
    base.yaml       # v1 config (3.97M params)
    v2.yaml         # v2 config (4.89M params)
notebooks/
    kaggle_submission.ipynb  # Kaggle inference notebook
```

## 7. Conclusion

Our best system achieves a **1.718 harmonic mean CER** on the public leaderboard, combining:

1. A **4.89M-parameter Conformer-CTC** model trained from scratch with comprehensive augmentations (speed perturbation, gain jitter, MUSAN noise, SpecAugment).
2. A **number-constrained CTC beam search** decoder that exploits the closed vocabulary of Russian number words to correct character-level CTC errors via multi-hypothesis search and Levenshtein-based word correction.
3. **Speaker-balanced sampling** and **audio caching** for efficient and equitable training.

The largest single improvement came from beam search decoding (44% relative improvement without retraining), followed by the combination of a larger model and stronger augmentations (an additional 21% relative improvement on the public leaderboard).
