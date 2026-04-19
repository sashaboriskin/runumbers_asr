"""CTC beam search decoding with word-level spelling correction.

The key idea: since all outputs must be valid Russian numbers in [1000, 999999],
we can use beam search to find multiple hypotheses and pick the best one that
actually parses to a valid number.  If none parse exactly, we apply word-level
spelling correction (Levenshtein distance to closest valid Russian number word).
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .text import BLANK_ID, ID2CHAR, VALID_WORDS, words_to_digits, words_to_digits_safe


# ---------------------------------------------------------------------------
# CTC prefix beam search
# ---------------------------------------------------------------------------

def _log_add(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a >= b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def ctc_beam_search(
    log_probs: np.ndarray,
    beam_width: int = 20,
    blank_id: int = BLANK_ID,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Standard CTC prefix beam search.

    Args:
        log_probs: [T, V] log probabilities (numpy, float32).
        beam_width: number of beams to retain at each step.
        blank_id: CTC blank token id.
        top_k: only consider top-k characters at each step (0 = all).
            Reduces cost from O(T * B * V) to O(T * B * K).

    Returns:
        List of (decoded_text, log_probability) sorted best-first.
    """
    T, V = log_probs.shape
    NEG_INF = -math.inf

    # State: prefix_str → [log_p_blank_end, log_p_non_blank_end]
    beams: dict[str, list[float]] = {"": [0.0, NEG_INF]}

    for t in range(T):
        # Pre-select top-k characters at this time-step
        if 0 < top_k < V:
            indices = np.argpartition(log_probs[t], -top_k)[-top_k:]
            if blank_id not in indices:
                indices = np.append(indices, blank_id)
        else:
            indices = np.arange(V)

        new_beams: dict[str, list[float]] = {}

        def _get(prefix: str) -> list[float]:
            if prefix not in new_beams:
                new_beams[prefix] = [NEG_INF, NEG_INF]
            return new_beams[prefix]

        # Prune to top-beam_width
        scored = sorted(
            beams.items(), key=lambda x: _log_add(x[1][0], x[1][1]), reverse=True
        )
        if len(scored) > beam_width:
            scored = scored[:beam_width]

        for prefix, (pb, pnb) in scored:
            ptotal = _log_add(pb, pnb)
            lp_blank = float(log_probs[t, blank_id])
            last_char = prefix[-1] if prefix else None

            # 1. Extend with blank → same prefix
            entry = _get(prefix)
            entry[0] = _log_add(entry[0], ptotal + lp_blank)

            # 2. Extend with each non-blank character
            for v in indices:
                if v == blank_id:
                    continue
                c = ID2CHAR[v]
                lp_c = float(log_probs[t, v])

                if c == last_char:
                    # Repeated char: duplicate from blank path, merge from non-blank
                    dup = _get(prefix + c)
                    dup[1] = _log_add(dup[1], pb + lp_c)
                    entry[1] = _log_add(entry[1], pnb + lp_c)
                else:
                    ext = _get(prefix + c)
                    ext[1] = _log_add(ext[1], ptotal + lp_c)

        beams = new_beams

    results = [(p, _log_add(pb, pnb)) for p, (pb, pnb) in beams.items()]
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Word-level spelling correction
# ---------------------------------------------------------------------------

_VALID_WORDS_LIST: list[str] = sorted(VALID_WORDS)


def _levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for c1 in s1:
        curr = [prev[0] + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def _correct_word(w: str, max_dist: int = 2) -> str | None:
    """Find the closest valid Russian number word within *max_dist* edits."""
    if w in VALID_WORDS:
        return w
    best_word: str | None = None
    best_dist = max_dist + 1
    for vw in _VALID_WORDS_LIST:
        if abs(len(vw) - len(w)) > max_dist:
            continue
        d = _levenshtein(w, vw)
        if d < best_dist:
            best_dist = d
            best_word = vw
            if d == 0:
                break
    return best_word if best_dist <= max_dist else None


def words_to_digits_corrected(text: str, fallback: int = 100_000) -> int:
    """Parse with per-word spelling correction for CTC errors."""
    tokens = text.strip().split()
    if not tokens:
        return fallback
    corrected: list[str] = []
    for t in tokens:
        c = _correct_word(t)
        if c is not None:
            corrected.append(c)
    if not corrected:
        return fallback
    try:
        return words_to_digits(" ".join(corrected))
    except ValueError:
        return fallback


# ---------------------------------------------------------------------------
# Top-level decode helper
# ---------------------------------------------------------------------------

def decode_beams(
    log_probs: np.ndarray,
    beam_width: int = 20,
    fallback: int = 100_000,
) -> tuple[int, str]:
    """Run beam search and pick the best hypothesis that yields a valid number.

    Strategy:
        1. Try exact ``words_to_digits`` on each beam (best-first).
        2. Try ``words_to_digits_corrected`` on top-5 beams.
        3. Fall back to ``words_to_digits_safe`` on the best beam.

    Returns:
        (predicted_number, decoded_text)
    """
    candidates = ctc_beam_search(log_probs, beam_width=beam_width)

    # 1. Exact parse
    for text, _score in candidates:
        try:
            n = words_to_digits(text)
            if 1000 <= n <= 999_999:
                return n, text
        except ValueError:
            pass

    # 2. Corrected parse on top beams
    for text, _score in candidates[:5]:
        n = words_to_digits_corrected(text, fallback=0)
        if 1000 <= n <= 999_999:
            return n, text

    # 3. Safe fallback
    if candidates:
        text = candidates[0][0]
        n = words_to_digits_safe(text, fallback=fallback)
        return int(max(1000, min(999_999, n))), text

    return fallback, ""
