"""Russian number text normalization / denormalization for spoken numbers ASR.

Our transcription target is the *spoken* form:
    139473 -> "сто тридцать девять тысяч четыреста семьдесят три"
Inference decodes back:
    "сто тридцать девять тысяч четыреста семьдесят три" -> 139473

The character vocabulary is the Russian lowercase alphabet + space + blank.
"""
from __future__ import annotations

from typing import Iterable

from num2words import num2words

# Russian alphabet used in number words. "ё" is safe to exclude since no number
# word in 1..999999 contains it, but we keep it in vocab for robustness.
RU_ALPHABET = "абвгдежзийклмнопрстуфхцчшщъыьэюя"
SPACE = " "
BLANK = "<blank>"

# Vocabulary indices:
#   0           blank (CTC)
#   1..len(A)   alphabet chars
#   len(A)+1    space
CHAR_VOCAB: list[str] = [BLANK, *RU_ALPHABET, SPACE]
CHAR2ID: dict[str, int] = {c: i for i, c in enumerate(CHAR_VOCAB)}
ID2CHAR: dict[int, str] = {i: c for c, i in CHAR2ID.items()}
BLANK_ID = 0
SPACE_ID = CHAR2ID[SPACE]
VOCAB_SIZE = len(CHAR_VOCAB)

assert VOCAB_SIZE == 34, VOCAB_SIZE  # 32 letters + space + blank


def digits_to_words(n: int) -> str:
    """Convert an integer to the Russian spoken form used as training target."""
    if not (0 <= n < 1_000_000):
        raise ValueError(f"n={n} out of supported range [0, 999999]")
    return num2words(n, lang="ru")


# --- Denormalization (words -> int) ---

_UNITS: dict[str, int] = {
    "ноль": 0,
    "один": 1, "одна": 1, "одно": 1,
    "два": 2, "две": 2,
    "три": 3,
    "четыре": 4,
    "пять": 5,
    "шесть": 6,
    "семь": 7,
    "восемь": 8,
    "девять": 9,
}
_TEENS: dict[str, int] = {
    "десять": 10,
    "одиннадцать": 11, "двенадцать": 12, "тринадцать": 13, "четырнадцать": 14,
    "пятнадцать": 15, "шестнадцать": 16, "семнадцать": 17, "восемнадцать": 18,
    "девятнадцать": 19,
}
_TENS: dict[str, int] = {
    "двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50,
    "шестьдесят": 60, "семьдесят": 70, "восемьдесят": 80, "девяносто": 90,
}
_HUNDREDS: dict[str, int] = {
    "сто": 100, "двести": 200, "триста": 300, "четыреста": 400, "пятьсот": 500,
    "шестьсот": 600, "семьсот": 700, "восемьсот": 800, "девятьсот": 900,
}
_THOUSAND_MARKERS: set[str] = {"тысяча", "тысячи", "тысяч"}

# Full set of valid tokens for LM / decoder constraint.
VALID_WORDS: set[str] = (
    set(_UNITS) | set(_TEENS) | set(_TENS) | set(_HUNDREDS) | _THOUSAND_MARKERS
)

_SINGLE_WORD_VALUE: dict[str, int] = {**_UNITS, **_TEENS, **_TENS, **_HUNDREDS}


def _parse_group(words: list[str]) -> int:
    """Parse a group of words that encodes a number in [0, 999]."""
    value = 0
    for w in words:
        v = _SINGLE_WORD_VALUE.get(w)
        if v is None:
            raise ValueError(f"unknown number word: {w!r}")
        value += v
    return value


def words_to_digits(text: str) -> int:
    """Parse Russian number words to integer. Raises ValueError on malformed input."""
    tokens = text.strip().split()
    if not tokens:
        raise ValueError("empty text")
    # Split on thousand markers.
    pre: list[str] = []
    post: list[str] = []
    seen_thousand = False
    for t in tokens:
        if t in _THOUSAND_MARKERS:
            if seen_thousand:
                raise ValueError("multiple thousand markers")
            seen_thousand = True
        elif seen_thousand:
            post.append(t)
        else:
            pre.append(t)

    if seen_thousand:
        thousands = _parse_group(pre) if pre else 1
        units = _parse_group(post) if post else 0
        return thousands * 1000 + units
    return _parse_group(pre)


def words_to_digits_safe(text: str, fallback: int = 1000) -> int:
    """Best-effort: drop unknown words, then parse. Return fallback on failure."""
    tokens = [t for t in text.strip().split() if t in VALID_WORDS]
    if not tokens:
        return fallback
    try:
        return words_to_digits(" ".join(tokens))
    except ValueError:
        return fallback


# --- Encode / decode for the char-level CTC target ---


def encode(text: str) -> list[int]:
    """Encode a text (already normalized, lowercase cyrillic + spaces) to ids."""
    ids: list[int] = []
    for ch in text:
        if ch not in CHAR2ID:
            raise ValueError(f"char not in vocab: {ch!r}")
        ids.append(CHAR2ID[ch])
    return ids


def decode_ids(ids: Iterable[int]) -> str:
    """Decode a flat sequence of ids (no CTC collapsing) to string."""
    return "".join(ID2CHAR[i] for i in ids if i != BLANK_ID)


def ctc_greedy_decode(ids: Iterable[int]) -> str:
    """CTC greedy decoding: collapse repeats, drop blanks."""
    out: list[str] = []
    prev = -1
    for i in ids:
        if i != prev:
            if i != BLANK_ID:
                out.append(ID2CHAR[i])
            prev = i
    return "".join(out)
