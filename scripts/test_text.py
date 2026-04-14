"""Round-trip tests for src/asr/text.py.

    uv run python scripts/test_text.py
"""
from __future__ import annotations

import random

from asr.text import (
    CHAR2ID,
    VOCAB_SIZE,
    VALID_WORDS,
    ctc_greedy_decode,
    digits_to_words,
    encode,
    words_to_digits,
    words_to_digits_safe,
)


def test_round_trip_exhaustive_small():
    for n in range(1000, 2100):
        w = digits_to_words(n)
        got = words_to_digits(w)
        assert got == n, f"{n} -> {w!r} -> {got}"


def test_round_trip_sampled_large():
    rng = random.Random(0)
    for _ in range(5000):
        n = rng.randint(1000, 999999)
        w = digits_to_words(n)
        got = words_to_digits(w)
        assert got == n, f"{n} -> {w!r} -> {got}"


def test_corner_cases():
    cases = {
        1000: "одна тысяча",
        2000: "две тысячи",
        5000: "пять тысяч",
        10000: "десять тысяч",
        100000: "сто тысяч",
        999999: "девятьсот девяносто девять тысяч девятьсот девяносто девять",
        1001: "одна тысяча один",
        20001: "двадцать тысяч один",
    }
    for n, expected in cases.items():
        assert digits_to_words(n) == expected, (n, digits_to_words(n), expected)
        assert words_to_digits(expected) == n


def test_char_vocab_covers_all_number_words():
    # Every character in every number word must be in our char vocab
    # (lowercase, space, and cyrillic letters)
    for n in range(0, 1000):
        for c in digits_to_words(n):
            assert c in CHAR2ID, f"missing char {c!r} from word for {n}"
    for n in (1000, 2000, 5000, 10000, 100000, 999999):
        for c in digits_to_words(n):
            assert c in CHAR2ID, f"missing char {c!r} from word for {n}"


def test_all_valid_words_parseable():
    # Every word in VALID_WORDS must be encodable
    for w in VALID_WORDS:
        for c in w:
            assert c in CHAR2ID, f"invalid char {c!r} in word {w!r}"


def test_safe_drops_garbage():
    assert words_to_digits_safe("сто тысяч xxxx пять", fallback=0) == 100005
    assert words_to_digits_safe("", fallback=1234) == 1234


def test_ctc_greedy_decode():
    # "пять" = п я т ь; with CTC blank (id=0) and repeats
    ids = [CHAR2ID["п"], CHAR2ID["п"], 0, CHAR2ID["я"], 0, CHAR2ID["т"], CHAR2ID["ь"], 0]
    assert ctc_greedy_decode(ids) == "пять"


def test_encode():
    ids = encode("сто один")
    assert all(0 < i < VOCAB_SIZE for i in ids)


def main():
    test_round_trip_exhaustive_small()
    test_round_trip_sampled_large()
    test_corner_cases()
    test_char_vocab_covers_all_number_words()
    test_all_valid_words_parseable()
    test_safe_drops_garbage()
    test_ctc_greedy_decode()
    test_encode()
    print("all text tests passed")


if __name__ == "__main__":
    main()
