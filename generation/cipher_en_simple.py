"""Cipher EN Simple - Dedicated low/mid/high-difficulty generator targeted at
Qwen/Qwen3-VL-8B-Instruct (8B class instruction-tuned model, 4-bit nf4).

Three calibrated difficulties (separate output files, never touches the
production cipher_en pipeline):

  easy   - Single-step REVERSE.                Target accuracy: 7-9/10 (70-90%)
  medium - Single-step CAESAR shift (key K).   Target accuracy: 4-6/10 (40-60%)
  hard   - Two-step CAESAR(K) then REVERSE.    Target accuracy: 2-4/10 (20-40%)

All puzzles use real common English words (4-6 letters) so tokenisation is
stable; algorithm description is in plain English; output format is the
`원문: <WORD>` line that the production CipherEvaluator parses.

Outputs
-------
  data/mini_simple/cipher_en_simple_easy.jsonl
  data/mini_simple/cipher_en_simple_medium.jsonl
  data/mini_simple/cipher_en_simple_hard.jsonl
  data/json/cipher_en_simple_<difficulty>.jsonl   (mirrors)

Run
---
    python generation/cipher_en_simple.py --difficulty easy   --num 5
    python generation/cipher_en_simple.py --difficulty medium --num 10
    python generation/cipher_en_simple.py --difficulty hard   --num 10
    python generation/cipher_en_simple.py --difficulty all
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# Common 5-letter nouns used by medium / hard (the tokenizer encodes them
# cleanly and the model recognises them).
WORD_BANK = [
    "APPLE", "CLOCK", "LIGHT", "HOUSE", "MOUSE",
    "TABLE", "CHAIR", "TRAIN", "WATER", "PHONE",
    "PLANT", "BREAD", "STONE", "GLASS", "MUSIC",
    "PAPER", "RIVER", "OCEAN", "SUGAR", "HONEY",
    "EARTH", "CLOUD", "GRASS", "BEACH", "CANDY",
    "HORSE", "ROBOT", "PIANO", "JUICE", "PIZZA",
]

# Easy uses 7-9 letter words so the REVERSE-only puzzle has enough room for
# the model to slip on a couple (target 7-9/10, not a ceiling at 100%).
EASY_WORD_BANK = [
    "LIBRARY", "CONCERT", "JOURNEY", "STADIUM", "RAINBOW",
    "HOLIDAY", "MORNING", "WHISPER", "FACTORY", "VILLAGE",
    "BICYCLE", "TRIANGLE", "COMPUTER", "ELEPHANT", "MOUNTAIN",
    "SANDWICH", "DIAMOND", "SUNSHINE", "BIRTHDAY", "BRIDGES",
    "PARALLEL", "EXERCISE", "PINEAPPLE", "TELESCOPE", "AIRPLANE",
    "UMBRELLA", "DOLPHIN", "MAGNOLIA", "VOLCANO", "WIZARD",
]

# Short hint pool for reverse worked-examples.
HINT_POOL_SHORT = [
    "DOG", "CAT", "SUN", "MOON", "STAR", "FISH", "BIRD",
    "TREE", "BOOK", "DESK", "RING", "ROAD", "LION", "WIND",
    "RAIN", "SNOW", "FIRE", "ROCK", "GOLD", "SHIP", "CAKE",
    "MILK", "DOOR", "WALL", "BELL", "CLAY", "ROSE", "LEAF",
]


# ---------------------------------------------------------------------------
# cipher primitives
# ---------------------------------------------------------------------------

def caesar(text: str, shift: int) -> str:
    out = []
    for ch in text.upper():
        if ch in ALPHABET:
            out.append(ALPHABET[(ALPHABET.index(ch) + shift) % 26])
        else:
            out.append(ch)
    return "".join(out)


def reverse(text: str) -> str:
    return text[::-1]


# ---------------------------------------------------------------------------
# question builders (one per difficulty)
# ---------------------------------------------------------------------------

def build_easy(answer: str, rng: random.Random, num_hints: int = 1) -> str:
    """REVERSE-only, 7-9 letter answers, one short worked example.

    Calibration for the 7-9/10 band: longer answers create transcription
    errors (skipped/swapped letters) on a non-trivial fraction of attempts
    while still being trivially understandable.
    """
    encrypted = reverse(answer)
    hint_words = rng.sample(HINT_POOL_SHORT, num_hints)
    hint_lines = "\n".join(f"  - {w} -> {reverse(w)}" for w in hint_words)
    return (
        "You are given a single English word that has been encrypted by "
        "REVERSING the order of its letters. Recover the original word.\n\n"
        f"Ciphertext: {encrypted}\n\n"
        "Algorithm: REVERSE (write the ciphertext from right to left).\n\n"
        "Worked example:\n"
        f"{hint_lines}\n\n"
        "Show your work briefly, then on the FINAL line write exactly:\n"
        "원문: <WORD>\n"
    )


def _shift_table(shift: int) -> str:
    """A compact one-line decryption mapping cipher_letter->plain_letter."""
    pairs = [f"{ALPHABET[i]}->{ALPHABET[(i - shift) % 26]}" for i in range(26)]
    return "  " + "  ".join(pairs)


def build_medium(answer: str, rng: random.Random, shift: int,
                 num_hints: int = 1) -> str:
    """Caesar shift with the shift value disclosed and ONE worked decryption
    example (no full alphabet table — that made it trivial in iter2).

    Calibration for the 4-6/10 band on Qwen3-VL-8B-Instruct:
    - shift values {3, 5, 7} (avoid 13/ROT13 lookup shortcut)
    - one worked example with per-letter substitution
    - no explicit decryption table — model must compute shifts itself
    """
    encrypted = caesar(answer, shift)

    hint_word = rng.choice(HINT_POOL_SHORT)
    enc_w = caesar(hint_word, shift)
    per_letter = ", ".join(
        f"{enc_w[i]}->{hint_word[i]}" for i in range(len(hint_word))
    )
    walkthrough = (
        f"  Example: Ciphertext '{enc_w}' (shift={shift}).\n"
        f"  Step-by-step decryption: {per_letter}\n"
        f"  Final line: 원문: {hint_word}"
    )

    return (
        "You are given an English word encrypted with a Caesar shift cipher.\n"
        f"Each plaintext letter was shifted FORWARD by {shift} in the alphabet "
        f"(A->{ALPHABET[shift % 26]}, B->{ALPHABET[(shift+1) % 26]}, ..., Z wraps to A).\n\n"
        f"Ciphertext: {encrypted}\n\n"
        f"To decrypt, shift each cipher letter BACKWARD by {shift} positions "
        "(A wraps to Z).\n\n"
        f"Worked decryption example:\n{walkthrough}\n\n"
        "Now decrypt the ciphertext above. Show your per-letter substitution, "
        "then on the FINAL line write exactly:\n"
        "원문: <WORD>\n"
    )


def build_hard(answer: str, rng: random.Random, shift: int,
               num_hints: int = 1) -> str:
    """Two-step: Caesar(shift) then REVERSE. To decrypt: REVERSE first, then
    Caesar back. One worked example with explicit per-letter substitution.

    Calibration for the 2-4/10 band on Qwen3-VL-8B-Instruct:
    - shift values {3, 5, 7}
    - one worked decryption walk-through example
    - no decryption table (otherwise becomes trivial)
    - two-step nature is the main difficulty driver
    """
    step1 = caesar(answer, shift)
    encrypted = reverse(step1)

    hint_word = rng.choice(HINT_POOL_SHORT)
    h_step1 = caesar(hint_word, shift)
    h_cipher = reverse(h_step1)
    per_letter = ", ".join(
        f"{h_step1[i]}->{hint_word[i]}" for i in range(len(hint_word))
    )
    walkthrough = (
        f"  Example: Ciphertext '{h_cipher}' (shift={shift}).\n"
        f"  Step A - reverse the ciphertext: '{h_cipher}' -> '{h_step1}'\n"
        f"  Step B - shift each letter back by {shift}: {per_letter}\n"
        f"  Final line: 원문: {hint_word}"
    )

    return (
        "You are given an English word encrypted in TWO sequential steps:\n"
        f"  Step 1: Caesar shift FORWARD by {shift} (A->{ALPHABET[shift % 26]}, "
        "..., Z wraps to A).\n"
        "  Step 2: Reverse the resulting string.\n\n"
        f"Ciphertext: {encrypted}\n\n"
        "To decrypt, undo the steps in REVERSE order:\n"
        "  A) Reverse the ciphertext.\n"
        f"  B) Shift each letter BACKWARD by {shift} (A wraps to Z).\n\n"
        f"Worked decryption example:\n{walkthrough}\n\n"
        "Now decrypt the ciphertext above. Show your work, then on the FINAL "
        "line write exactly:\n"
        "원문: <WORD>\n"
    )


# ---------------------------------------------------------------------------
# generator
# ---------------------------------------------------------------------------

DIFFICULTY_DEFAULTS = {
    "easy":   {"num": 10, "seed": 8001},
    "medium": {"num": 10, "seed": 8201},
    "hard":   {"num": 10, "seed": 8301},
}


def _pick_words(rng: random.Random, n: int, bank: List[str]) -> List[str]:
    if n <= len(bank):
        return rng.sample(bank, n)
    # If we somehow ask for more than the bank, allow repeats with shuffle.
    out = list(bank)
    rng.shuffle(out)
    while len(out) < n:
        extra = list(bank)
        rng.shuffle(extra)
        out.extend(extra)
    return out[:n]


def generate(difficulty: str, num: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    bank = EASY_WORD_BANK if difficulty == "easy" else WORD_BANK
    words = _pick_words(rng, num, bank)

    rows: List[Dict] = []
    for i, answer in enumerate(words, start=1):
        if difficulty == "easy":
            q = build_easy(answer, rng)
        elif difficulty == "medium":
            shift = rng.choice([3, 5, 7])
            q = build_medium(answer, rng, shift=shift)
        elif difficulty == "hard":
            shift = rng.choice([3, 5, 7])
            q = build_hard(answer, rng, shift=shift)
        else:
            raise ValueError(f"unknown difficulty: {difficulty}")

        rows.append({
            "id": f"cipher_en_simple_{difficulty}_{i}",
            "difficulty": difficulty,
            "task": "cipher_en_simple",
            "question": q,
            "answer": answer,
        })
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def emit(rows: List[Dict], difficulty: str, project_root: Path) -> None:
    targets = [
        project_root / "data" / "mini_simple" / f"cipher_en_simple_{difficulty}.jsonl",
        project_root / "data" / "json"        / f"cipher_en_simple_{difficulty}.jsonl",
    ]
    for p in targets:
        write_jsonl(p, rows)
        print(f"wrote {len(rows):2d} {difficulty:6s} puzzles -> {p}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard", "all"],
                        default="easy")
    parser.add_argument("--num", type=int, default=None,
                        help="Override puzzle count (default: easy=5, medium/hard=10)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override seed (default per difficulty)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    difficulties = ["easy", "medium", "hard"] if args.difficulty == "all" else [args.difficulty]

    for diff in difficulties:
        defaults = DIFFICULTY_DEFAULTS[diff]
        num = args.num if args.num is not None else defaults["num"]
        seed = args.seed if args.seed is not None else defaults["seed"]
        rows = generate(diff, num=num, seed=seed)
        emit(rows, diff, project_root)
        print(f"\nSample [{diff}]:\n" + rows[0]["question"])
        print(f"Answer: {rows[0]['answer']}\n")


if __name__ == "__main__":
    main()
