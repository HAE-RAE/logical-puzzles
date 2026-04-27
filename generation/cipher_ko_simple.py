"""Cipher KO Simple - Korean counterpart to cipher_en_simple, calibrated to
the same target bands on `Qwen/Qwen3-VL-8B-Instruct` (4-bit nf4).

Calibration applied (see desc_method_diff_en_ko.md for the rationale):

  easy   - REVERSE syllables of a 4-5 syllable word, NO worked example.
           (Target 7-9/10. KO syllable reverse with one example was 10/10.)
  medium - 초성 Caesar shift on a SHRUNK 14-자모 alphabet (no 쌍자음),
           shifts in {1, 2}, explicit decryption table inlined,
           2-syllable answers, fully decomposed worked example.
           (Target 4-6/10. The 19-자모 + arithmetic version was 0/10.)
  hard   - 14-자모 Caesar shift then reverse syllables, same alphabet
           and shift restrictions as medium, with the reverse step shown
           explicitly in the walkthrough. (Target 2-4/10.)

Outputs
-------
  data/mini_simple/cipher_ko_simple_<difficulty>.jsonl
  data/json/cipher_ko_simple_<difficulty>.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


# Full 19-element 초성 row used by Hangul syllable composition.
CHO_19 = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
# 14 basic 초성 used by medium / hard — drops 쌍자음 (ㄲ, ㄸ, ㅃ, ㅆ, ㅉ),
# the documented #1 source of model mis-counts.
CHO_14 = "ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ"

# Index translations between the two alphabets.
CHO14_TO_19 = [CHO_19.index(c) for c in CHO_14]
CHO19_TO_14 = {CHO_19.index(c): i for i, c in enumerate(CHO_14)}

HANGUL_BASE = 0xAC00
NUM_JUNG = 21
NUM_JONG = 28


# ---------------------------------------------------------------------------
# Hangul helpers
# ---------------------------------------------------------------------------

def is_hangul(ch: str) -> bool:
    return 0xAC00 <= ord(ch) <= 0xD7A3


def decompose(ch: str) -> Tuple[int, int, int]:
    code = ord(ch) - HANGUL_BASE
    cho = code // (NUM_JUNG * NUM_JONG)
    jung = (code % (NUM_JUNG * NUM_JONG)) // NUM_JONG
    jong = code % NUM_JONG
    return cho, jung, jong


def compose(cho: int, jung: int, jong: int) -> str:
    return chr(HANGUL_BASE + (cho * NUM_JUNG + jung) * NUM_JONG + jong)


def shift_cho_basic(syllable: str, shift: int) -> str:
    """Shift the 초성 inside the 14-element basic alphabet (mod 14).

    If a syllable's 초성 is one of the 5 쌍자음, leave the syllable untouched.
    Word banks below are filtered to avoid this.
    """
    if not is_hangul(syllable):
        return syllable
    cho19, jung, jong = decompose(syllable)
    if cho19 not in CHO19_TO_14:
        return syllable
    idx14 = CHO19_TO_14[cho19]
    new14 = (idx14 + shift) % 14
    new19 = CHO14_TO_19[new14]
    return compose(new19, jung, jong)


def caesar14_ko(text: str, shift: int) -> str:
    return "".join(shift_cho_basic(c, shift) for c in text)


def reverse_str(text: str) -> str:
    return text[::-1]


def all_basic(word: str) -> bool:
    """True if every Hangul syllable's 초성 is in the 14-basic alphabet."""
    for ch in word:
        if not is_hangul(ch):
            continue
        cho, _, _ = decompose(ch)
        if cho not in CHO19_TO_14:
            return False
    return True


# ---------------------------------------------------------------------------
# Word banks (each filtered for the 14-basic alphabet where applicable)
# ---------------------------------------------------------------------------

# Easy uses 4-5 syllable words and does NOT use the Caesar shift, so 쌍자음
# entries are allowed (e.g. 코끼리). Calibrated for 7-9/10.
EASY_WORD_BANK_KO = [
    "도서관", "자전거", "컴퓨터", "코끼리", "운동회",
    "박물관", "비행기", "냉장고", "지하철", "냉면집",
    "수요일", "고양이", "독수리", "설악산", "한강물",
    "사무실", "달팽이", "기차역", "할아버지", "강아지",
    "해바라기", "도토리묵", "솔방울", "손가락", "거북이",
    "송아지", "감자전", "백과사전", "도화지", "바람개비",
]

# Medium / hard need every 초성 to live in the 14-basic alphabet so the
# Caesar shift is well-defined for every syllable. 2 syllables only.
_MED_HARD_CANDIDATES = [
    "사과", "시계", "의자", "기차", "전화",
    "식물", "강물", "음악", "구름", "지구",
    "과자", "유리", "포도", "라면", "주스",
    "공원", "버스", "달걀", "노래", "그림",
    "나무", "바다", "별빛", "햇살", "바위",
    "촛불", "수박", "고래", "사자", "산책",
    "거울", "가방", "지도", "농장", "도서",
]
WORD_BANK_KO_BASIC = [w for w in _MED_HARD_CANDIDATES if all_basic(w)]

# Sample words reused inside worked-example walkthroughs.
WALK_SAMPLES = [w for w in ["사람", "공책", "마음", "고기", "노을"] if all_basic(w)]


# ---------------------------------------------------------------------------
# Question builders
# ---------------------------------------------------------------------------

def build_easy_ko(answer: str, rng: random.Random) -> str:
    """REVERSE syllables. No worked example — KO syllable reverse with even
    one example ceilings at 100%. We want it to slip on 1-3/10."""
    encrypted = reverse_str(answer)
    return (
        "주어진 한글 단어는 음절(글자)의 순서를 뒤집어(REVERSE) 암호화한 것입니다. "
        "원래 단어를 복원하세요.\n\n"
        f"암호문: {encrypted}\n\n"
        "알고리즘: REVERSE — 암호문의 음절을 오른쪽에서 왼쪽으로 읽으면 원문이 됩니다. "
        "음절 하나하나를 빠짐없이 옮겨야 합니다.\n\n"
        "풀이 과정을 간단히 보이고, 마지막 줄에 정확히 다음 형식으로 출력하세요:\n"
        "원문: <단어>\n"
    )


def _decryption_table_14(shift: int) -> str:
    """One-line cipher_초성 -> plain_초성 mapping for the 14-basic alphabet."""
    pairs = [
        f"{CHO_14[i]}->{CHO_14[(i - shift) % 14]}"
        for i in range(14)
    ]
    return "  " + "  ".join(pairs)


def _decompose_str(syllable: str) -> str:
    cho, jung, jong = decompose(syllable)
    JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
    JONG = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"
    cho_c = CHO_19[cho]
    jung_c = JUNG[jung]
    jong_c = JONG[jong] if jong else "∅"
    return f"({cho_c}+{jung_c}+{jong_c})"


def _pre_decompose_block(text: str) -> str:
    """Show every Hangul syllable already decomposed so the model's only
    remaining work is the lookup + recomposition."""
    lines = []
    for ch in text:
        if is_hangul(ch):
            lines.append(f"  {ch} = {_decompose_str(ch)}")
    return "\n".join(lines)


def build_medium_ko(answer: str, rng: random.Random, shift: int) -> str:
    """14-자모 Caesar. Now also pre-decomposes the ciphertext so the model
    only does lookup -> recomposition (decomposition is the documented
    failure mode)."""
    encrypted = caesar14_ko(answer, shift)
    table = _decryption_table_14(shift)

    sample_word = rng.choice(WALK_SAMPLES)
    sample_enc = caesar14_ko(sample_word, shift)
    walk_lines = []
    for s_in, s_out in zip(sample_enc, sample_word):
        c_in = CHO_19[decompose(s_in)[0]]
        c_out = CHO_19[decompose(s_out)[0]]
        walk_lines.append(
            f"    {s_in}{_decompose_str(s_in)} -> 초성 {c_in}->{c_out} -> "
            f"{s_out}{_decompose_str(s_out)}"
        )
    walkthrough = (
        f"  예시: 암호문 '{sample_enc}' (shift={shift}).\n"
        f"  사전 분해:\n{_pre_decompose_block(sample_enc)}\n"
        + "\n".join(walk_lines) + "\n"
        f"  결합: {sample_word}\n"
        f"  마지막 줄: 원문: {sample_word}"
    )

    return (
        "주어진 한글 단어는 각 음절의 초성을 14-자모 순서에서 +"
        f"{shift}만큼 이동하여 암호화한 것입니다.\n\n"
        f"14개 초성 순서 (쌍자음 제외): {' '.join(CHO_14)}\n\n"
        f"암호문: {encrypted}\n\n"
        f"암호문 사전 분해 (각 음절을 이미 (초성+중성+종성)으로 분해해 두었습니다):\n"
        f"{_pre_decompose_block(encrypted)}\n\n"
        f"복호화 매핑 (각 암호 초성 -> 평문 초성, shift={shift}):\n"
        f"{table}\n\n"
        "복호화 절차: 위 분해에서 각 음절의 초성만 매핑으로 바꾼 뒤, "
        "원래의 중성/종성과 다시 결합하세요. 중성/종성은 절대 변경하지 마세요.\n\n"
        f"풀이 예시:\n{walkthrough}\n\n"
        "이제 위 암호문을 복호화하세요. 풀이를 보이고, 마지막 줄에 정확히 다음 형식으로 출력하세요. "
        "`원문:`은 마지막 줄에만 한 번 적습니다.\n"
        "원문: <단어>\n"
    )


def build_hard_ko(answer: str, rng: random.Random, shift: int) -> str:
    """14-자모 Caesar then syllable reverse. Pre-decomposes both the
    ciphertext and the post-reverse intermediate so the model only does
    lookup + recomposition."""
    step1 = caesar14_ko(answer, shift)
    encrypted = reverse_str(step1)
    reversed_back = reverse_str(encrypted)  # equals step1 by construction
    table = _decryption_table_14(shift)

    sample_word = rng.choice(WALK_SAMPLES)
    s_step1 = caesar14_ko(sample_word, shift)
    s_encrypted = reverse_str(s_step1)
    walk_lines = []
    for s_in, s_out in zip(s_step1, sample_word):
        c_in = CHO_19[decompose(s_in)[0]]
        c_out = CHO_19[decompose(s_out)[0]]
        walk_lines.append(
            f"    {s_in}{_decompose_str(s_in)} -> 초성 {c_in}->{c_out} -> "
            f"{s_out}{_decompose_str(s_out)}"
        )
    walkthrough = (
        f"  예시: 암호문 '{s_encrypted}' (shift={shift}).\n"
        f"  단계 A — 음절 순서 뒤집기: '{s_encrypted}' -> '{s_step1}'\n"
        f"  단계 B — 사전 분해:\n{_pre_decompose_block(s_step1)}\n"
        f"  단계 B — 초성 매핑:\n"
        + "\n".join(walk_lines) + "\n"
        f"  결합: {sample_word}\n"
        f"  마지막 줄: 원문: {sample_word}"
    )

    return (
        "주어진 한글 단어는 다음 두 단계로 암호화되었습니다:\n"
        f"  단계 1: 각 음절의 초성을 14-자모 순서에서 +{shift} 이동.\n"
        "  단계 2: 음절 순서 뒤집기.\n\n"
        f"14개 초성 순서 (쌍자음 제외): {' '.join(CHO_14)}\n\n"
        f"암호문: {encrypted}\n"
        f"음절 순서 뒤집기 후 (즉 단계 B의 입력): {reversed_back}\n\n"
        f"단계 B의 입력 사전 분해:\n{_pre_decompose_block(reversed_back)}\n\n"
        f"복호화 매핑 (각 암호 초성 -> 평문 초성, shift={shift}):\n"
        f"{table}\n\n"
        "복호화는 역순으로 두 단계를 거칩니다:\n"
        "  A) 음절 순서를 뒤집습니다 (위 \"단계 B의 입력\" 참고).\n"
        "  B) 위 분해된 각 음절의 초성만 매핑으로 바꿔 다시 결합합니다. 중성/종성은 유지합니다.\n\n"
        f"풀이 예시:\n{walkthrough}\n\n"
        "이제 복호화하세요. 풀이를 보이고, 마지막 줄에 정확히 다음 형식으로 출력하세요. "
        "`원문:`은 마지막 줄에만 한 번 적습니다.\n"
        "원문: <단어>\n"
    )


# ---------------------------------------------------------------------------
# Generator entry-point
# ---------------------------------------------------------------------------

DIFFICULTY_DEFAULTS = {
    "easy":   {"num": 10, "seed": 7001},
    "medium": {"num": 10, "seed": 7201},
    "hard":   {"num": 10, "seed": 7301},
}


def _pick_words(rng: random.Random, n: int, bank: List[str]) -> List[str]:
    if n <= len(bank):
        return rng.sample(bank, n)
    out = list(bank)
    rng.shuffle(out)
    while len(out) < n:
        extra = list(bank); rng.shuffle(extra); out.extend(extra)
    return out[:n]


def generate(difficulty: str, num: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    if difficulty == "easy":
        bank = EASY_WORD_BANK_KO
    else:
        bank = WORD_BANK_KO_BASIC
    words = _pick_words(rng, num, bank)

    rows: List[Dict] = []
    for i, answer in enumerate(words, start=1):
        if difficulty == "easy":
            q = build_easy_ko(answer, rng)
        elif difficulty == "medium":
            shift = rng.choice([1, 2])
            q = build_medium_ko(answer, rng, shift=shift)
        elif difficulty == "hard":
            shift = rng.choice([1, 2])
            q = build_hard_ko(answer, rng, shift=shift)
        else:
            raise ValueError(f"unknown difficulty: {difficulty}")

        rows.append({
            "id": f"cipher_ko_simple_{difficulty}_{i}",
            "difficulty": difficulty,
            "task": "cipher_ko_simple",
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
        project_root / "data" / "mini_simple" / f"cipher_ko_simple_{difficulty}.jsonl",
        project_root / "data" / "json"        / f"cipher_ko_simple_{difficulty}.jsonl",
    ]
    for p in targets:
        write_jsonl(p, rows)
        print(f"wrote {len(rows):2d} {difficulty:6s} puzzles -> {p}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard", "all"],
                        default="all")
    parser.add_argument("--num", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    difficulties = ["easy", "medium", "hard"] if args.difficulty == "all" else [args.difficulty]

    for diff in difficulties:
        defaults = DIFFICULTY_DEFAULTS[diff]
        num = args.num if args.num is not None else defaults["num"]
        seed = args.seed if args.seed is not None else defaults["seed"]
        rows = generate(diff, num=num, seed=seed)
        emit(rows, diff, project_root)


if __name__ == "__main__":
    main()
