"""Hangul-based Cipher Puzzle Generator
[진행도] ☑ 완료 / ☐ 미완성
[파일명] hangul_cipher.py
[목적] 한글의 자모 구조를 활용한 고난도 암호 퍼즐 생성
"""

import random
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
import math

# ============================================================================
# 난이도 설정 - 한글 자모 구조 활용
# ============================================================================

DIFFICULTY_CONFIG = {
    "LEVEL_0": {
        "name": "easy",
        "cipher_stack": ["cho_shift"],
        "keyword_logic": "direct",
        "hint_count": 3,
        "description": "Easy (Avg Accuracy ~75%): Cho_shift / Direct Logic / 3 Hints"
    },
    "LEVEL_1": {
        "name": "medium",
        "cipher_stack": ["cho_shift", "jung_sub"],
        "keyword_logic": "positional",
        "hint_count": 1,
        "description": "Medium (Avg Accuracy ~50%): Cho_shift + Jung_sub / Positional Logic / 1 Hint"
    },
    "LEVEL_2": {
        "name": "hard",
        "cipher_stack": ["cho_shift", "jung_sub", "reverse", "cho_shift"],
        "keyword_logic": "extraction",
        "hint_count": 0,
        "description": "Hard (Avg Accuracy ~25%): Double Shift + Jung_sub + Reverse / Extraction Logic / 0 Hints"
    }
}

# ============================================================================
# 한글 처리 시스템
# ============================================================================

class HangulCipherSystem:
    def __init__(self):
        self.CHO = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        self.JUNG = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
        self.JONG = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        self.BASE = 0xAC00

    def decompose(self, char: str) -> Tuple[int, int, int]:
        if not char or not (0xAC00 <= ord(char) <= 0xD7A3):
            return -1, -1, -1
        code = ord(char) - self.BASE
        cho = code // (21 * 28)
        jung = (code % (21 * 28)) // 28
        jong = code % 28
        return cho, jung, jong

    def compose(self, cho: int, jung: int, jong: int) -> str:
        if cho < 0 or jung < 0: return ""
        code = self.BASE + (cho * 21 * 28) + (jung * 28) + jong
        return chr(code)

    def cho_shift_encrypt(self, text: str, keyword: str) -> str:
        result = []
        for i, char in enumerate(text):
            key_char = keyword[i % len(keyword)]
            k_cho, _, _ = self.decompose(key_char)
            shift = k_cho if k_cho >= 0 else ord(key_char) % 19
            
            c_cho, c_jung, c_jong = self.decompose(char)
            if c_cho >= 0:
                new_cho = (c_cho + shift) % 19
                result.append(self.compose(new_cho, c_jung, c_jong))
            else:
                result.append(char)
        return "".join(result)

    def jung_sub_encrypt(self, text: str, keyword: str) -> str:
        # Generate substitution table based on keyword's vowels
        keyword_jungs = []
        for char in keyword:
            _, jung, _ = self.decompose(char)
            if jung >= 0 and jung not in keyword_jungs:
                keyword_jungs.append(jung)
        
        mapping = keyword_jungs.copy()
        for i in range(21):
            if i not in mapping:
                mapping.append(i)
        
        result = []
        for char in text:
            cho, jung, jong = self.decompose(char)
            if cho >= 0:
                result.append(self.compose(cho, mapping[jung], jong))
            else:
                result.append(char)
        return "".join(result)

# ============================================================================
# 가상 로그 생성기
# ============================================================================

class KoreanMissionLogGenerator:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.statuses = ["심각", "안정", "동기화", "잠금", "과부하"]
        self.targets = ["위성", "데이터베이스", "그리드", "단말기", "코어"]
        self.actions = ["오프라인", "준비완료", "대기", "침입됨"]
        self.key_labels = ["주요키", "인증코드", "시드값", "벡터값", "암호키", "접속토큰"]

    def generate_log(self) -> Tuple[str, str]:
        keyword = self.rng.choice(["하늘", "바다", "나무", "구름", "태양", "달빛", "지도", "열쇠", "비밀"])
        label = self.rng.choice(self.key_labels)
        
        sentences = [
            f"시스템 보고서 ID {self.rng.randint(100, 999)}: 상태 {self.rng.choice(self.statuses)}.",
            f"대상 {self.rng.choice(self.targets)} 상태는 {self.rng.choice(self.actions)}.",
            f"암호화 {label} 설정 단계에서 키워드 {keyword} 부수어짐 없이 적용됨."
        ]
        self.rng.shuffle(sentences)
        return " ".join(sentences), keyword

# ============================================================================
# Guided-distillation style solution (teacher trace)
# ============================================================================

SFT_SOLUTION_RUBRIC_KO = (
    "STEP0=문제 메타 · STEP1=주어진 조건 · STEP2=풀이 전개 · STEP3=답·검산"
)


def _build_cipher_ko_solution(
    config: Dict,
    process: List[str],
    answer: str,
    keyword: str,
    encrypted: str,
    kw_logic: str,
    kw_instruction: str,
    pos_for_solution: int = None,
) -> str:
    """SFT용: 메타 → 키워드 → 역파이프라인 → 검산."""
    stack = config["cipher_stack"]
    lines = [
        SFT_SOLUTION_RUBRIC_KO,
        "[STEP 0] 문제 메타",
        f"  - 난이도: {config['name']}",
        f"  - 암호문: '{encrypted}' (길이 {len(encrypted)}자)",
        f"  - 평문(정답): {answer}",
        f"  - 암호화 적용 순서(평문→암호문): {' → '.join(process)}",
        "  - 복호화: **마지막에 적용된 변환부터** 역순으로 역연산을 곱한다.",
        "[STEP 1] 주어진 조건 (키워드·지문 규칙)",
        f"  - 지문 규칙: {kw_instruction}",
    ]
    if kw_logic == "positional" and pos_for_solution is not None:
        lines.append(
            f"  - 절차: 로그에서 구두점(., :, ,) 제거 → 공백 단위 토큰화 → "
            f"{pos_for_solution}번째 단어 = '{keyword}'")
    elif kw_logic == "extraction":
        lines.append(
            "  - 절차: 라벨(주요키·인증코드·시드값·벡터값·암호키·접속토큰 등) 바로 뒤 "
            "토큰이 키워드.")
    else:
        lines.append(f"  - 키워드는 지문에 명시: '{keyword}'")

    rev = list(reversed(stack))
    _cipher_op_name_ko = {
        "cho_shift": "초성 시프트⁻¹",
        "jung_sub": "중성 치환⁻¹",
        "reverse": "문자열 뒤집기⁻¹",
    }
    decrypt_pipeline = " → ".join(_cipher_op_name_ko.get(s, s) for s in rev)
    lines.append("[STEP 2] 풀이 전개 (암호문 → 평문, 역연산)")
    lines.append(
        f"  · 요약: 스택 {len(stack)}단 · 키워드 '{keyword}' · "
        f"복호 파이프라인: {decrypt_pipeline} · SEG {len(rev)}개"
    )
    for i, st in enumerate(rev, 1):
        if st == "cho_shift":
            lines.append(
                f"    [SEG {i}] 초성 시프트 역연산: 키워드 '{keyword}'의 각 글자 초성 인덱스를 "
                f"키로 쓰되, 암호화 때 더했던 만큼 **빼서** mod 19로 초성 복원 "
                f"(비한글·공백은 그대로).")
        elif st == "jung_sub":
            lines.append(
                f"    [SEG {i}] 중성 치환 역연산: 키워드에서 등장 순서대로 모음 인덱스를 앞에 두고 "
                f"나머지 21개 중성 인덱스를 채운 치환표의 **역치환**으로 중성 복원.")
        elif st == "reverse":
            lines.append(f"    [SEG {i}] 문자열 **전체 뒤집기**로 역순 단계 해제.")
        else:
            lines.append(f"    [SEG {i}] {st} 역연산 적용.")

    lines.extend([
        "[STEP 3] 답·검산",
        f"  - 최종 답: '{answer}' (공백 없는 한글)",
        f"  - 복호 결과가 '{answer}'와 일치하는지 확인.",
        "  - 예제 행(문제 본문)을 같은 키·같은 스택으로 암호화해 암호문과 규칙이 맞는지 대조.",
    ])
    return "\n".join(lines)


# ============================================================================
# Generator
# ============================================================================

class HangulCipherGenerator:
    def __init__(self):
        self.cipher = HangulCipherSystem()

    def generate_problem(self, config: Dict, seed: int = None) -> Dict:
        rng = random.Random(seed)
        log_gen = KoreanMissionLogGenerator(rng)
        
        answer_pool = ["대한민국", "정보보안", "미래기술", "평화통일", "민주주의", "자유만세", "과학 발전", "산업혁명"]
        answer = rng.choice(answer_pool).replace(" ", "")
        
        log_text, keyword = log_gen.generate_log()
        
        current_text = answer
        process = []
        for stage in config["cipher_stack"]:
            if stage == "cho_shift":
                current_text = self.cipher.cho_shift_encrypt(current_text, keyword)
                process.append(f"초성 시프트(키={keyword})")
            elif stage == "jung_sub":
                current_text = self.cipher.jung_sub_encrypt(current_text, keyword)
                process.append(f"중성 치환(키={keyword})")
            elif stage == "reverse":
                current_text = current_text[::-1]
                process.append("역순")
        
        encrypted = current_text
        
        # Build hints
        hint_examples = []
        for _ in range(config["hint_count"]):
            test_word = rng.choice(["사과", "바다", "친구", "공부", "사랑"])
            temp = test_word
            for stage in config["cipher_stack"]:
                if stage == "cho_shift": temp = self.cipher.cho_shift_encrypt(temp, keyword)
                elif stage == "jung_sub": temp = self.cipher.jung_sub_encrypt(temp, keyword)
                elif stage == "reverse": temp = temp[::-1]
            hint_examples.append(f"  - {test_word} -> {temp}")

        kw_logic = config["keyword_logic"]
        pos_for_solution = None
        if kw_logic == "direct":
            kw_instruction = f"암호화 키워드는 '{keyword}'입니다."
        elif kw_logic == "positional":
            clean_log = log_text.replace(".", "").replace(":", "").replace(",", "")
            words = clean_log.split()
            pos_for_solution = words.index(keyword) + 1
            kw_instruction = (
                f"암호화 키워드는 아래 로그 지문의 {pos_for_solution}번째 단어입니다. "
                f"(문장 부호 제외)")
        else:
            kw_instruction = (
                "암호화 키워드는 로그 지문 내에 숨겨져 있습니다. '주요키', '인증코드', "
                "'시드값' 등의 라벨 다음에 오는 단어가 키워드입니다.")

        problem_text = f"--- [복구된 미션 로그] ---\n{log_text}\n---------------------------\n\n"
        problem_text += f"암호문: '{encrypted}'\n\n"
        problem_text += f"암호화 가이드:\n1. {kw_instruction}\n2. 적용된 알고리즘: {' -> '.join(config['cipher_stack']).upper()}"
        
        if hint_examples:
            problem_text += "\n예제:\n" + "\n".join(hint_examples) + "\n"
        
        problem_text += "\n복호화된 원문을 입력하세요 (공백 없이 한글로)."

        solution = _build_cipher_ko_solution(
            config=config,
            process=process,
            answer=answer,
            keyword=keyword,
            encrypted=encrypted,
            kw_logic=kw_logic,
            kw_instruction=kw_instruction,
            pos_for_solution=pos_for_solution,
        )

        return {
            "difficulty": config["name"],
            "problem": problem_text,
            "answer": answer,
            "solution": solution,
        }

def create_hangul_dataset(num_per_level: int = 3):
    print(f"Hangul-based Cipher 문제 생성 중...")
    print(f"난이도별 {num_per_level}개씩 생성")
    print("="*70)

    generator = HangulCipherGenerator()
    all_problems = []

    # 쉬운 난이도부터 어려운 순서로 생성 (LEVEL_0: easy -> LEVEL_2: hard)
    for level_key in sorted(DIFFICULTY_CONFIG.keys()):
        config = DIFFICULTY_CONFIG[level_key]
        difficulty = config["name"]
        
        print(f"\n[{difficulty}] {config['description']}")

        for i in range(num_per_level):
            seed = 5000 + len(all_problems)
            problem = generator.generate_problem(config, seed)
            all_problems.append({
                "id": f"cipher_ko_{difficulty}_{i:04d}",
                "question": problem["problem"],
                "answer": problem["answer"],
                "solution": problem["solution"],
                "difficulty": difficulty,
            })
            # print(f"  {i+1}. {problem['answer'][:15]}... 생성 완료")

    # JSONL 저장
    output_jsonl_path = Path(__file__).resolve().parent.parent / "data" / "jsonl" / "cipher_ko.jsonl"
    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for p in all_problems:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # CSV 저장 (id, question, answer, solution, difficulty 만)
    import pandas as pd
    output_csv_path = Path(__file__).resolve().parent.parent / "data" / "csv" / "cipher_ko.csv"
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["id", "question", "answer", "solution", "difficulty"]
    df = pd.DataFrame([{k: p[k] for k in cols} for p in all_problems])
    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

    print(f"생성 완료: {output_jsonl_path}")
    print(f"CSV 파일 생성 완료: {output_csv_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate Hangul Cipher Puzzles')
    parser.add_argument('--num', type=int, default=2, help='Number of puzzles per difficulty level')
    args = parser.parse_args()
    
    # 각 난이도별 n개씩 생성
    create_hangul_dataset(num_per_level=args.num)
