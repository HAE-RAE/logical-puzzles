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
        "name": "EXTREME",
        "cipher_stack": ["cho_shift", "jung_sub", "reverse", "cho_shift"],
        "keyword_logic": "extraction",
        "hint_count": 0,
        "description": "Double Shift + Substitution + Reverse / 0 Hints"
    },
    "LEVEL_1": {
        "name": "VERY_HARD",
        "cipher_stack": ["cho_shift", "jung_sub", "reverse"],
        "keyword_logic": "positional",
        "hint_count": 1,
        "description": "Shift + Substitution + Reverse / 1 Hint"
    },
    "LEVEL_2": {
        "name": "HARD",
        "cipher_stack": ["cho_shift", "jung_sub"],
        "keyword_logic": "positional",
        "hint_count": 2,
        "description": "Shift + Substitution / 2 Hints"
    },
    "LEVEL_3": {
        "name": "MEDIUM",
        "cipher_stack": ["cho_shift"],
        "keyword_logic": "direct",
        "hint_count": 2,
        "description": "Initial Consonant Shift / 2 Hints"
    },
    "LEVEL_4": {
        "name": "EASY",
        "cipher_stack": ["cho_shift"],
        "keyword_logic": "direct",
        "hint_count": 4,
        "description": "Initial Consonant Shift / 4 Hints"
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
        if kw_logic == "direct":
            kw_instruction = f"암호화 키워드는 '{keyword}'입니다."
        elif kw_logic == "positional":
            clean_log = log_text.replace(".", "").replace(":", "").replace(",", "")
            words = clean_log.split()
            pos = words.index(keyword) + 1
            kw_instruction = f"암호화 키워드는 아래 로그 지문의 {pos}번째 단어입니다. (문장 부호 제외)"
        else:
            kw_instruction = "암호화 키워드는 로그 지문 내에 숨겨져 있습니다. '주요키', '인증코드', '시드값' 등의 라벨 다음에 오는 단어가 키워드입니다."

        problem_text = f"--- [복구된 미션 로그] ---\n{log_text}\n---------------------------\n\n"
        problem_text += f"암호문: '{encrypted}'\n\n"
        problem_text += f"암호화 가이드:\n1. {kw_instruction}\n2. 적용된 알고리즘: {' -> '.join(config['cipher_stack']).upper()}"
        
        if hint_examples:
            problem_text += "\n예제:\n" + "\n".join(hint_examples) + "\n"
        
        problem_text += "\n복호화된 원문을 입력하세요 (공백 없이 한글로)."

        return {
            "difficulty": config["name"],
            "problem": problem_text,
            "answer": answer,
            "solution": f"단계: {' -> '.join(process)} | 정답: {answer}"
        }

def create_hangul_dataset(num_per_level: int = 3):
    print("Hangul-based Cipher 문제 생성 중...")
    generator = HangulCipherGenerator()
    all_problems = []

    for level_key in sorted(DIFFICULTY_CONFIG.keys()):
        config = DIFFICULTY_CONFIG[level_key]
        for i in range(num_per_level):
            seed = 5000 + len(all_problems)
            problem = generator.generate_problem(config, seed)
            all_problems.append({
                "question": problem["problem"],
                "answer": problem["answer"],
                "solution": problem["solution"],
                "difficulty": config["name"]
            })

    output_path = Path(__file__).resolve().parent.parent / "data" / "json" / "HANGUL_CIPHER.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in all_problems:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"생성 완료: {output_path}")

if __name__ == "__main__":
    create_hangul_dataset()
