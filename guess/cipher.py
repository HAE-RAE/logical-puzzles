"""Cipher 퍼즐 v260112 - Intermediate Layer Added"""

import random
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
import math

# ============================================================================
# 난이도 설정 - 알고리즘 복잡도 + 정보 비대칭성
# ============================================================================

DIFFICULTY_CONFIG = {
    "LEVEL_0": {
        "name": "EXPERT",
        "cipher_stack": ["playfair", "transposition", "vigenere"],
        "keyword_logic": "extraction",
        "hint_count": 2,
        "description": "Playfair + Transposition + Vigenere with Conditional Key / 2 Hints"
    },
    "LEVEL_1": {
        "name": "HARD",
        "cipher_stack": ["playfair", "vigenere"],
        "keyword_logic": "positional",
        "hint_count": 4,
        "description": "Playfair + Vigenere / 4 Hints"
    },
    "LEVEL_2": {
        "name": "MEDIUM",
        "cipher_stack": ["vigenere", "reverse"],
        "keyword_logic": "positional",
        "hint_count": 6,
        "description": "Vigenere + Reverse / 6 Hints"
    },
    "LEVEL_3": {
        "name": "EASY",
        "cipher_stack": ["vigenere"],
        "keyword_logic": "direct", # 키워드 직접 명시
        "hint_count": 12,
        "description": "Single Vigenere / 12 Hints"
    }
}

# ============================================================================
# 가상 컨텍스트 생성기 (외부 지식 차단)
# ============================================================================

class MissionLogGenerator:
    """완전 무작위 지문 생성기"""
    
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.components = {
            "id": lambda: str(self.rng.randint(100, 9999)),
            "status": lambda: self.rng.choice(["CRITICAL", "STABLE", "SYNCING", "LOCKED", "OVERLOADED"]),
            "target": lambda: self.rng.choice(["SATELLITE", "DATABASE", "GRID", "TERMINAL", "CORE"]),
            "action": lambda: self.rng.choice(["OFFLINE", "READY", "STANDBY", "BREACHED"]),
            "coord": lambda: f"{self.rng.randint(0, 180)}.{self.rng.randint(0, 99)}",
            "hex": lambda: hex(self.rng.randint(4096, 65535)).upper()[2:],
        }
        self.key_labels = ["PRIMARY KEY", "AUTH CODE", "SEED", "VECTOR", "CIPHER KEY", "ACCESS TOKEN"]

    def generate_log(self) -> Tuple[str, str]:
        """지문과 랜덤 키워드 생성"""
        keyword = self.rng.choice(["ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO", "FOXTROT", "GOLF", "HOTEL", "INDIA"])
        label = self.rng.choice(self.key_labels)
        
        sentences = [
            f"SYSTEM REPORT ID {self.components['id']()}: STATUS {self.components['status']()}.",
            f"TARGET {self.components['target']()} IS {self.components['action']()}.",
            f"COORDINATES SET TO {self.components['coord']()} | SECTOR {self.components['hex']()}.",
            f"ENCRYPTION {label} IS DETERMINED AS {keyword}."
        ]
        self.rng.shuffle(sentences)
        return " ".join(sentences), keyword

# ============================================================================
# Advanced Cipher System
# ============================================================================

class AdvancedCipher:
    def __init__(self):
        self.alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def vigenere_encrypt(self, text: str, keyword: str) -> str:
        result = []
        keyword_repeated = (keyword * (len(text) // len(keyword) + 1))[:len(text)]
        for i, char in enumerate(text.upper()):
            if char in self.alphabet:
                new_idx = (self.alphabet.index(char) + self.alphabet.index(keyword_repeated[i])) % 26
                result.append(self.alphabet[new_idx])
            else: result.append(char)
        return ''.join(result)

    def substitution_encrypt(self, text: str, keyword: str) -> str:
        seen = set()
        key_chars = []
        for char in keyword.upper():
            if char in self.alphabet and char not in seen:
                key_chars.append(char); seen.add(char)
        for char in self.alphabet:
            if char not in seen: key_chars.append(char)
        key = ''.join(key_chars)
        
        result = []
        for char in text.upper():
            if char in self.alphabet:
                result.append(key[self.alphabet.index(char)])
            else: result.append(char)
        return ''.join(result)

    def playfair_encrypt(self, text: str, keyword: str) -> str:
        """플레이페어 암호 (두 글자 쌍 치환)"""
        # 5x5 행렬 생성 (J는 I와 동일하게 처리)
        matrix = []
        seen = set(['J'])
        chars = []
        for char in (keyword.upper() + self.alphabet):
            if char not in seen and char in self.alphabet:
                chars.append(char)
                seen.add(char)
        
        matrix = [chars[i:i+5] for i in range(0, 25, 5)]
        
        # 텍스트 전처리 (짝수 길이, 연속된 글자 처리)
        text = text.upper().replace('J', 'I')
        processed = ""
        i = 0
        while i < len(text):
            a = text[i]
            b = text[i+1] if i+1 < len(text) else 'X'
            if a == b:
                processed += a + 'X'
                i += 1
            else:
                processed += a + b
                i += 2
        
        def find_pos(char):
            for r in range(5):
                for c in range(5):
                    if matrix[r][c] == char: return r, c
            return 0, 0

        result = ""
        for i in range(0, len(processed), 2):
            r1, c1 = find_pos(processed[i])
            r2, c2 = find_pos(processed[i+1])
            
            if r1 == r2: # Same row
                result += matrix[r1][(c1+1)%5] + matrix[r2][(c2+1)%5]
            elif c1 == c2: # Same column
                result += matrix[(r1+1)%5][c1] + matrix[(r2+1)%5][c2]
            else: # Rectangle
                result += matrix[r1][c2] + matrix[r2][c1]
        return result

    def columnar_transpose(self, text: str, key: str) -> str:
        # (기존 로직 유지 또는 강화 - 여기서는 그대로 유지)
        text = text.replace(" ", "")
        key_order = sorted(range(len(key)), key=lambda k: key[k])
        cols = len(key)
        rows = math.ceil(len(text) / cols)
        padded_text = text.ljust(rows * cols, 'X')
        grid = [padded_text[i:i+cols] for i in range(0, len(padded_text), cols)]
        result = ""
        for k_idx in key_order:
            for r in range(rows):
                result += grid[r][k_idx]
        return result

# ============================================================================
# Generator
# ============================================================================

class SelfContainedCipherGenerator:
    def __init__(self):
        self.cipher = AdvancedCipher()

    def generate_random_string(self, rng: random.Random, length: int) -> str:
        """순수 랜덤 문자열 생성 (의미 제거)"""
        return ''.join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(length))

    def generate_problem(self, config: Dict, seed: int = None) -> Dict:
        rng = random.Random(seed)
        log_gen = MissionLogGenerator(rng)
        
        # 1. 완전 랜덤 정답 생성 (의미성 배제)
        # 길이는 10~14자 사이로 랜덤 설정
        answer = self.generate_random_string(rng, rng.randint(10, 14))
        
        log_text, keyword = log_gen.generate_log()
        
        # 2. 암호화 레이어 적용 (논리 분기 추가)
        current_text = answer
        process_log = []
        
        # 로그 지문의 숫자 정보를 가져옴
        log_id = int(re.search(r'\d+', log_text).group())
        
        for stage in config["cipher_stack"]:
            if stage == "vigenere":
                current_text = self.cipher.vigenere_encrypt(current_text, keyword)
                process_log.append(f"Vigenere(key={keyword})")
            elif stage == "playfair":
                # 분기 논리: 로그 ID가 짝수면 키워드 정방향, 홀수면 역방향 사용
                actual_key = keyword if log_id % 2 == 0 else keyword[::-1]
                current_text = self.cipher.playfair_encrypt(current_text, actual_key)
                process_log.append(f"Playfair(key={actual_key})")
            elif stage == "transposition":
                current_text = self.cipher.columnar_transpose(current_text, keyword)
                process_log.append(f"Columnar Transposition(key={keyword})")
            elif stage == "reverse":
                current_text = current_text[::-1]
                process_log.append("Reverse")
            elif stage == "substitution":
                current_text = self.cipher.substitution_encrypt(current_text, keyword)
                process_log.append(f"Substitution(key={keyword})")
        
        encrypted = current_text
        
        # 3. 힌트 생성 (완전 랜덤 단어)
        hint_examples = []
        num_hints = config["hint_count"]
        
        for _ in range(num_hints):
            # 힌트 단어 역시 의미 없는 4~6자 문자열
            test_word = self.generate_random_string(rng, rng.randint(4, 6))
            temp = test_word
            for stage in config["cipher_stack"]:
                if stage == "vigenere": temp = self.cipher.vigenere_encrypt(temp, keyword)
                elif stage == "playfair":
                    # 분기 논리: 로그 ID가 짝수면 키워드 정방향, 홀수면 역방향 사용
                    actual_key = keyword if log_id % 2 == 0 else keyword[::-1]
                    temp = self.cipher.playfair_encrypt(temp, actual_key)
                elif stage == "transposition": temp = self.cipher.columnar_transpose(temp, keyword)
                elif stage == "reverse": temp = temp[::-1]
                elif stage == "substitution": temp = self.cipher.substitution_encrypt(temp, keyword)
            hint_examples.append(f"  - {test_word} -> {temp}")

        # 4. 지문 구성
        kw_logic = config["keyword_logic"]
        if kw_logic == "direct":
            kw_instruction = f"암호화 키워드는 '{keyword}'입니다."
        elif kw_logic == "positional":
            # 문장 부호 제거 후 단어 리스트 생성
            clean_log = log_text.replace(".", "").replace(":", "").replace(",", "")
            words = clean_log.split()
            try:
                pos = words.index(keyword) + 1
                kw_instruction = f"암호화 키워드는 아래 로그 지문의 {pos}번째 단어입니다. (문장 부호 제외)"
            except ValueError:
                kw_instruction = f"암호화 키워드는 '{keyword}'입니다."
        else: # extraction (Extreme)
            kw_instruction = "암호화 키워드는 로그 지문 내에 숨겨져 있습니다. 'PRIMARY KEY', 'AUTH CODE', 'SEED', 'VECTOR' 등의 라벨 다음에 오는 단어가 키워드입니다."

        stack_desc = " -> ".join([s.upper() for s in config["cipher_stack"]])
        
        # 조건부 로직 설명 추가
        logic_hint = ""
        if "playfair" in config["cipher_stack"]:
            logic_hint = "\n[SPECIAL RULE]: Playfair 알고리즘 사용 시, 로그 ID(숫자)가 홀수이면 키워드를 역순(Reverse)으로 뒤집어서 사용하고, 짝수이면 그대로 사용합니다."

        # 알고리즘 상세 가이드 추가 (하위 난이도용)
        algo_details = ""
        if config["name"] in ["EASY", "MEDIUM"]:
            details = []
            for s in config["cipher_stack"]:
                if s == "substitution":
                    details.append("- SUBSTITUTION: 키워드의 고유 문자를 선두에 두고, 나머지 알파벳을 순서대로 채워 26자의 대응표를 만듭니다.")
                elif s == "vigenere":
                    details.append("- VIGENERE: 키워드를 반복하여 평문과 더합니다 (A=0, B=1...).")
                elif s == "reverse":
                    details.append("- REVERSE: 문자열의 순서를 뒤집습니다.")
            if details:
                algo_details = "\n알고리즘 상세 설명:\n" + "\n".join(details) + "\n"

        problem_text = f"--- [RECOVERED MISSION LOG] ---\n{log_text}\n-------------------------------\n\n"
        problem_text += f"암호문: '{encrypted}'\n\n"
        problem_text += f"암호화 가이드:\n1. {kw_instruction}\n2. 적용된 알고리즘: {stack_desc}{logic_hint}\n{algo_details}"
        
        if hint_examples:
            problem_text += "\n동일한 키워드와 알고리즘으로 암호화된 예제:\n" + "\n".join(hint_examples) + "\n"
        
        problem_text += "\n복호화된 원문을 입력하세요 (대문자, 공백 없음)."

        return {
            "difficulty": config["name"],
            "problem": problem_text,
            "answer": answer.replace(" ", ""),
            "solution": f"Steps: {' -> '.join(process_log)} | Answer: {answer}"
        }

# ============================================================================
# 데이터셋 생성 및 저장
# ============================================================================

def create_advanced_dataset(num_per_level: int = 2):
    """자가완결형 고난도 암호 데이터셋 생성"""
    import pandas as pd
    import json

    print(f"Algorithm-focused Cypher 문제 생성 중...")
    print(f"난이도별 {num_per_level}개씩 생성")
    print("="*70)

    generator = SelfContainedCipherGenerator()
    all_problems = []

    for level_key in sorted(DIFFICULTY_CONFIG.keys()):
        config = DIFFICULTY_CONFIG[level_key]
        difficulty = config["name"]

        print(f"\n[{difficulty}] {config['description']}")

        for i in range(num_per_level):
            # 시드 설정: 4000번대 사용
            seed = 4000 + len(all_problems)
            problem = generator.generate_problem(config, seed)

            all_problems.append({
                "question": problem["problem"],
                "answer": problem["answer"],
                "solution": problem["solution"],
                "difficulty": difficulty,
                "description": config["description"]
            })

            print(f"  {i+1}. {problem['answer'][:15]}... 생성 완료")

    # DataFrame 생성
    df = pd.DataFrame(all_problems)

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # CSV 저장
    csv_dir = PROJECT_ROOT / "data" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"cipher.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # JSONL 저장
    json_dir = PROJECT_ROOT / "data" / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = json_dir / f"cipher.jsonl"

    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for problem in all_problems:
            f.write(json.dumps(problem, ensure_ascii=False) + '\n')

    print(f"\n{'='*70}")
    print(f"생성 완료:")
    print(f"  총 문제 수: {len(all_problems)}개")
    print(f"  CSV: {csv_path}")
    print(f"  JSONL: {jsonl_path}")
    print(f"{'='*70}")

    return df

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate Cipher Puzzles')
    parser.add_argument('--num', type=int, default=2, help='Number of puzzles per difficulty level')
    args = parser.parse_args()
    
    # 각 난이도별 n개씩 생성
    create_advanced_dataset(num_per_level=args.num)
