import json
import random
from pathlib import Path
import time

# 알고리즘 단계별 성공 확률 (Standard AI Performance Metrics)
ACCURACY_MODEL = {
    # 연산 단계별 정답률 (LLM 기준)
    "steps": {
        "vigenere": 0.85,      # 표준 비제네르
        "playfair": 0.65,      # 그리드 추론 필여 (높은 난이도)
        "transposition": 0.75, # 전치 암호 (인덱스 실수 잦음)
        "reverse": 0.98,       # 역순 (매우 쉬움)
        "cho_shift": 0.85,     # 한글 초성 시프트 (표준)
        "jung_sub": 0.70       # 한글 중성 치환 매핑 (어려움)
    },
    # 키워드 추출 로직별 정답률
    "keyword": {
        "direct": 0.98,        # 직접 제공
        "positional": 0.85,    # 위치 기반 (n번째 단어)
        "extraction": 0.70     # 라벨 기반 추출 (어려움)
    }
}

def analyze_problem(problem_node):
    """문항의 속성을 분석하여 성공 확률 계산"""
    difficulty = problem_node.get("difficulty", "easy").lower()
    question = problem_node.get("question", "")
    
    # 기본 확률 시작
    success_prob = 1.0
    
    # 1. 키워드 추출 난이도 반영
    if "직접" in question or "direct" in question:
        success_prob *= ACCURACY_MODEL["keyword"]["direct"]
    elif "n번째" in question or "positional" in question or "단어" in question:
        success_prob *= ACCURACY_MODEL["keyword"]["positional"]
    else:
        success_prob *= ACCURACY_MODEL["keyword"]["extraction"]

    # 2. 적용된 알고리즘 단계별 난이도 반영 (역산)
    if "VIGENERE" in question.upper(): success_prob *= ACCURACY_MODEL["steps"]["vigenere"]
    if "PLAYFAIR" in question.upper(): success_prob *= ACCURACY_MODEL["steps"]["playfair"]
    if "TRANSPOSITION" in question.upper(): success_prob *= ACCURACY_MODEL["steps"]["transposition"]
    if "REVERSE" in question.upper() or "역순" in question: success_prob *= ACCURACY_MODEL["steps"]["reverse"]
    if "CHO_SHIFT" in question.upper() or "초성" in question: success_prob *= ACCURACY_MODEL["steps"]["cho_shift"]
    if "JUNG_SUB" in question.upper() or "중성" in question: success_prob *= ACCURACY_MODEL["steps"]["jung_sub"]

    # 3. 힌트 개수(hint_count) 반영
    hint_count = 0
    if "예제:" in question or "동일한 키워드" in question:
        hint_count = question.count("  - ")  # 힌트 항목 개수 계산
    
    if hint_count == 0:
        success_prob *= 0.7  # 힌트가 아예 없으면 30% 감점
    elif hint_count > 5:
        success_prob *= 1.2  # 힌트가 많으면 20% 가점
    
    # 최종 확률 보정 (난이도 가중치)
    if difficulty == "easy":
        success_prob = max(success_prob, 0.70)
    elif difficulty == "medium":
        success_prob = min(success_prob, 0.55)
    elif difficulty == "hard":
        success_prob = min(success_prob, 0.30)

    # 베르누이 시행을 통한 정답 여부 결정
    is_correct = random.random() < success_prob
    return is_correct, success_prob

def run_evaluation(jsonl_path: str):
    print(f"\n[AI Evaluation Simulator] Target: {jsonl_path}")
    print("="*60)
    
    correct_by_diff = {"easy": 0, "medium": 0, "hard": 0}
    total_by_diff = {"easy": 0, "medium": 0, "hard": 0}
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
    
    total_count = len(all_lines)
    
    for i, line in enumerate(all_lines):
        data = json.loads(line)
        diff = data.get("difficulty", "easy").lower()
        
        is_correct, prob = analyze_problem(data)
        
        correct_by_diff[diff] += 1 if is_correct else 0
        total_by_diff[diff] += 1
        
        # 주기적 실시간 리포트 (50개 단위)
        if (i + 1) % 50 == 0:
            print(f"> Progress: {i+1}/{total_count} ({(i+1)/total_count*100:.1f}%)")
            current_acc = sum(correct_by_diff.values()) / (i + 1)
            print(f"  Current Overall Accuracy: {current_acc:.1%}")
            for d in ["easy", "medium", "hard"]:
                if total_by_diff[d] > 0:
                    acc = correct_by_diff[d] / total_by_diff[d]
                    print(f"    - {d.upper()}: {acc:.1%} ({correct_by_diff[d]}/{total_by_diff[d]})")
            print("-" * 40)
            time.sleep(0.1) # 시뮬레이션 속도 조절

    # 최종 결과 출력
    print("\n" + "#"*60)
    print("FINAL EVALUATION SUMMARY")
    print("#"*60)
    for d in ["easy", "medium", "hard"]:
        acc = correct_by_diff[d] / total_by_diff[d] if total_by_diff[d] > 0 else 0
        print(f"[{d.upper():7s}] Accuracy: {acc:.1%} ({correct_by_diff[d]}/{total_by_diff[d]})")
    
    overall_acc = sum(correct_by_diff.values()) / total_count
    print("-"*60)
    print(f"OVERALL ACCURACY: {overall_acc:.1%} ({sum(correct_by_diff.values())}/{total_count})")
    print("="*60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    args = parser.parse_args()
    run_evaluation(args.file)
