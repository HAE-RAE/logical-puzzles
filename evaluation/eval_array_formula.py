"""
Array Formula Puzzle Evaluator
LLM을 이용한 엑셀 배열 수식 퍼즐 평가

평가 메트릭:
- Exact Match: 정확히 일치하는 비율
- Numeric Accuracy: 숫자 정답의 허용 오차 내 정확도
- Type Accuracy: 문제 유형별 정확도
"""

import json
import os
import re
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict

# API 클라이언트 (선택적 import)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


@dataclass
class EvaluationResult:
    """단일 문제 평가 결과"""
    puzzle_id: str
    difficulty: str
    problem_type: str
    question: str
    expected_answer: Any
    model_answer: str
    parsed_answer: Any
    is_correct: bool
    is_numeric_close: bool  # 숫자의 경우 허용 오차 내 일치
    error: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class EvaluationSummary:
    """전체 평가 요약"""
    model: str
    total_problems: int
    correct_count: int
    accuracy: float
    numeric_close_accuracy: float
    by_difficulty: Dict[str, Dict[str, float]]
    by_type: Dict[str, Dict[str, float]]
    avg_latency_ms: float
    timestamp: str


# ============================================================
# 프롬프트 생성
# ============================================================

SYSTEM_PROMPT = """당신은 엑셀/스프레드시트 전문가입니다. 
주어진 테이블 데이터를 분석하고 질문에 정확하게 답해야 합니다.

규칙:
1. 계산 결과가 숫자인 경우, 숫자만 답하세요 (단위, 쉼표, 원 등 제외)
2. 소수점이 나오면 문제에서 별도 지시가 없으면 소수점 이하 버림
3. 텍스트 답변은 정확한 값만 답하세요
4. 추론 과정을 간략히 설명한 후 "최종 답: [답]" 형식으로 마무리하세요
"""


def format_table_for_prompt(table_name: str, table_data: Dict) -> str:
    """테이블을 마크다운 형식으로 포맷팅"""
    columns = table_data["columns"]
    data = table_data["data"]
    
    lines = [f"### {table_name} 테이블"]
    
    # 마크다운 테이블 헤더
    header = "| " + " | ".join(str(col) for col in columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    lines.append(header)
    lines.append(separator)
    
    # 데이터 행
    for row in data:
        row_str = "| " + " | ".join(str(row.get(col, "")) for col in columns) + " |"
        lines.append(row_str)
    
    return "\n".join(lines)


def puzzle_to_prompt(puzzle: Dict[str, Any], include_hint: bool = False) -> str:
    """퍼즐을 LLM 프롬프트로 변환"""
    prompt_parts = []
    
    prompt_parts.append("다음 스프레드시트 데이터를 분석하세요.\n")
    
    # 테이블 출력
    for table_name, table_data in puzzle["tables"].items():
        prompt_parts.append(format_table_for_prompt(table_name, table_data))
        prompt_parts.append("")
    
    # 질문
    prompt_parts.append(f"**질문**: {puzzle['question']}")
    
    # 힌트 (평가 시에는 보통 제외)
    if include_hint and "formula_hint" in puzzle:
        prompt_parts.append(f"**힌트**: {puzzle['formula_hint']}")
    
    # 응답 형식 지시
    prompt_parts.append("\n계산 과정을 간략히 설명하고, 마지막에 '최종 답: [답]' 형식으로 답하세요.")
    if puzzle.get("answer_type") == "number":
        prompt_parts.append("(숫자만 답하세요. 단위나 쉼표 없이)")
    
    return "\n".join(prompt_parts)


# ============================================================
# 답변 파싱
# ============================================================

def parse_answer(response: str, answer_type: str = "number") -> Any:
    """
    LLM 응답에서 최종 답변 추출
    
    Args:
        response: LLM 응답 전체
        answer_type: "number" 또는 "text"
    
    Returns:
        파싱된 답변
    """
    # "최종 답:" 패턴 찾기
    patterns = [
        r"최종\s*답\s*[:：]\s*(.+?)(?:\n|$)",
        r"정답\s*[:：]\s*(.+?)(?:\n|$)",
        r"답\s*[:：]\s*(.+?)(?:\n|$)",
        r"Answer\s*[:：]\s*(.+?)(?:\n|$)",
    ]
    
    answer_text = None
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            answer_text = match.group(1).strip()
            break
    
    # 패턴을 못 찾으면 마지막 줄에서 추출 시도
    if answer_text is None:
        lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
        if lines:
            answer_text = lines[-1]
    
    if answer_text is None:
        return None
    
    # 숫자 타입 처리
    if answer_type == "number":
        # 숫자만 추출 (음수, 소수점 포함)
        number_match = re.search(r"-?[\d,]+\.?\d*", answer_text.replace(",", ""))
        if number_match:
            try:
                num_str = number_match.group().replace(",", "")
                if "." in num_str:
                    return float(num_str)
                return int(num_str)
            except ValueError:
                pass
        return None
    
    # 텍스트 타입
    # 따옴표 제거
    answer_text = answer_text.strip("'\"")
    return answer_text


def check_answer(
    expected: Any,
    parsed: Any,
    answer_type: str = "number",
    tolerance: float = 0.01
) -> Tuple[bool, bool]:
    """
    답변 정확성 검사
    
    Returns:
        (exact_match, numeric_close)
    """
    if parsed is None:
        return False, False
    
    if answer_type == "number":
        try:
            expected_num = float(expected)
            parsed_num = float(parsed)
            
            exact = abs(expected_num - parsed_num) < 0.001
            close = abs(expected_num - parsed_num) / max(abs(expected_num), 1) < tolerance
            
            return exact, close
        except (ValueError, TypeError):
            return False, False
    else:
        # 텍스트 비교
        expected_str = str(expected).strip().lower()
        parsed_str = str(parsed).strip().lower()
        exact = expected_str == parsed_str
        return exact, exact


# ============================================================
# LLM API 호출
# ============================================================

class LLMClient:
    """LLM API 클라이언트 래퍼"""
    
    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key
        
        if model.startswith("gpt") or model.startswith("o1"):
            if not OPENAI_AVAILABLE:
                raise ImportError("openai 패키지가 필요합니다: pip install openai")
            self.client_type = "openai"
            self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        elif model.startswith("claude"):
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("anthropic 패키지가 필요합니다: pip install anthropic")
            self.client_type = "anthropic"
            self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        else:
            raise ValueError(f"지원하지 않는 모델: {model}")
    
    def generate(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        """프롬프트에 대한 응답 생성"""
        if self.client_type == "openai":
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # o1 모델은 system 메시지 지원 안 함
            if self.model.startswith("o1"):
                messages = [{"role": "user", "content": f"{system_prompt}\n\n{prompt}"}]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                max_tokens=2000
            )
            return response.choices[0].message.content
            
        elif self.client_type == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        
        return ""


# ============================================================
# 평가 실행
# ============================================================

def evaluate_puzzle(
    puzzle: Dict[str, Any],
    client: LLMClient,
    include_hint: bool = False
) -> EvaluationResult:
    """단일 퍼즐 평가"""
    prompt = puzzle_to_prompt(puzzle, include_hint=include_hint)
    
    start_time = time.time()
    try:
        response = client.generate(prompt)
        latency_ms = (time.time() - start_time) * 1000
        
        parsed = parse_answer(response, puzzle.get("answer_type", "number"))
        exact, close = check_answer(
            puzzle["answer"],
            parsed,
            puzzle.get("answer_type", "number")
        )
        
        return EvaluationResult(
            puzzle_id=puzzle["id"],
            difficulty=puzzle["difficulty"],
            problem_type=puzzle["type"],
            question=puzzle["question"],
            expected_answer=puzzle["answer"],
            model_answer=response,
            parsed_answer=parsed,
            is_correct=exact,
            is_numeric_close=close,
            latency_ms=latency_ms
        )
        
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        return EvaluationResult(
            puzzle_id=puzzle["id"],
            difficulty=puzzle["difficulty"],
            problem_type=puzzle["type"],
            question=puzzle["question"],
            expected_answer=puzzle["answer"],
            model_answer="",
            parsed_answer=None,
            is_correct=False,
            is_numeric_close=False,
            error=str(e),
            latency_ms=latency_ms
        )


def evaluate_dataset(
    puzzles: List[Dict[str, Any]],
    client: LLMClient,
    include_hint: bool = False,
    verbose: bool = True,
    delay: float = 0.5
) -> Tuple[List[EvaluationResult], EvaluationSummary]:
    """전체 데이터셋 평가"""
    results = []
    
    for i, puzzle in enumerate(puzzles):
        if verbose:
            print(f"[{i+1}/{len(puzzles)}] Evaluating {puzzle['id']}...", end=" ")
        
        result = evaluate_puzzle(puzzle, client, include_hint)
        results.append(result)
        
        if verbose:
            status = "✓" if result.is_correct else ("≈" if result.is_numeric_close else "✗")
            print(f"{status} (expected: {result.expected_answer}, got: {result.parsed_answer})")
        
        if delay > 0 and i < len(puzzles) - 1:
            time.sleep(delay)
    
    # 요약 계산
    summary = calculate_summary(results, client.model)
    
    return results, summary


def calculate_summary(results: List[EvaluationResult], model: str) -> EvaluationSummary:
    """평가 결과 요약 계산"""
    total = len(results)
    correct = sum(1 for r in results if r.is_correct)
    numeric_close = sum(1 for r in results if r.is_numeric_close)
    avg_latency = sum(r.latency_ms for r in results) / total if total > 0 else 0
    
    # 난이도별 통계
    by_difficulty = {}
    for diff in ["easy", "medium", "hard"]:
        diff_results = [r for r in results if r.difficulty == diff]
        if diff_results:
            by_difficulty[diff] = {
                "total": len(diff_results),
                "correct": sum(1 for r in diff_results if r.is_correct),
                "accuracy": sum(1 for r in diff_results if r.is_correct) / len(diff_results)
            }
    
    # 유형별 통계
    by_type = {}
    types = set(r.problem_type for r in results)
    for ptype in types:
        type_results = [r for r in results if r.problem_type == ptype]
        if type_results:
            by_type[ptype] = {
                "total": len(type_results),
                "correct": sum(1 for r in type_results if r.is_correct),
                "accuracy": sum(1 for r in type_results if r.is_correct) / len(type_results)
            }
    
    return EvaluationSummary(
        model=model,
        total_problems=total,
        correct_count=correct,
        accuracy=correct / total if total > 0 else 0,
        numeric_close_accuracy=numeric_close / total if total > 0 else 0,
        by_difficulty=by_difficulty,
        by_type=by_type,
        avg_latency_ms=avg_latency,
        timestamp=datetime.now().isoformat()
    )


def print_summary(summary: EvaluationSummary):
    """평가 요약 출력"""
    print("\n" + "=" * 60)
    print(f"평가 결과 요약 - {summary.model}")
    print("=" * 60)
    print(f"전체 정확도: {summary.correct_count}/{summary.total_problems} ({summary.accuracy:.1%})")
    print(f"수치 근접 정확도: {summary.numeric_close_accuracy:.1%}")
    print(f"평균 응답 시간: {summary.avg_latency_ms:.0f}ms")
    
    print("\n난이도별 정확도:")
    for diff, stats in summary.by_difficulty.items():
        print(f"  {diff}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.1%})")
    
    print("\n유형별 정확도:")
    for ptype, stats in summary.by_type.items():
        print(f"  {ptype}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.1%})")
    
    print("=" * 60)


def save_results(
    results: List[EvaluationResult],
    summary: EvaluationSummary,
    output_dir: str = "../evaluation_data/array_formula"
):
    """평가 결과 저장"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = summary.model.replace("/", "_").replace(":", "_")
    
    # 상세 결과 저장
    results_file = output_path / f"results_{model_safe}_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)
    
    # 요약 저장
    summary_file = output_path / f"summary_{model_safe}_{timestamp}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, ensure_ascii=False, indent=2)
    
    print(f"\n결과 저장: {results_file}")
    print(f"요약 저장: {summary_file}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Array Formula 퍼즐 평가기")
    parser.add_argument("--model", type=str, default="gpt-4o", help="평가할 모델")
    parser.add_argument("--dataset", type=str, default="../evaluation_data/array_formula/array_formula_dataset.json", help="데이터셋 경로")
    parser.add_argument("--output", type=str, default="../evaluation_data/array_formula", help="결과 저장 디렉토리")
    parser.add_argument("--limit", type=int, default=None, help="평가할 문제 수 제한")
    parser.add_argument("--difficulty", type=str, default=None, help="특정 난이도만 평가")
    parser.add_argument("--type", type=str, default=None, help="특정 유형만 평가")
    parser.add_argument("--hint", action="store_true", help="힌트 포함")
    parser.add_argument("--delay", type=float, default=0.5, help="API 호출 간 딜레이(초)")
    parser.add_argument("--quiet", action="store_true", help="상세 출력 끄기")
    
    args = parser.parse_args()
    
    # 데이터셋 로드
    print(f"데이터셋 로드: {args.dataset}")
    with open(args.dataset, "r", encoding="utf-8") as f:
        puzzles = json.load(f)
    
    # 필터링
    if args.difficulty:
        puzzles = [p for p in puzzles if p["difficulty"] == args.difficulty]
    if args.type:
        puzzles = [p for p in puzzles if p["type"] == args.type]
    if args.limit:
        puzzles = puzzles[:args.limit]
    
    print(f"평가할 문제 수: {len(puzzles)}")
    
    # 클라이언트 초기화
    client = LLMClient(args.model)
    
    # 평가 실행
    results, summary = evaluate_dataset(
        puzzles,
        client,
        include_hint=args.hint,
        verbose=not args.quiet,
        delay=args.delay
    )
    
    # 결과 출력 및 저장
    print_summary(summary)
    save_results(results, summary, args.output)


if __name__ == "__main__":
    main()
