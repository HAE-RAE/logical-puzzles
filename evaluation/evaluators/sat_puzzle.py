"""
SAT Puzzle Evaluator

SAT 논리 퍼즐 평가 (영문)
"""

import logging
import json
import re
import time
from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING

from ..core.base import BaseEvaluator, EvaluationResult
from ..task_names import locale_from_task_name

if TYPE_CHECKING:
    from ..model.base import BaseLLMClient

logger = logging.getLogger(__name__)


class SATPuzzleEvaluator(BaseEvaluator):
    """
    SAT Puzzle 퍼즐 평가자
    
    답변 형식: JSON (변수: bool 매핑, 부분 점수 지원)
    """
    
    SYSTEM_PROMPT = """### Instructions
You are an expert at solving logic puzzles.

### Rules
Carefully analyze all constraints and provide accurate answers.

### Output format
Your final line MUST be exactly one-line JSON:
Answer: {"K team": false, "L team": true}

Use lowercase boolean literals (`true` / `false`) in the final JSON line.
Do not use markdown code blocks in the final answer."""

    KOREAN_SYSTEM_PROMPT = """### 지시사항
당신은 논리 퍼즐 전문가입니다.

### 규칙
주어진 모든 제약을 꼼꼼히 분석하고 변수별 참/거짓을 정확한 답(JSON)으로 제시하세요.

### 출력 형식
마지막 줄은 반드시 한 줄 JSON으로 작성하세요:
Answer: {"K팀": false, "L팀": true}

최종 JSON의 불리언은 소문자(`true` / `false`)를 사용하세요.
최종 답에 markdown 코드블록은 사용하지 마세요."""

    def _is_korean(self, puzzle: Optional[Dict] = None) -> bool:
        """task_name에 sat_puzzles_ko_easy 등 포함 시 한국어; question에서도 추론."""
        task = getattr(self, "_task_name", None) or ""
        hint = locale_from_task_name(task)
        if hint is not None:
            return hint
        if puzzle is not None:
            q = str(puzzle.get("question", ""))
            if re.search(r"[가-힣]", q):
                return True
            expected = puzzle.get("answer", "")
            return bool(re.search(r"[가-힣]", str(expected)))
        return False

    def _get_system_prompt(self, puzzle: Dict) -> str:
        if self._is_korean(puzzle):
            return self.KOREAN_SYSTEM_PROMPT
        return self.SYSTEM_PROMPT

    def _evaluate_single(
        self,
        puzzle: Dict[str, Any],
        llm_client: "BaseLLMClient",
    ) -> EvaluationResult:
        system_prompt = self._get_system_prompt(puzzle)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": puzzle["question"]},
        ]
        start = time.time()
        try:
            response, usage = llm_client.generate(messages)
            latency = (time.time() - start) * 1000
            return self._process_response(puzzle, response, latency, usage)
        except Exception as e:
            latency = (time.time() - start) * 1000
            return self._process_response(puzzle, "", latency, {"error": str(e)})

    async def _evaluate_async(
        self,
        puzzles: List[Dict[str, Any]],
        llm_client: "BaseLLMClient",
        verbose: bool = True,
        max_concurrent: int = 10,
    ) -> List[EvaluationResult]:
        from ..core.base import logger

        messages_list = []
        for puzzle in puzzles:
            system_prompt = self._get_system_prompt(puzzle)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": puzzle["question"]},
            ]
            messages_list.append(messages)

        total_puzzles = len(puzzles)
        task_name = getattr(self, "_task_name", None)
        task_prefix = f"[{task_name}] " if task_name else ""

        if verbose:
            logger.info(
                f"{task_prefix}Starting async evaluation: {total_puzzles} puzzles, "
                f"max_concurrent={max_concurrent}"
            )

        start_time = time.time()

        def progress_callback(completed, total):
            if verbose:
                percentage = (completed / total) * 100
                if completed % max(1, total // 10) == 0 or completed == total:
                    logger.info(
                        f"{task_prefix}API calls progress: {completed}/{total} ({percentage:.0f}%)"
                    )

        responses = await llm_client.async_batch_generate(
            messages_list,
            max_concurrent=max_concurrent,
            progress_callback=progress_callback if verbose else None,
        )
        total_latency = (time.time() - start_time) * 1000

        if verbose:
            logger.info(
                f"{task_prefix}API calls completed: {total_puzzles}/{total_puzzles} in "
                f"{total_latency:.0f}ms ({total_latency/total_puzzles:.0f}ms per puzzle)"
            )

        results = []
        correct_count = 0
        error_count = 0

        for puzzle, (response, usage) in zip(puzzles, responses):
            latency_ms = usage.get("latency_ms", 0)
            result = self._process_response(puzzle, response, latency_ms, usage)
            if result.correct:
                correct_count += 1
            if result.error:
                error_count += 1
            results.append(result)

        if verbose:
            incorrect_count = total_puzzles - correct_count - error_count
            logger.info(
                f"Processing completed: {correct_count} correct, {incorrect_count} incorrect, "
                f"{error_count} errors"
            )

        return results

    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[Dict[str, bool]]:
        """
        LLM 응답에서 JSON 답변 추출
        
        Args:
            response: LLM 응답 텍스트
            puzzle: 퍼즐 데이터
        """
        variables = puzzle.get("variables", [])
        answer_text = self._extract_final_answer_text(response, allow_boxed_fallback=False)

        def _validate(answer_obj: Any) -> Optional[Dict[str, bool]]:
            if not isinstance(answer_obj, dict):
                return None
            for var in variables:
                if var not in answer_obj or not isinstance(answer_obj[var], bool):
                    return None
            return answer_obj

        def _parse_json_like(text: str) -> Optional[Dict[str, bool]]:
            if not text:
                return None
            candidate = text.strip()
            candidate = re.sub(r'//.*', '', candidate)
            # Accept Python-style bools too: True/False -> true/false
            candidate = re.sub(r'\bTrue\b', 'true', candidate)
            candidate = re.sub(r'\bFalse\b', 'false', candidate)
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                return None
            return _validate(parsed)
        
        try:
            if answer_text and answer_text.strip().startswith("{"):
                parsed = _parse_json_like(answer_text)
                if parsed is not None:
                    return parsed

            # Handle multiline style:
            # Answer:
            # ```json
            # { ... }
            # ```
            answer_block = re.search(
                r'(?is)answer\s*[:：]\s*```(?:json)?\s*(\{.*?\})\s*```',
                response,
            )
            if answer_block:
                parsed = _parse_json_like(answer_block.group(1))
                if parsed is not None:
                    return parsed

            # Handle multiline style without fences:
            # Answer:
            # { ... }
            answer_multiline = re.search(
                r'(?is)answer\s*[:：]\s*(\{.*?\})',
                response,
            )
            if answer_multiline:
                parsed = _parse_json_like(answer_multiline.group(1))
                if parsed is not None:
                    return parsed

            # Try to find JSON in the response
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                parsed = _parse_json_like(json_match.group(1))
                if parsed is not None:
                    return parsed
            
            # Try to find JSON without markdown
            json_match = re.search(r'\{[^{}]*"[^"]+"\s*:\s*(?:true|false|True|False)[^{}]*\}', response, re.DOTALL)
            if json_match:
                parsed = _parse_json_like(json_match.group(0))
                if parsed is not None:
                    return parsed
            
            # Try to parse line-by-line format
            # "Alice: True" or "Alice: False"
            answer = {}
            for line in response.split('\n'):
                for var in variables:
                    if var in line:
                        if 'true' in line.lower() or ': true' in line.lower():
                            answer[var] = True
                            break
                        elif 'false' in line.lower() or ': false' in line.lower():
                            answer[var] = False
                            break
            
            # Check if we got all variables
            if len(answer) == len(variables) and all(var in answer for var in variables):
                return answer
            
            return None
        
        except (json.JSONDecodeError, AttributeError):
            return None
    
    def _check_answer(
        self,
        expected: Dict[str, bool],
        predicted: Optional[Dict[str, bool]]
    ) -> Tuple[bool, float]:
        """
        답변 확인
        
        Returns:
            (is_correct, partial_score) 튜플
        """
        if predicted is None:
            return False, 0.0
        
        # 완전 일치만 정답으로 인정
        for var in expected:
            if var not in predicted or predicted[var] != expected[var]:
                return False, 0.0
        
        return True, 1.0
