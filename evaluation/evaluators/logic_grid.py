"""
Logic Grid Evaluator

논리 그리드 퍼즐 평가 (영문)
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


class LogicGridEvaluator(BaseEvaluator):
    """
    Logic Grid 퍼즐 평가자
    
    답변 형식: JSON (부분 점수 지원)
    """
    
    SYSTEM_PROMPT = """### Instructions
You are an expert at solving logic puzzles.
Carefully analyze all constraints and provide accurate answers.

### Rules
1. You may provide brief reasoning first.
2. The final line MUST be exactly in this format: Answer: {JSON object}
3. The JSON object must follow the schema required in the user message.
4. Use a single-line valid JSON object (double quotes, no trailing commas).
5. Do NOT wrap JSON with markdown code fences.
6. Do NOT add any text after the final `Answer:` line.

### Output format
Solve the puzzle step by step, then end your response with exactly one line:
Answer: {"Alice":{"Pet":"Cat","Drink":"Tea"},"Bob":{"Pet":"Dog","Drink":"Coffee"}}"""

    KOREAN_SYSTEM_PROMPT = """### 지시사항
당신은 논리 퍼즐 전문가입니다.
주어진 모든 단서를 꼼꼼히 분석하고 정확한 답(JSON)을 제시하세요.

### 규칙
1. 필요하면 간단한 풀이를 먼저 작성해도 됩니다.
2. 마지막 줄은 반드시 다음 형식을 정확히 지키세요: Answer: {JSON 객체}
3. JSON 객체는 사용자 메시지의 스키마를 정확히 따라야 합니다.
4. JSON은 한 줄의 유효한 JSON이어야 합니다(쌍따옴표, trailing comma 금지).
5. JSON을 마크다운 코드펜스로 감싸지 마세요.
6. 마지막 `Answer:` 줄 뒤에 추가 텍스트를 쓰지 마세요.

### 출력 형식
마지막 줄은 반드시 아래 형식으로 작성하세요:
Answer: {"민수":{"애완동물":"고양이","음료":"차"},"지훈":{"애완동물":"강아지","음료":"커피"}}"""

    def _is_korean(self, puzzle: Optional[Dict] = None) -> bool:
        """task_name에 logic_grid_ko_easy 등 포함 시 한국어; question/answer에서도 추론."""
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

    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[Dict[str, Dict[str, str]]]:
        """
        LLM 응답에서 JSON 답변 추출
        
        Args:
            response: LLM 응답 텍스트
            puzzle: 퍼즐 데이터
        """
        people = puzzle.get("people", [])
        categories = list(puzzle.get("attributes", {}).keys())
        answer_text = self._extract_final_answer_text(response, allow_boxed_fallback=False)
        
        try:
            if answer_text and answer_text.strip().startswith("{"):
                answer = json.loads(answer_text.strip())
                if isinstance(answer, dict):
                    valid = True
                    for person in people:
                        if person not in answer or not isinstance(answer[person], dict):
                            valid = False
                            break
                        for cat in categories:
                            if cat not in answer[person]:
                                valid = False
                                break
                    if valid:
                        return answer

            # Try to find JSON in the response
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                answer_json = json_match.group(1)
                answer = json.loads(answer_json)
                
                # Validate structure
                if not isinstance(answer, dict):
                    return None
                
                # Check all people are present
                for person in people:
                    if person not in answer:
                        return None
                    if not isinstance(answer[person], dict):
                        return None
                    
                    # Check all categories are present
                    for cat in categories:
                        if cat not in answer[person]:
                            return None
                
                return answer
            
            # Try to find JSON without markdown
            json_match = re.search(r'\{[^{}]*"[^"]+"\s*:\s*\{[^{}]+\}[^{}]*\}', response, re.DOTALL)
            if json_match:
                answer = json.loads(json_match.group(0))
                
                # Validate
                if isinstance(answer, dict):
                    valid = True
                    for person in people:
                        if person not in answer or not isinstance(answer[person], dict):
                            valid = False
                            break
                        for cat in categories:
                            if cat not in answer[person]:
                                valid = False
                                break
                    
                    if valid:
                        return answer
            
            return None
        
        except (json.JSONDecodeError, AttributeError):
            return None
    
    def _check_answer(
        self,
        expected: Dict[str, Dict[str, str]],
        predicted: Optional[Dict[str, Dict[str, str]]]
    ) -> Tuple[bool, float]:
        """
        답변 확인
        
        Returns:
            (is_correct, partial_score) 튜플
        """
        if predicted is None:
            return False, 0.0
        
        # 완전 일치만 정답으로 인정
        for person, attrs in expected.items():
            if person not in predicted:
                return False, 0.0
            
            for cat, val in attrs.items():
                if cat not in predicted[person] or predicted[person][cat] != val:
                    return False, 0.0
        
        return True, 1.0
