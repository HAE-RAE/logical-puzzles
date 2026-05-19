"""
Hanoi Evaluator

하노이 탑 퍼즐 평가
"""

import logging
import re
import time
from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING

from ..core.base import BaseEvaluator, EvaluationResult
from ..task_names import locale_from_task_name

if TYPE_CHECKING:
    from ..model.base import BaseLLMClient

logger = logging.getLogger(__name__)


class HanoiEvaluator(BaseEvaluator):
    """
    Hanoi 퍼즐 평가자
    
    답변 형식: (disk, from, to) 튜플
    """

    _TUPLE_RE = re.compile(r"\((\d+),\s*(\d+),\s*(\d+)\)")
    _COMMA_TRIPLE_RE = re.compile(r"(\d+)\s*,\s*(\d+)\s*,\s*(\d+)")

    SYSTEM_PROMPT = """### Instructions
You are an expert at Tower of Hanoi puzzles and related counting questions.

### Rules
1. Follow the puzzle and peg labels given in the user message.
2. Always end with exactly one Answer: line containing a 3-integer tuple that matches
   the format requested in the user message. Examples by query type:
   - move query          → Answer: (disk, from_peg, to_peg)
   - disk location       → Answer: (disk, peg, peg)
   - three peg locations → Answer: (peg_a, peg_b, peg_c)
   - disk count per peg  → Answer: (count_peg0, count_peg1, count_peg2)
   - total disk count    → Answer: (n, n, n)
3. Explain your reasoning clearly, then present your final answer in the format above.
   Write only one Answer: line with nothing after it.

### Output format
Your final line must be:
Answer: (a, b, c)
"""

    KOREAN_SYSTEM_PROMPT = """### 지시사항
당신은 하노이 탑 퍼즐과 관련 수치 문제를 정확히 푸는 전문가입니다.

### 규칙
1. 사용자 메시지의 퍼즐·기둥 표기를 그대로 따르세요.
2. 사용자 메시지에서 요청한 형식에 맞는 3정수 튜플로 Answer: 줄을 마무리하세요. 유형별 예시:
   - 이동 쿼리             → Answer: (원반, 출발기둥, 도착기둥)
   - 원판 위치             → Answer: (원판번호, 기둥, 기둥)
   - 세 원판 기둥          → Answer: (기둥a, 기둥b, 기둥c)
   - 기둥별 원판 수        → Answer: (기둥0_개수, 기둥1_개수, 기둥2_개수)
   - 전체 원판 수          → Answer: (n, n, n)
3. 풀이 과정을 명확히 서술한 뒤, 최종 결론을 아래 형식으로 제시하세요.
   Answer: 줄은 한 번만, 그 뒤에 아무것도 쓰지 마세요.

### 출력 형식
마지막 줄은 반드시 아래 형식으로 작성하세요:
Answer: (a, b, c)
"""

    def _is_korean(self, puzzle: Optional[Dict] = None) -> bool:
        """Prefer task_name (e.g. …_ko_easy); else infer from expected answer."""
        task = getattr(self, "_task_name", None) or ""
        hint = locale_from_task_name(task)
        if hint is not None:
            return hint
        if puzzle is not None:
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

    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[Tuple[int, int, int]]:
        """
        LLM 응답에서 (a, b, c) 튜플 추출.

        이전 구현은 본문에 등장하는 *첫* 세 숫자를 잡아 오탐이 잦았고, 첫 튜플만
        보면 중간 풀이의 (disk, from, to)를 답으로 오인할 수 있었다.
        우선순위: 라벨(Answer:/정답:) payload → 전체 응답에서 *마지막* (a,b,c) 매칭.
        """
        cleaned = self._strip_code_fences(response or "").strip()
        labeled = self._extract_final_answer_text(response, allow_boxed_fallback=True)
        search_segments: List[str] = []
        if labeled:
            search_segments.append(labeled.strip())
        if cleaned:
            search_segments.append(cleaned)

        for seg in search_segments:
            matches = list(self._TUPLE_RE.finditer(seg))
            if matches:
                m = matches[-1]
                return (int(m.group(1)), int(m.group(2)), int(m.group(3)))

        for seg in search_segments:
            matches = list(self._COMMA_TRIPLE_RE.finditer(seg))
            if matches:
                m = matches[-1]
                return (int(m.group(1)), int(m.group(2)), int(m.group(3)))

        # 단일 숫자 답변 보정: 역문제/총 이동 횟수 문제 (라벨 줄 또는 전체에서 마지막 단일 토큰)
        question_raw = str(puzzle.get("question", ""))
        question = question_raw.lower()
        for seg in search_segments:
            nums = re.findall(r"\d+", seg)
            if len(nums) != 1:
                continue
            v = int(nums[0])
            # Only trigger for inverse_find_n ("How many disks are IN this puzzle?"),
            # not for count_per_peg_after_continuation ("how many disks are ON each peg").
            if ("how many disks are in" in question or "원판이 몇 개 있" in question_raw) and \
                    "on peg" not in question and "each peg" not in question:
                return (v, v, v)
            m = re.search(r"disk\s*(\d+)", question, re.IGNORECASE)
            if "how many times" in question and m:
                k = int(m.group(1))
                return (k, v, v)
            m_ko = re.search(r"원반\s*(\d+)", question_raw)
            if "몇 번" in question_raw and m_ko:
                k = int(m_ko.group(1))
                return (k, v, v)

        return None
    
    def _check_answer(
        self,
        expected: Any,
        predicted: Optional[Tuple[int, int, int]]
    ) -> Tuple[bool, float]:
        """
        답변 확인
        
        Returns:
            (is_correct, partial_score) 튜플
        """
        if predicted is None:
            return False, 0.0
        
        # expected가 문자열일 수 있으므로 파싱
        if isinstance(expected, str):
            expected = self._parse_answer(expected, {})
        
        correct = predicted == expected
        return correct, 1.0 if correct else 0.0
