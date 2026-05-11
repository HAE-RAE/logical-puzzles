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
    
    SYSTEM_PROMPT = """You are an expert puzzle solver specializing in the Tower of Hanoi.

    [PREREQUISITES & RULES]

    Standard rules apply: only one disk can be moved at a time, and a larger disk cannot be placed on top of a smaller disk. Disks are numbered from 1 (smallest) to n (largest).
    The optimal solution for 'n' disks requires exactly 2^n - 1 moves.
    In an optimal sequence, a specific Disk 'd' moves exactly 2^(n-d) times.
    [ANSWER FORMAT INSTRUCTIONS]
    You must output your final answer formatted exactly as a tuple wrapped inside tags. Format rules based on the question asked:

    Minimum number of moves needed: (moves, moves, moves)
    How many times a specific disk moves: (disk_number, moves, moves)
    Which disk is moved on the k-th move / Describe a specific move: (disk_number, from_peg, to_peg)
    On which peg is a specific disk located: (disk_number, peg_number, peg_number)
    Which disks are on a specific peg: (*disks_in_ascending_order, peg_number, peg_number). If empty, use 'none' for the disk part.
    How many disks are in this Tower of Hanoi puzzle: (n, n, n)
    On which move number does Disk d first move, and on which move number does it last move: (first_move_number, last_move_number, disk_number)
    Provide ONLY the tag with the tuple inside. Do not provide any step-by-step reasoning, mathematical formulas, or additional text."""


    KOREAN_SYSTEM_PROMPT = """당신은 하노이 탑(Tower of Hanoi) 퍼즐을 전문으로 해결하는 전문가입니다.

    [사전 지식 및 규칙]
    표준 규칙이 적용됩니다: 한 번에 하나의 원판만 이동할 수 있으며, 큰 원판을 작은 원판 위에 놓을 수 없습니다. 원판은 1(가장 작음)부터 n(가장 큼)까지 번호가 매겨져 있습니다.
    'n'개의 원판을 옮기는 최적의 해법은 정확히 2^n - 1번의 이동이 필요합니다.
    최적의 이동 순서에서, 특정 원판 'd'는 정확히 2^(n-d)번 이동합니다.

    [답변 형식 지침]
    최종 답변은 반드시 태그 안에 튜플 형태로 작성해야 합니다. 질문의 종류에 따른 형식 규칙은 다음과 같습니다:

    - 필요한 최소 이동 횟수: (이동횟수, 이동횟수, 이동횟수)
    - 특정 원판의 이동 횟수: (원판번호, 이동횟수, 이동횟수)
    - k번째 이동에 움직이는 원판 / 특정 이동 설명: (원판번호, 출발기둥, 도착기둥)
    - 특정 원판이 위치한 기둥: (원판번호, 기둥번호, 기둥번호)
    - 특정 기둥에 있는 원판들: (*오름차순_원판들, 기둥번호, 기둥번호). 만약 비어있다면 원판 부분에 'none'을 사용.
    - 이 하노이 탑 퍼즐에 있는 원판의 총 개수: (n, n, n)
    - 원판 d가 처음 이동하는 횟수와 마지막으로 이동하는 횟수: (첫이동번째, 마지막이동번째, 원판번호)

    튜플이 포함된 태그만 제공하십시오. 단계별 추론 과정, 수학 공식 또는 추가 텍스트를 제공하지 마십시오."""


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
    
    def _is_easy(self, puzzle: Optional[Dict] = None) -> bool:
       
        if puzzle is not None:
            expected = puzzle.get("difficulty", "")
            if expected.lower()=="easy":
               return True 
            else:
               return False

    def _get_system_prompt(self, puzzle: Dict) -> str:
        
        if self._is_korean(puzzle):
            return self.KOREAN_SYSTEM_PROMPT
        else:
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
        LLM 응답에서 (disk, from, to) 튜플 추출
        
        숫자 3개를 찾아서 튜플로 반환
        
        Args:
            response: LLM 응답 텍스트
            puzzle: 퍼즐 데이터 (사용하지 않음)
        """
        answer_text = self._extract_final_answer_text(response) or response

        # 패턴 1: (숫자, 숫자, 숫자) 형식
        match = re.search(r'\((\d+),\s*(\d+),\s*(\d+)\)', answer_text)
        if match:
            return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        
        # 패턴 2: 숫자 3개 추출
        nums = re.findall(r'\d+', answer_text)
        if len(nums) >= 3:
            return (int(nums[0]), int(nums[1]), int(nums[2]))

        # 단일 숫자 답변 보정: 역문제/총 이동 횟수 문제
        question_raw = str(puzzle.get("question", ""))
        question = question_raw.lower()
        if len(nums) == 1:
            v = int(nums[0])

            # How many disks total? (inverse_find_n) -> (n, n, n)
            # Exclude "top disk per peg" questions which also ask about disks per peg
            is_inverse_en = (
                "how many disks" in question
                and "top" not in question
                and "top_of_peg" not in question
            )
            is_inverse_ko = (
                "원판이 몇 개" in question_raw
                and "최상단" not in question_raw
            )
            if is_inverse_en or is_inverse_ko:
                return (v, v, v)

            # How many times does Disk k move? -> (k, t, t)
            m = re.search(r"disk\s*(\d+)", question, re.IGNORECASE)
            if "how many times" in question and m:
                k = int(m.group(1))
                return (k, v, v)

            # Korean variant: "원반 k" + "몇 번"
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
