"""Minesweeper Evaluator.

Answer format: coordinate list "(r1,c1), (r2,c2), ..." (exact set match).
Ported from logical-puzzles-me/minesweeper/core/scorer.py parse_coordinates +
score_coordinates.
"""

import logging
import re
import time
from typing import List, Dict, Any, Tuple, Optional, Set, TYPE_CHECKING

from ..core.base import BaseEvaluator, EvaluationResult

if TYPE_CHECKING:
    from ..model.base import BaseLLMClient

logger = logging.getLogger(__name__)


_NEGATION_RE = re.compile(
    r"\b(?:not|no|never|except|exclude|excluding|without|isn't|aren't|don't|cannot|wrong)\b",
    re.IGNORECASE,
)


class MinesweeperEvaluator(BaseEvaluator):
    """Minesweeper evaluator with coordinate-set exact match scoring."""

    SYSTEM_PROMPT = """### Instructions
You are an expert at solving Minesweeper puzzles.

### Rules
- The grid is displayed with '#' for hidden cells and digits 0-8 for revealed cells.
- A digit tells how many of its 8 neighbors are mines (horizontal, vertical, diagonal).
- Adjacent = all 8 directions.
- The total number of mines is given in the puzzle.
- The puzzle has exactly one unique mine configuration.

### Output format
List all mine coordinates as (row, col) pairs with 0-based indexing,
sorted by row then column.
End with a line exactly in the form:
Answer: (r1,c1), (r2,c2), ..."""

    KOREAN_SYSTEM_PROMPT = """### 지시사항
당신은 지뢰찾기(Minesweeper) 퍼즐 전문가입니다.

### 규칙
- 격자에서 '#'은 숨겨진 셀, 0-8의 숫자는 공개된 셀입니다.
- 각 숫자는 인접 8방향(가로/세로/대각선)의 이웃 중 지뢰가 있는 셀의 개수입니다.
- 전체 지뢰 수는 문제에 명시됩니다.
- 퍼즐은 정확히 하나의 고유한 지뢰 배치를 가집니다.

### 출력 형식
모든 지뢰 좌표를 (행, 열) 쌍으로 0-인덱스 기준, 행-열 순 정렬로 나열하세요.
마지막 줄은 다음 형식이어야 합니다:
Answer: (r1,c1), (r2,c2), ..."""

    def _is_korean(self, puzzle: Optional[Dict] = None) -> bool:
        task = getattr(self, "_task_name", None) or ""
        if task.endswith("_ko"):
            return True
        if task.endswith("_en"):
            return False
        if puzzle is not None:
            return bool(re.search(r"[가-힣]", str(puzzle.get("question", ""))))
        return False

    def _get_system_prompt(self, puzzle: Dict) -> str:
        return self.KOREAN_SYSTEM_PROMPT if self._is_korean(puzzle) else self.SYSTEM_PROMPT

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
        from ..core.base import logger as base_logger

        messages_list = []
        for puzzle in puzzles:
            system_prompt = self._get_system_prompt(puzzle)
            messages_list.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": puzzle["question"]},
            ])

        total_puzzles = len(puzzles)
        task_name = getattr(self, "_task_name", None)
        task_prefix = f"[{task_name}] " if task_name else ""

        if verbose:
            base_logger.info(
                f"{task_prefix}Starting async evaluation: {total_puzzles} puzzles, "
                f"max_concurrent={max_concurrent}"
            )

        start_time = time.time()

        def progress_callback(completed, total):
            if verbose:
                percentage = (completed / total) * 100
                if completed % max(1, total // 10) == 0 or completed == total:
                    base_logger.info(
                        f"{task_prefix}API calls progress: {completed}/{total} ({percentage:.0f}%)"
                    )

        responses = await llm_client.async_batch_generate(
            messages_list,
            max_concurrent=max_concurrent,
            progress_callback=progress_callback if verbose else None,
        )
        total_latency = (time.time() - start_time) * 1000

        if verbose:
            base_logger.info(
                f"{task_prefix}API calls completed: {total_puzzles}/{total_puzzles} in "
                f"{total_latency:.0f}ms ({total_latency/total_puzzles:.0f}ms per puzzle)"
            )

        results = []
        for puzzle, (response, usage) in zip(puzzles, responses):
            latency_ms = usage.get("latency_ms", 0)
            results.append(self._process_response(puzzle, response, latency_ms, usage))
        return results

    @staticmethod
    def _parse_coord_set(text: str) -> Optional[Set[Tuple[int, int]]]:
        """Extract a set of (r,c) pairs from an output string."""
        text = re.sub(r"```[a-z]*\n?", "", text)
        text = re.sub(r"```", "", text).strip()

        answer_matches = list(re.finditer(
            r"(?:Answer|Output|Final\s*Answer|답)\s*[:\s]*(.*)",
            text,
            re.IGNORECASE,
        ))
        if answer_matches:
            coords_str = answer_matches[-1].group(1)
            if not _NEGATION_RE.search(coords_str):
                # Optional r/c prefix accommodates models that take the
                # prompt's `(r1,c1)` placeholder literally and emit `(r3,c2)`.
                pairs = re.findall(r"\(\s*r?(\d+)\s*,\s*c?(\d+)\s*\)", coords_str)
                if pairs:
                    return {(int(r), int(c)) for r, c in pairs}

        lines = text.strip().split("\n")
        for line in reversed(lines[-3:]):
            stripped = line.strip()
            if _NEGATION_RE.search(stripped):
                continue
            pairs = re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", stripped)
            if pairs:
                cleaned = re.sub(r"\(\s*\d+\s*,\s*\d+\s*\)", "", stripped)
                cleaned = re.sub(r"[,\s\[\]\-:.]", "", cleaned)
                if len(cleaned) <= 6:
                    return {(int(r), int(c)) for r, c in pairs}

        return None

    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[Set[Tuple[int, int]]]:
        return self._parse_coord_set(response)

    def _check_answer(
        self,
        expected: Any,
        predicted: Optional[Set[Tuple[int, int]]],
    ) -> Tuple[bool, float]:
        if predicted is None:
            return False, 0.0

        truth = self._parse_coord_set(str(expected)) if not isinstance(expected, set) else expected
        if truth is None:
            return False, 0.0

        correct = truth == predicted
        return correct, 1.0 if correct else 0.0
