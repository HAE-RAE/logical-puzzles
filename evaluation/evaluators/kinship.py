import base64
import logging
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING

from ..core.base import BaseEvaluator, EvaluationResult

if TYPE_CHECKING:
    from ..model.base import BaseLLMClient

logger = logging.getLogger(__name__)


class KinshipEvaluator(BaseEvaluator):
    """
    Kinship 퍼즐 평가자
    
    객관식 문제 (선택지 수 가변)
    kinship_vision의 경우 이미지도 함께 전송
    """
    
    SYSTEM_PROMPT = """### Instructions
You are an expert at Korean family kinship (honorific) puzzles.

### Rules
1. Analyze the given relationship chain step by step to find the correct title for the target person.
2. Map your conclusion to exactly one option letter (A–Z) from the multiple-choice list in the user message.
3. Explain your reasoning clearly, then present your final conclusion in the format below.

### Output format
Your final line must be:
Answer: A
(where A is a single letter A–Z matching a provided choice.)
"""

    KOREAN_SYSTEM_PROMPT = """### 지시사항
당신은 한국어 가족 관계 호칭 문제를 정확히 푸는 전문가입니다.

### 규칙
1. 주어진 관계를 단계별로 분석하여 대상에 대한 올바른 호칭을 찾으세요.
2. 사용자 메시지의 보기 중 정확히 하나의 선택지 문자(A~Z)만 고르세요.
3. 풀이 과정을 명확히 서술한 뒤, 최종 결론을 아래 형식으로 제시하세요.

### 출력 형식
마지막 줄은 반드시 아래 형식으로 작성하세요:
Answer: A (A~Z 중 보기에 해당하는 한 글자)
"""

    VISION_SYSTEM_PROMPT = """### Instructions
You are an expert at Korean family kinship puzzles that include a reference photo.

### Rules
1. Use the provided family photo together with the dialogue to identify who is being described.
2. Map your conclusion to exactly one option letter (A–Z) from the multiple-choice list in the user message.
3. Explain your reasoning clearly, then present your final conclusion in the format below.

### Output format
Your final line must be:
Answer: A
(where A is a single letter A–Z matching a provided choice.)
"""

    KOREAN_VISION_SYSTEM_PROMPT = """### 지시사항
당신은 가족 사진이 함께 주어지는 한국어 가족 호칭 문제를 정확히 푸는 전문가입니다.

### 규칙
1. 제공된 가족 사진과 대화를 함께 사용하여 묻는 인물을 특정하세요.
2. 사용자 메시지의 보기 중 정확히 하나의 선택지 문자(A~Z)만 고르세요.
3. 풀이 과정을 명확히 서술한 뒤, 최종 결론을 아래 형식으로 제시하세요.

### 출력 형식
마지막 줄은 반드시 아래 형식으로 작성하세요:
Answer: A (A~Z 중 보기에 해당하는 한 글자)
"""
    
    def __init__(self):
        super().__init__()
        # 이미지 경로 (evaluation 디렉토리 기준)
        script_dir = Path(__file__).parent.parent
        self.image_path = script_dir / "eval_data" / "kinship_vision" / "kinship.jpg"
        self._image_base64 = None
    
    def _get_image_base64(self) -> Optional[str]:
        """이미지를 base64로 인코딩 (캐싱)"""
        if self._image_base64 is not None:
            return self._image_base64
        
        if not self.image_path.exists():
            logger.warning(f"Image not found: {self.image_path}")
            return None
        
        try:
            with open(self.image_path, 'rb') as image_file:
                self._image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
            return self._image_base64
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            return None
    
    def _prepare_messages(self, puzzle: Dict[str, Any], task_name: Optional[str] = None) -> List[Dict]:
        """
        메시지 준비 (이미지 포함 여부 결정)
        
        Args:
            puzzle: 퍼즐 데이터
            task_name: task 이름 (kinship_vision인 경우 이미지 포함)
        """
        # kinship_vision인 경우 이미지 포함
        if task_name == "kinship_vision":
            system_prompt = self.KOREAN_VISION_SYSTEM_PROMPT
            image_base64 = self._get_image_base64()
            
            if image_base64:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                            },
                            {
                                "type": "text",
                                "text": puzzle["question"]
                            }
                        ]
                    }
                ]
            else:
                # 이미지 로드 실패 시 텍스트만
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": puzzle["question"]}
                ]
        else:
            # 일반 kinship (텍스트만)
            messages = [
                {"role": "system", "content": self.KOREAN_SYSTEM_PROMPT},
                {"role": "user", "content": puzzle["question"]}
            ]
        
        return messages
    
    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[str]:
        """
        Extract choice letter from LLM response (Answer: line and heuristics).
        """
        valid_choices = set(puzzle.get("choices", {}).keys()) if isinstance(puzzle.get("choices"), dict) else None
        if valid_choices:
            max_letter = max(valid_choices)
        else:
            max_letter = "Z"

        def _valid(letter: str) -> bool:
            return valid_choices is None or letter in valid_choices

        answer_text = self._extract_final_answer_text(response) or response

        answer_match = re.search(r'(?i)\banswer\s*[:：]\s*([A-Z])\b', answer_text)
        if answer_match:
            letter = answer_match.group(1).upper()
            if len(letter) == 1 and letter.isalpha() and letter <= max_letter and _valid(letter):
                return letter

        upper = answer_text.upper().strip()

        match = re.search(r'(?:^|[^A-Z])([A-' + max_letter + r'])(?:[^A-Z]|$)', upper)
        if match and _valid(match.group(1)):
            return match.group(1)

        match = re.search(r'^([A-' + max_letter + r'])', upper)
        if match and _valid(match.group(1)):
            return match.group(1)

        match = re.search(r'([A-' + max_letter + r'])(?:[^A-Z]|$)', upper)
        if match and _valid(match.group(1)):
            return match.group(1)

        match = re.search(r'[답정][변답]?\s*[:：]?\s*([A-' + max_letter + r'])', upper)
        if match and _valid(match.group(1)):
            return match.group(1)

        match = re.search(r'([A-' + max_letter + r'])', upper)
        if match and _valid(match.group(1)):
            return match.group(1)

        return None
    
    def _check_answer(
        self,
        expected: str,
        predicted: Optional[str]
    ) -> Tuple[bool, float]:
        """
        답변 확인
        
        Returns:
            (is_correct, partial_score) 튜플
        """
        if predicted is None:
            return False, 0.0
        
        correct = predicted == expected
        return correct, 1.0 if correct else 0.0
    
    def _evaluate_single(
        self,
        puzzle: Dict[str, Any],
        llm_client: "BaseLLMClient"
    ) -> EvaluationResult:
        """
        단일 퍼즐 평가 (이미지 포함 가능)
        """
        messages = self._prepare_messages(puzzle, getattr(self, '_task_name', None))
        
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
        max_concurrent: int = 10
    ) -> List[EvaluationResult]:
        """
        비동기 평가 실행 (이미지 포함 가능)
        """
        # 모든 메시지 준비
        messages_list = []
        task_name = getattr(self, '_task_name', None)
        for puzzle in puzzles:
            messages = self._prepare_messages(puzzle, task_name)
            messages_list.append(messages)
        
        total_puzzles = len(puzzles)
        task_name = getattr(self, '_task_name', None)
        task_prefix = f"[{task_name}] " if task_name else ""
        
        if verbose:
            logger.info(f"{task_prefix}Starting async evaluation: {total_puzzles} puzzles, max_concurrent={max_concurrent}")
        
        # 비동기 배치 생성
        start_time = time.time()
        
        def progress_callback(completed, total):
            if verbose:
                percentage = (completed / total) * 100
                if completed % max(1, total // 10) == 0 or completed == total:
                    logger.info(f"{task_prefix}API calls progress: {completed}/{total} ({percentage:.0f}%)")
        
        responses = await llm_client.async_batch_generate(
            messages_list, 
            max_concurrent=max_concurrent,
            progress_callback=progress_callback if verbose else None
        )
        total_latency = (time.time() - start_time) * 1000
        
        if verbose:
            logger.info(f"{task_prefix}API calls completed: {total_puzzles}/{total_puzzles} in {total_latency:.0f}ms ({total_latency/total_puzzles:.0f}ms per puzzle)")
        
        # 결과 처리
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
            logger.info(f"Processing completed: {correct_count} correct, {incorrect_count} incorrect, {error_count} errors")
        
        return results