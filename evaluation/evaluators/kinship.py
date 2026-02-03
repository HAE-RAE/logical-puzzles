import logging
import re
import base64
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING

from ..core.base import BaseEvaluator, EvaluationResult

if TYPE_CHECKING:
    from ..core.llm_client import UnifiedLLMClient

logger = logging.getLogger(__name__)


class KinshipEvaluator(BaseEvaluator):
    """
    Kinship 퍼즐 평가자
    
    객관식 문제 (A-E 중 선택)
    kinship_vision의 경우 이미지도 함께 전송
    """
    
    SYSTEM_PROMPT = """당신은 한국어 가족 관계 호칭 문제를 푸는 전문가입니다. 

### 규칙
1. 주어진 가족 관계를 단계별로 분석하여 올바른 호칭을 찾으세요.
2. 문제에 제시된 선택지 중 정답에 해당하는 알파벳(A, B, C, D, E)만 답하세요.
3. 추가 설명 없이 알파벳 하나만 출력하세요.

### 출력 형식
정답 알파벳만 출력하세요. 예: A
"""
    
    VISION_SYSTEM_PROMPT = """당신은 한국어 가족 관계 호칭 문제를 푸는 전문가입니다. 

### 규칙
1. 제공된 가족 사진을 참고하여, 주어진 대화에서 설명하는 인물을 찾으세요.
2. 대화를 단계별로 분석하여 각 인물 간의 관계를 추론하세요.
3. 문제에 제시된 선택지 중 정답에 해당하는 알파벳(A, B, C, D, E)만 답하세요.
4. 추가 설명 없이 알파벳 하나만 출력하세요.

### 출력 형식
정답 알파벳만 출력하세요. 예: A
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
            system_prompt = self.VISION_SYSTEM_PROMPT
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
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": puzzle["question"]}
            ]
        
        return messages
    
    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[str]:
        """
        LLM 응답에서 알파벳 A-E 추출
        
        여러 패턴을 시도하여 가장 적절한 알파벳을 찾습니다.
        
        Args:
            response: LLM 응답 텍스트
            puzzle: 퍼즐 데이터 (사용하지 않음)
        """
        response = response.upper().strip()
        
        # 패턴 1: 단독으로 나타나는 A-E
        match = re.search(r'(?:^|[^A-Z])([A-E])(?:[^A-Z]|$)', response)
        if match:
            return match.group(1)
        
        # 패턴 2: 시작 부분의 A-E
        match = re.search(r'^([A-E])', response)
        if match:
            return match.group(1)
        
        # 패턴 3: 끝 부분의 A-E
        match = re.search(r'([A-E])(?:[^A-Z]|$)', response)
        if match:
            return match.group(1)
        
        # 패턴 4: 답변/정답 뒤의 A-E
        match = re.search(r'[답정][변답]?\s*[:：]?\s*([A-E])', response)
        if match:
            return match.group(1)
        
        # 패턴 5: 아무 A-E (마지막 수단)
        match = re.search(r'([A-E])', response)
        if match:
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
    
    def evaluate(
        self,
        puzzles: List[Dict[str, Any]],
        llm_client: "UnifiedLLMClient",
        verbose: bool = True,
        use_async: bool = False,
        max_concurrent: int = 10,
        task_name: Optional[str] = None
    ) -> List[EvaluationResult]:
        """
        평가 실행 (task_name을 저장하여 이미지 처리 여부 결정)
        """
        self._task_name = task_name
        return super().evaluate(puzzles, llm_client, verbose, use_async, max_concurrent)
    
    def _evaluate_single(
        self,
        puzzle: Dict[str, Any],
        llm_client: "UnifiedLLMClient"
    ) -> EvaluationResult:
        """
        단일 퍼즐 평가 (이미지 포함 가능)
        """
        import time
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
        llm_client: "UnifiedLLMClient",
        verbose: bool = True,
        max_concurrent: int = 10
    ) -> List[EvaluationResult]:
        """
        비동기 평가 실행 (이미지 포함 가능)
        """
        import time
        from ..core.base import logger
        
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