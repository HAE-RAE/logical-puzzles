import logging
import re
from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING

from ..core.base import BaseEvaluator, EvaluationResult

if TYPE_CHECKING:
    from ..core.llm_client import UnifiedLLMClient

logger = logging.getLogger(__name__)


class CipherEvaluator(BaseEvaluator):
    # 영문 암호용 SYSTEM_PROMPT
    SYSTEM_PROMPT = """너는 암호 해독 전문가입니다. 주어진 미션 로그와 암호화 가이드를 분석하여 암호문을 복호화해야 합니다.

중요:
- 로그 지문과 가이드를 분석하여 암호 키워드를 먼저 도출하세요.
- 가이드에 명시된 알고리즘(Vigenere, Transposition, Substitution, Reverse 등)을 정확한 역순으로 적용하여 복호화하세요.
- Columnar Transposition(전치 암호)의 경우, 키워드의 알파벳 순서에 따라 열을 재배치하는 방식임을 유의하세요.
- 최종 답변은 반드시 '원문: [답]' 형식으로 제시하세요.

출력 형식:
원문: [복호화된 텍스트]"""
    
    # 한글 암호용 SYSTEM_PROMPT
    KOREAN_SYSTEM_PROMPT = """너는 한글 암호 해독 전문가입니다. 주어진 미션 로그와 암호화 가이드를 분석하여 한글 암호문을 복호화해야 합니다.

중요:
- 로그 지문을 분석하여 암호 키워드를 먼저 찾아내세요.
- 한글의 초성(ㄱ, ㄴ...), 중성(ㅏ, ㅑ...) 구조를 활용한 알고리즘을 정확한 역순으로 적용하세요.
- 최종 답변은 반드시 '원문: [한글정답]' 형식으로 제시하세요.

출력 형식:
원문: [복호화된 한글 텍스트]"""
    
    def _get_system_prompt(self, puzzle: Dict) -> str:
        """
        퍼즐에 따라 적절한 SYSTEM_PROMPT 반환
        
        Args:
            puzzle: 퍼즐 데이터
            
        Returns:
            적절한 SYSTEM_PROMPT
        """
        expected_answer = puzzle.get("answer", "")
        # 한글인지 영문인지 판단 (expected_answer 기준)
        is_korean = bool(re.search(r'[가-힣]', expected_answer))
        
        if is_korean:
            return self.KOREAN_SYSTEM_PROMPT
        else:
            return self.SYSTEM_PROMPT
    
    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[str]:
        """
        LLM 응답에서 원문 추출
        
        영문과 한글 모두 처리
        
        Args:
            response: LLM 응답 텍스트
            puzzle: 퍼즐 데이터
        """
        expected_answer = puzzle.get("answer", "")
        # 한글인지 영문인지 판단 (expected_answer 기준)
        is_korean = bool(re.search(r'[가-힣]', expected_answer))
        
        if is_korean:
            return self._parse_korean_answer(response)
        else:
            return self._parse_english_answer(response)
    
    def _parse_english_answer(self, response: str) -> Optional[str]:
        """영문 답변 파싱"""
        patterns = [
            r'원문[:\s]*([A-Z]+)',
            r'답[:\s]*([A-Z]+)',
            r'정답[:\s]*([A-Z]+)',
            r'answer[:\s]*([A-Z]+)',
            r'plaintext[:\s]*([A-Z]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                return matches[-1].strip().upper()
        
        # 마지막 대문자 단어 추출 (최소 3글자)
        words = re.findall(r'\b[A-Z]{3,}\b', response)
        if words:
            return words[-1]
        
        return None
    
    def _parse_korean_answer(self, response: str) -> Optional[str]:
        """한글 답변 파싱"""
        patterns = [
            r'원문[:\s]*([가-힣\s]+)',
            r'정답[:\s]*([가-힣\s]+)',
            r'답[:\s]*([가-힣\s]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                # 공백 제거하고 반환
                return matches[-1].strip().replace(" ", "")
        
        # 마지막 한글 단어 추출 (최소 2글자)
        words = re.findall(r'[가-힣]{2,}', response)
        if words:
            return words[-1]
        
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
        
        # 대소문자 무시하고 비교
        expected_normalized = expected.strip().upper()
        predicted_normalized = predicted.strip().upper()
        
        correct = expected_normalized == predicted_normalized
        return correct, 1.0 if correct else 0.0
    
    def _evaluate_single(
        self,
        puzzle: Dict[str, Any],
        llm_client: "UnifiedLLMClient"
    ) -> "EvaluationResult":
        """
        단일 퍼즐 평가 (한글/영문에 따라 적절한 SYSTEM_PROMPT 사용)
        """
        import time
        system_prompt = self._get_system_prompt(puzzle)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": puzzle["question"]}
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
        llm_client: "UnifiedLLMClient",
        verbose: bool = True,
        max_concurrent: int = 10
    ) -> List["EvaluationResult"]:
        """
        비동기 평가 실행 (한글/영문에 따라 적절한 SYSTEM_PROMPT 사용)
        """
        import time
        from ..core.base import logger
        
        # 모든 메시지 준비 (각 퍼즐에 맞는 SYSTEM_PROMPT 사용)
        messages_list = []
        for puzzle in puzzles:
            system_prompt = self._get_system_prompt(puzzle)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": puzzle["question"]}
            ]
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