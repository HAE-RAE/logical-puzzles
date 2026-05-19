import logging
import re
from typing import Dict, Any, Tuple, Optional

from ..core.base import BaseEvaluator

logger = logging.getLogger(__name__)


class CipherEvaluator(BaseEvaluator):
    SYSTEM_PROMPT = """### Instructions
You are an expert at pencil-and-paper ciphers and decoding puzzles.

### Rules
1. Follow the ciphertext, mission log (if any), and encryption guide; derive keywords from the log when the puzzle requires it.
2. Decrypt by reversing each named step in the correct order (e.g. Vigenère, columnar transposition with keyword-sorted columns, substitution, reverse).
3. Explain your reasoning clearly, then present your final conclusion in the format below.

### Output format
Your final line must be:
Answer: WORD
(WORD: uppercase A–Z, no spaces. The evaluator uses only the last Answer: line for scoring.)
"""

    KOREAN_SYSTEM_PROMPT = """### 지시사항
당신은 암호·복호 퍼즐을 정확히 푸는 전문가입니다.

### 규칙
1. 암호문, 미션 로그(있을 경우), 암호화 가이드를 따르고 필요 시 로그에서 키워드를 찾으세요.
2. 안내된 단계를 역순으로 적용해 복호화하세요(비즈네르, 키워드 열 순서의 전치, 치환, 역순 등).
3. 풀이 과정을 명확히 서술한 뒤, 최종 결론을 아래 형식으로 제시하세요.

### 출력 형식
마지막 줄은 반드시 아래 형식으로 작성하세요:
Answer: 복호결과
(공백 없는 한글 등 지문이 요구하는 형태. 평가기는 가장 마지막 Answer: 줄만 채점에 사용합니다.)
"""

    @staticmethod
    def trim_to_last_answer_line(raw: str) -> str:
        """Keep from the last ``Answer:`` / ``Answer：`` onward (canonical).

        If there is no ``Answer:`` line, fall back to the last ``원문:`` slice
        for older puzzle prompts that still ask for ``원문:`` in the user text.
        """
        if not raw:
            return raw
        answer_matches = list(re.finditer(r"answer\s*[:：]", raw, flags=re.IGNORECASE))
        if answer_matches:
            return raw[answer_matches[-1].start() :]
        won = list(re.finditer(r"원문\s*[:：]", raw))
        if won:
            return raw[won[-1].start() :]
        return raw

    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[str]:
        """
        LLM 응답에서 원문 추출
        
        영문과 한글 모두 처리
        
        Args:
            response: LLM 응답 텍스트
            puzzle: 퍼즐 데이터
        """
        trimmed = self.trim_to_last_answer_line(response or "")
        if self._is_korean(puzzle):
            return self._parse_korean_answer(trimmed)
        return self._parse_english_answer(trimmed)
    
    def _parse_english_answer(self, response: str) -> Optional[str]:
        """영문 답변 파싱 — ``Answer:`` 우선."""
        answer_text = self._extract_final_answer_text(response) or response
        patterns = [
            r"answer[:\s]*([A-Z]+)",
            r"plaintext[:\s]*([A-Z]+)",
            r"원문[:\s]*([A-Z]+)",
            r"답[:\s]*([A-Z]+)",
            r"정답[:\s]*([A-Z]+)",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, answer_text, re.IGNORECASE)
            if matches:
                return matches[-1].strip().upper()
        
        # 마지막 대문자 단어 추출 (최소 3글자)
        words = re.findall(r'\b[A-Z]{3,}\b', answer_text)
        if words:
            return words[-1]
        
        return None
    
    def _parse_korean_answer(self, response: str) -> Optional[str]:
        """한글 답변 파싱 — ``Answer:`` 우선."""
        answer_text = self._extract_final_answer_text(response) or response
        # Do not use \\s inside the capture group — it includes newlines and can
        # merge multiple label lines into one wrong span.
        patterns = [
            r"answer[:\s]*([가-힣]+)",
            r"원문[:\s]*([가-힣]+)",
            r"정답[:\s]*([가-힣]+)",
            r"답[:\s]*([가-힣]+)",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, answer_text, re.IGNORECASE)
            if matches:
                # 공백 제거하고 반환
                return matches[-1].strip().replace(" ", "")
        
        # 마지막 한글 단어 추출 (최소 2글자)
        words = re.findall(r'[가-힣]{2,}', answer_text)
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
    