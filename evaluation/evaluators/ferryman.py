import re
from typing import Dict, Any, Tuple, Optional

from ..core.base import BaseEvaluator


class FerrymanEvaluator(BaseEvaluator):
    """
    Ferryman puzzle evaluator (EN / KO / JA / ZH).

    Answer formats: ``X hours Y minutes``, ``X시간 Y분``, ``X時間Y分``,
    ``X小时Y分钟`` (spaces optional where noted in regexes).
    """
    
    SYSTEM_PROMPT = """### Instructions
You are an expert at solving boat navigation problems.

### Rules
1. Analyze all given navigation regulations step by step.
2. Apply all speed limits, mandatory rest stops, and cargo regulations in your calculations.
3. Explain your reasoning clearly, then present your final conclusion in the format below.

### Output format
Your final line must be:
Answer: X hours Y minutes
"""

    KOREAN_SYSTEM_PROMPT = """### 지시사항
당신은 뱃사공 운항 문제를 정확히 해결하는 전문가입니다.

### 규칙
1. 주어진 운항 규정을 모두 고려하여 단계별로 분석하세요.
2. 속도 제한, 의무 휴식, 화물 규정을 모두 적용하여 계산하세요.
3. 풀이 과정을 명확히 서술한 뒤, 최종 결론을 아래 형식으로 제시하세요.

### 출력 형식
마지막 줄은 반드시 아래 형식으로 작성하세요:
Answer: N시간 M분
"""

    JAPANESE_SYSTEM_PROMPT = """### 指示
あなたは船舶航行・休憩・貨物規則を正確に適用できる専門家です。

### ルール
1. 与えられた航行規則をすべて段階的に分析してください。
2. 速度制限、強制休憩、貨物規定をすべて計算に反映してください。
3. 推理を明確に述べたうえで、最終結論を次の形式で書いてください。

### 出力形式
最後の行は必ず次の形式にしてください:
Answer: N時間M分
"""

    CHINESE_SYSTEM_PROMPT = """### 说明
你是能严格应用船舶航行、休息与货物规则的解题专家。

### 规则
1. 逐步分析题中全部航行规定。
2. 在计算中落实所有限速、强制休息与货物条款。
3. 写清推理过程后，用下列格式给出最终结论。

### 输出格式
最后一行必须写成:
Answer: N小时M分钟
"""

    @staticmethod
    def _strip_latex(text: str) -> str:
        """Remove common LaTeX noise (e.g. \\text{}, math spacing) from answer text."""
        text = re.sub(r'\\text\s*\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\[,;!]', ' ', text)
        text = text.replace('$', '')
        text = re.sub(r'[{}]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def _get_system_prompt(self, puzzle: Dict) -> str:
        task = getattr(self, "_task_name", "") or ""
        if re.search(r"_ko(?:_|$)", task) and self.KOREAN_SYSTEM_PROMPT:
            return self.KOREAN_SYSTEM_PROMPT
        if re.search(r"_ja(?:_|$)", task) and self.JAPANESE_SYSTEM_PROMPT:
            return self.JAPANESE_SYSTEM_PROMPT
        if re.search(r"_zh(?:_|$)", task) and self.CHINESE_SYSTEM_PROMPT:
            return self.CHINESE_SYSTEM_PROMPT
        return self.SYSTEM_PROMPT

    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[str]:
        """
        Extract time answer from the Answer: line and normalize to canonical form.
        Routes by task_name suffix (_ko / _ja / _zh / default English).
        """
        answer_text = self._extract_final_answer_text(response) or response
        task = getattr(self, "_task_name", "") or ""
        clean = self._strip_latex(answer_text)
        if re.search(r"_ko(?:_|$)", task):
            return self._extract_korean(answer_text)
        if re.search(r"_ja(?:_|$)", task):
            j = self._normalize_japanese(clean)
            if j:
                return j
            return self._normalize_english(clean)
        if re.search(r"_zh(?:_|$)", task):
            z = self._normalize_chinese(clean)
            if z:
                return z
            return self._normalize_english(clean)
        return self._extract_english(answer_text)

    def _extract_korean(self, response: str) -> Optional[str]:
        clean = self._strip_latex(response)
        k = self._normalize_korean(clean)
        if k:
            return k
        return self._normalize_english(clean)

    def _extract_english(self, response: str) -> Optional[str]:
        clean = self._strip_latex(response)
        return self._normalize_english(clean)

    @staticmethod
    def _normalize_korean(text: str) -> Optional[str]:
        m = re.search(r'(\d+)\s*시간\s*(\d+)\s*분', text)
        if m:
            return f"{int(m.group(1))}시간 {int(m.group(2))}분"
        return None

    @staticmethod
    def _normalize_english(text: str) -> Optional[str]:
        m = re.search(r'(\d+)\s*hours?\s+(\d+)\s*minutes?', text, re.IGNORECASE)
        if m:
            return f"{int(m.group(1))} hours {int(m.group(2))} minutes"
        m = re.search(r'(\d+)\s*hr?s?\s+(\d+)\s*min', text, re.IGNORECASE)
        if m:
            return f"{int(m.group(1))} hours {int(m.group(2))} minutes"
        return None

    @staticmethod
    def _normalize_japanese(text: str) -> Optional[str]:
        m = re.search(r'(\d+)\s*時間\s*(\d+)\s*分', text)
        if m:
            return f"{int(m.group(1))}時間{int(m.group(2))}分"
        return None

    @staticmethod
    def _normalize_chinese(text: str) -> Optional[str]:
        m = re.search(r'(\d+)\s*小时\s*(\d+)\s*分钟', text)
        if m:
            return f"{int(m.group(1))}小时{int(m.group(2))}分钟"
        return None

    def _parse_time_to_minutes(self, time_str: str) -> Optional[int]:
        match = re.search(r'(\d+)\s*小时\s*(\d+)\s*分钟', time_str)
        if match:
            return int(match.group(1)) * 60 + int(match.group(2))
        match = re.search(r'(\d+)\s*時間\s*(\d+)\s*分', time_str)
        if match:
            return int(match.group(1)) * 60 + int(match.group(2))
        match = re.search(r'(\d+)\s*시간\s*(\d+)\s*분', time_str)
        if match:
            return int(match.group(1)) * 60 + int(match.group(2))
        match = re.search(
            r'(\d+)\s*hours?\s+(\d+)\s*minutes?', time_str, re.IGNORECASE)
        if match:
            return int(match.group(1)) * 60 + int(match.group(2))
        return None

    def _check_answer(
        self,
        expected: str,
        predicted: Optional[str]
    ) -> Tuple[bool, float]:
        if predicted is None:
            return False, 0.0

        expected_minutes = self._parse_time_to_minutes(expected)
        predicted_minutes = self._parse_time_to_minutes(predicted)

        if expected_minutes is None or predicted_minutes is None:
            return False, 0.0

        correct = expected_minutes == predicted_minutes
        return correct, 1.0 if correct else 0.0

