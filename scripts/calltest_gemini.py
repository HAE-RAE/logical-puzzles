import os
import time
from pathlib import Path

from dotenv import load_dotenv
from litellm import completion
import litellm

litellm.set_verbose = True

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
if not os.getenv("GEMINI_API_KEY"):
    raise SystemExit("GEMINI_API_KEY is not set. Please check the .env file.")


system_prompt = """당신은 뱃사공 운항 문제를 정확히 해결하는 전문가입니다.

### 규칙
1. 주어진 운항 규정을 모두 고려하여 단계별로 분석하세요.
2. 속도 제한, 의무 휴식, 화물 규정을 모두 적용하여 계산하세요.
3. 풀이 과정을 명확히 서술한 뒤, 최종 결론을 아래 형식으로 제시하세요.

### 최종 답 형식
마지막에 $\\boxed{N시간 M분}$ 형식으로 정답을 표시하세요.
"""


content = """
수로 안내원 오씨는 이 강에서만 9년을 일한 베테랑으로, 총 길이 206km의 상류 지역에 물품을 운송하는 임무를 맡았다. 그는 오전 8시에 12kg짜리 의료품 키트 10개와 15kg짜리 식수통 5개를 싣고 출발했다. 배는 정수(靜水)에서 시속 22km로 이동 가능하다.
이 강에는 시속 7km의 물살이 있는데, B구역에서는 순류(실효 속력 = 배 속력 + 유속), A구역에서는 역류(실효 속력 = 배 속력 - 유속)이다.
첫 23km는 A구역(제한속도 17km/h), 이후 B구역(제한속도 13km/h)이다. 제한 속도는 유속 반영 후 실효 속력에 적용된다.
안전 중량 기준(400kg) 초과 시, A구역 제한속도는 21% 감소하고 B구역 제한속도는 33% 감소한다.
오전 9시부터 오후 2시까지는 혼잡시간대로, 모든 구역의 제한속도가 추가로 41% 감소한다. 이 감속은 화물 규정 적용 후 제한속도에 추가 적용된다.
60km 지점의 중간 기착지에서 식수통 3개를 하역한다. 하역 후 잔여 화물 무게에 따라 화물 규정이 재적용된다.
연속 101분 이상 운항할 수 없으며, 휴게 지점(매 12km)에서만 쉴 수 있다. 기본 휴식 시간은 44분이다. 단, 누적 피로 규정에 따라 매 휴식마다 20분씩 추가된다(첫째 44분, 둘째 64분, 셋째 84분, …).
이 모든 조건을 준수하여 최종 목적지까지 도착했을 때, 의무 휴식을 포함한 총 소요 시간은 몇 시간 몇 분입니까? (분은 소숫점 첫째 자리에서 반올림)\
"""

# hard 68시간 24분

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": content}
]

print(f"[Messages] {len(messages)} messages loaded.")
print("-" * 50)

start = time.time()
try:
    response = completion(
        model="gemini/gemini-3-flash-preview",
        messages=messages,
        temperature=1.0,
        max_tokens=65539,
        top_p=0.95,
        top_k=64,
        reasoning_effort="high"
    )
    latency = time.time() - start
    print(f"[Response Time] {latency:.2f}s")

    message = response.choices[0].message
    content = message.content or ""
    
    reasoning = getattr(message, "reasoning_content", None)

    print(f"\n[Reasoning]\n{reasoning if reasoning else '(None - may be included in the content or may not be separated depending on the LiteLLM version)'}")
    print(f"\n[Answer]\n{content}")

    if hasattr(response, 'usage'):
        print(f"\n[Usage] {response.usage}")

except Exception as e:
    print(f"[Error] {e}")