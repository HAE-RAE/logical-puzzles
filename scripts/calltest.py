import json
import requests
import time

URL = "https://tremendously-bureaucratic-alda.ngrok-free.dev"

system_prompt = """당신은 뱃사공 운항 문제를 정확히 해결하는 전문가입니다.

### 규칙
1. 주어진 운항 규정을 모두 고려하여 단계별로 분석하세요.
2. 속도 제한, 의무 휴식, 화물 규정을 모두 적용하여 계산하세요.
3. 풀이 과정을 명확히 서술한 뒤, 최종 결론을 아래 형식으로 제시하세요.

### 최종 답 형식
마지막에 $\\boxed{N시간 M분}$ 형식으로 정답을 표시하세요.
"""

content = """
물품 운송원 박씨는 이 강에서만 8년을 일한 베테랑으로, 총 길이 111km의 상류 지역에 물품을 운송하는 임무를 맡았다. 그는 오전 7시에 16kg짜리 식수통 15개와 12kg짜리 의료품 키트 2개를 싣고 출발했다. 배는 정수(靜水)에서 시속 50km로 이동 가능하다.

이 강에는 시속 3km의 물살이 있는데, A구역에서는 순류(실효 속력 = 배 속력 + 유속), B구역에서는 역류(실효 속력 = 배 속력 - 유속)이다.

첫 29km는 A구역(제한속도 38km/h), 이후 B구역(제한속도 27km/h)이다. 제한 속도는 유속 반영 후 실효 속력에 적용된다.

안전 중량 기준(1600kg) 초과 시, 모든 구역의 제한속도가 16% 감소한다.

오전 11시부터 낮 12시까지는 혼잡시간대로, 모든 구역의 제한속도가 추가로 5% 감소한다. 이 감속은 화물 규정 적용 후 제한속도에 추가 적용된다.

연속 99분 이상 운항할 수 없으며, 휴게 지점(매 18km)에서만 쉴 수 있다. 기본 휴식 시간은 30분이다.

이 모든 조건을 준수하여 최종 목적지까지 도착했을 때, 의무 휴식을 포함한 총 소요 시간은 몇 시간 몇 분입니까? (분은 소숫점 첫째 자리에서 반올림)
"""


data = {
    # "model": "Qwen/Qwen3-0.6B",
    "model": "Qwen/Qwen3-1.7B",

    "messages": [
        {"role": "user", "content": "세종대왕 맥북 던짐 사건 알려줘"},
    ],
    # "messages": [
    #     {"role": "system", "content": system_prompt},
    #     {"role": "user", "content": content}
    # ],
    "temperature": 0.6,
    "max_tokens": 16384,
    "top_p": 0.95,
    "top_k": 20,
    "stream": False,
    "chat_template_kwargs": {"enable_thinking": True},
}

print(f"[URL] {URL}/v1/chat/completions")
print(f"[Model] {data['model']}")
print(f"[Thinking] {data['chat_template_kwargs']['enable_thinking']}")
print(f"[Prompt] {data['messages']}")
print("-" * 50)

start = time.time()
try:
    resp = requests.post(f"{URL}/v1/chat/completions", json=data, timeout=600)
    latency = time.time() - start

    print(f"[HTTP Status] {resp.status_code}")
    print(f"[Response Time] {latency:.2f}s")

    try:
        result = resp.json()
    except ValueError:
        print("[Body (not JSON)]")
        print(resp.text[:4000])
        raise

    if resp.status_code != 200 or "choices" not in result:
        print("[Body]")
        print(json.dumps(result, ensure_ascii=False, indent=2) if isinstance(result, dict) else result)
        raise SystemExit(1)

    choice = result["choices"][0]["message"]
    thinking = (choice.get("reasoning_content") or choice.get("reasoning") or "").strip()
    content = (choice.get("content") or "").strip()

    if thinking:
        print(f"\n[Reasoning]\n{thinking}")
    print(f"\n[Answer]\n{content}")

    if "usage" in result:
        print(f"\n[Usage] {result['usage']}")

except requests.exceptions.Timeout:
    print(f"[Error] Timeout: {URL}")
except requests.exceptions.ConnectionError:
    print(f"[Error] Connection failed: {URL} — Check Colab server/ngrok URL.")
except Exception as e:
    print(f"[Error] {e}")
