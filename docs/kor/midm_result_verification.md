# Midm-2.0-Base-Instruct 평가 결과 검증 리포트

- 작성: 2026-07-18 (송영숙)
- 대상: `K-intelligence/Midm-2.0-Base-Instruct` (12B, non-reasoning) 본 실험 93개 태스크
- 결론 요약: **전체 평균 1.6점(100점 만점)은 파이프라인 오류가 아닌 유효한 실측값**

## 1. 실행 환경

| 항목 | 값 |
|---|---|
| 실행 일시 | 2026-07-17 20:20 ~ 07-18 01:26 (약 5시간) |
| 하드웨어 | Backend.AI H100 1장 (TP=1), vLLM 0.24.0 |
| 샘플링 | `temperature=0.8, top_p=0.7, top_k=20, repetition_penalty=1.05` (HF generation_config, model_configs.yaml 스펙) |
| max_tokens | 14336 (ctx 32k 모델 정책) |
| 태스크 | 93/93 완주, 실패 0 |
| 결과 위치 | `results/K-intelligence_Midm-2.0-Base-Instruct/` |

## 2. 점수 요약 (100점 만점)

| task | en_easy | en_med | en_hard | ko_easy | ko_med | ko_hard |
|---|---|---|---|---|---|---|
| array_formula | 3 | 1 | 3 | 4 | 0 | 2 |
| causal_dag | 6 | 0 | 1 | 4 | 0 | 0 |
| cipher | 0 | 0 | 0 | 0 | 0 | 0 |
| cryptarithmetic | 0 | 0 | 0 | 0 | 0 | 0 |
| ferryman | 0 | 0 | 0 | 0 | 0 | 0 |
| hanoi | 0 | 0 | 0 | 0 | 0 | 0 |
| inequality | 0 | 0 | 0 | 0 | 0 | 0 |
| jamo | - | - | - | 1 | 1 | 0 |
| kinship | - | - | - | 5 | 3 | 4 |
| korean_units | - | - | - | 0 | 0 | 0 |
| logic_grid | 1 | 0 | 0 | 0 | 0 | 0 |
| minesweeper | 0 | 0 | 0 | 0 | 0 | 0 |
| number_baseball | 0 | 0 | 0 | 0 | 0 | 0 |
| saju | - | - | - | 3 | 3 | 2 |
| sat_puzzles | 0 | 0 | 0 | 0 | 0 | 0 |
| sudoku | 1 | 0 | 0 | 0 | 0 | 0 |
| **time** | - | - | - | **65** | 15 | 8 |
| yacht_dice | 3 | 1 | 0 | 11 | 1 | 1 |

**전체 평균 1.6점.** 유일한 정상 밴드 점수는 time_ko_easy(65점).

## 3. "왜 이렇게 낮은가" — 3단계 검증

점수가 낮은 원인이 (a) 생성 실패/잘림, (b) 답 추출(파싱) 실패, (c) 실제 오답 중 무엇인지 원 데이터(CSV)로 확인했다.

### 3-1. 생성 실패/잘림 아님

`finish_reason` 분포: 대부분 태스크에서 `stop` 96~100%. API 에러 0건, 빈 응답 0건.
(예: inequality_en_easy = stop 96 / length 4, sudoku_en_easy = stop 100)

### 3-2. 파싱 실패 아님

모든 검사 샘플에서 모델이 `Answer: ...` 형식으로 답을 완결했고, evaluator가 그 답을 정확히 추출했다. 추출 답의 **형식**은 정답과 완전히 일치한다.

sat_puzzles 예 (형식 일치, 값 불일치):

```
정답: {'Alice': False, 'Leo': True,  'Emma': True,  'David': True, ...}
추출: {'Alice': False, 'Leo': True,  'Emma': False, 'David': True, ...}
```

### 3-3. 실제 오답 (모델의 논리 능력 문제)

- **sudoku_en_easy**: 모델이 채운 최종 그리드에 같은 행에 9가 두 번 등장 (규칙 위반 풀이를 자신 있게 제출) → 좌표값 오답
- **inequality_en_easy**: 정답 `768254913` vs 모델 `135249876` — 제약 검증 없이 그럴듯한 순열 제출
- **time_ko_easy (0.65, 정상)**: 단순 날짜 산수는 정확히 수행 → 생성·파싱 경로가 정상임을 교차 확인

## 4. 해석

1. **exact-match 채점의 특성**: sat_puzzles처럼 변수 8개를 모두 맞혀야 정답인 태스크는 "절반쯤 맞히는" 모델이 구조적으로 0.00이 된다. 부분 점수가 없으므로 저성능 모델의 점수는 급락한다.
2. **non-reasoning 12B의 한계**: Midm은 thinking 모드가 없어 다단계 논리 검증이 필요한 본 벤치마크에서 구조적으로 불리하다. reasoning을 켠 EXAONE-4.0-32B도 평균 12점에 그친 것과 같은 방향이며, Midm은 크기·모드 모두 불리해 더 극단적이다.
3. **문화 특화 ≠ 논리 추론**: 한국어 특화 카테고리(jamo/saju/kinship/korean_units)에서도 0~5점으로, 한국어 지식만으로는 풀리지 않음을 보여준다. time_ko_easy만 산수 수준이라 예외.

## 5. 재현 정보

- 러너: `run/eval/eval_midm_1gpu.sh` (vLLM TP=1, resume 지원)
- 검증 스크립트: CSV의 `finish_reason`/`filtered_resps`/`resps` 컬럼 대조 (evaluation/run.py 출력 스키마)
- 원 데이터: 태스크별 CSV에 문항 단위 질문/정답/모델응답/추출답/finish_reason 전체 보존

## 부록. DeepSeek-V4-Flash의 ferryman truncation 노트 (진행 중 실험, 2026-07-18)

Midm과 달리 **DeepSeek-V4-Flash(OpenRouter, thinking on)의 ferryman 저점수는 절반 이상이 토큰 한도 잘림**이다. API 에러는 아니다:

| 태스크 | 점수 (100점) | finish_reason=length (잘림 비율) |
|---|---|---|
| ferryman_en_medium | 0 | **54%** |
| ferryman_en_hard | 2 | **52%** |
| ferryman_ko_easy | 23 | 23% |

즉 문항의 절반에서 thinking이 `max_tokens=32768`을 다 쓰고도 답에 못 도달했다. ferryman이 탐색형 계획 문제라 DeepSeek이 thinking을 극단적으로 길게 쓰는 패턴이다.

**다만 이는 재실행 사유가 아니다.** 32768은 이미 `model_configs.yaml`의 본 실험 통일 스펙 상한이고, config에 이 상황에 대한 정책이 명시되어 있다: "finish_reason 기반 truncation rate를 태스크×난이도별 집계, 유의한 셀은 결과표에 각주 처리." 현재 파이프라인이 finish_reason을 전부 기록하고 있으므로 정책대로 처리 가능한 상태다. 전 모델 완주 후 태스크×난이도별 truncation rate 표를 집계해 각주 대상 셀을 확정한다.
