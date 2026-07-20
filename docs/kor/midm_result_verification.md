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

## 부록. 전 모델 truncation 집계 (최종, 2026-07-19)

Qwen3.5-27B·DeepSeek-V4-Flash(OpenRouter) 완주 후, config 정책("finish_reason 기반
truncation rate를 태스크×난이도별 집계, 유의한 셀은 결과표에 각주 처리")에 따라
**잘림(finish_reason=length) 비율 ≥20% 셀**을 집계했다. 에러 행은 세 모델 모두 0건
(에러 발생 태스크는 전량 재실행)이다.

**Qwen3.5-27B** (22개 셀):

| 태스크 | 잘림 비율 |
|---|---|
| cryptarithmetic_en_hard | 44% |
| sat_puzzles_en_hard | 39% |
| cryptarithmetic_en_medium | 34% |
| minesweeper_en_hard | 33% |
| cipher_en_easy | 27% |
| hanoi_en_hard | 27% |
| cipher_en_hard | 26% |
| cipher_ko_medium | 26% |
| cryptarithmetic_ko_hard | 26% |
| minesweeper_en_medium | 26% |
| cipher_ko_hard | 25% |
| number_baseball_en_hard | 25% |
| inequality_en_hard | 24% |
| sudoku_en_hard | 23% |
| cipher_en_medium | 22% |
| cryptarithmetic_en_easy | 22% |
| minesweeper_ko_hard | 22% |
| sat_puzzles_ko_hard | 22% |
| causal_dag_en_hard | 21% |
| causal_dag_ko_hard | 21% |
| sat_puzzles_ko_medium | 21% |
| cryptarithmetic_ko_medium | 20% |

**DeepSeek-V4-Flash** (29개 셀):

| 태스크 | 잘림 비율 |
|---|---|
| ferryman_ko_hard | 64% |
| ferryman_en_medium | 54% |
| ferryman_en_hard | 52% |
| minesweeper_ko_hard | 46% |
| ferryman_ko_medium | 43% |
| sat_puzzles_en_hard | 43% |
| sat_puzzles_ko_hard | 42% |
| hanoi_en_hard | 40% |
| ferryman_en_easy | 37% |
| array_formula_ko_hard | 36% |
| minesweeper_en_hard | 33% |
| minesweeper_ko_medium | 33% |
| number_baseball_ko_hard | 33% |
| cryptarithmetic_en_hard | 32% |
| hanoi_ko_hard | 32% |
| cipher_ko_hard | 30% |
| causal_dag_ko_hard | 29% |
| number_baseball_en_hard | 29% |
| sudoku_ko_hard | 29% |
| sudoku_en_hard | 27% |
| number_baseball_ko_medium | 26% |
| array_formula_en_hard | 25% |
| inequality_en_hard | 24% |
| ferryman_ko_easy | 23% |
| sat_puzzles_ko_medium | 22% |
| causal_dag_ko_medium | 21% |
| sat_puzzles_en_medium | 21% |
| cryptarithmetic_ko_hard | 20% |
| kinship_ko_hard | 20% |

**Midm-2.0** (2개 셀):

| 태스크 | 잘림 비율 |
|---|---|
| korean_units_ko_hard | 35% |
| array_formula_ko_easy | 21% |

패턴: hard 난이도 + 탐색형(ferryman·hanoi·minesweeper)·기호조작형(cipher·cryptarithmetic)
태스크에 잘림이 집중된다. `max_tokens=32768`은 본 실험 통일 스펙 상한이므로 재실행이 아니라
각주 대상이다. Midm은 출력이 짧아(non-reasoning) 잘림이 거의 없다.

참고 관찰: Qwen3.5-27B의 일부 태스크(causal_dag, logic_grid)는 시트1의 로컬 vLLM(BF16)
실측 대비 큰 폭으로 낮았다. 잘림(10~21%)만으로는 설명되지 않는 격차로, 서빙 환경
(로컬 BF16 vs OpenRouter 프로바이더 서빙)에 대한 민감도가 태스크별로 크게 다름을 시사한다.
단, 시트1은 설정 변경으로 무효 처리된 기준이므로 직접 비교보다는 방법론 각주로 기록한다.
