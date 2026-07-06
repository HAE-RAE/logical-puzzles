# 추론 퍼즐 벤치마크 결과

8개 퍼즐 유형 × 한/영 × 난이도(easy·medium·hard), 태스크당 100문제 정확도. 로컬 vLLM, reasoning on, temp 0.6, top_p 0.95.

시각화(표+막대그래프): [`reports/eval_results.html`](eval_results.html)

## 유형별 정확도 (%, 양 언어·전 난이도 평균)

| Model | Array | Causal | Cipher | Crypt | Ferry | Hanoi | Ineq | Jamo | **종합** |
|---|---|---|---|---|---|---|---|---|---|
| gpt-oss-120b | 29 | 38 | 28 | 24 | 15 | 42 | 56 | 7 | **31** |
| gemma-4-31b-it | 32 | 48 | 7 | 38 | 22 | 13 | 66 | 15 | **31** |
| EXAONE-4.0-32B | 7 | 21 | 18 | 6 | 0 | 3 | 56 | 0 | **12** |

## 난이도별 (%)

| Model | easy | medium | hard |
|---|---|---|---|
| gpt-oss-120b | 53 | 28 | 12 |
| gemma-4-31b-it | 52 | 25 | 16 |
| EXAONE-4.0-32B | 28 | 5 | 3 |

## 태스크별 정확도 (%)

| Task | gpt-oss-120b | gemma-4-31b-it | EXAONE-4.0-32B |
|---|---|---|---|
| array_formula_en_easy | 53 | 53 | 17 |
| array_formula_en_medium | 32 | 31 | 3 |
| array_formula_en_hard | 6 | 10 | 0 |
| array_formula_ko_easy | 47 | 56 | 18 |
| array_formula_ko_medium | 29 | 30 | 4 |
| array_formula_ko_hard | 6 | 9 | 2 |
| causal_dag_en_easy | 71 | 79 | 58 |
| causal_dag_en_medium | 28 | 44 | 3 |
| causal_dag_en_hard | 11 | 29 | 3 |
| causal_dag_ko_easy | 72 | 76 | 56 |
| causal_dag_ko_medium | 33 | 40 | 5 |
| causal_dag_ko_hard | 14 | 20 | 3 |
| cipher_en_easy | 89 | 11 | 57 |
| cipher_en_medium | 55 | 9 | 41 |
| cipher_en_hard | 22 | 11 | 12 |
| cipher_ko_easy | 3 | 11 | 0 |
| cipher_ko_medium | 0 | 0 | 0 |
| cipher_ko_hard | 0 | 0 | 0 |
| cryptarithmetic_en_easy | 63 | 66 | 25 |
| cryptarithmetic_en_medium | 32 | 31 | 6 |
| cryptarithmetic_en_hard | 19 | 6 | 2 |
| cryptarithmetic_ko_easy | 16 | 76 | 0 |
| cryptarithmetic_ko_medium | 10 | 36 | 0 |
| cryptarithmetic_ko_hard | 2 | 14 | 0 |
| ferryman_en_easy | 45 | 73 | 1 |
| ferryman_en_medium | 1 | 9 | 0 |
| ferryman_en_hard | 0 | 0 | 0 |
| ferryman_ko_easy | 40 | 46 | 1 |
| ferryman_ko_medium | 3 | 2 | 0 |
| ferryman_ko_hard | 0 | 0 | 0 |
| hanoi_en_easy | 65 | 13 | 0 |
| hanoi_en_medium | 49 | 1 | 0 |
| hanoi_en_hard | 14 | 23 | 7 |
| hanoi_ko_easy | 55 | 15 | 0 |
| hanoi_ko_medium | 51 | 2 | 0 |
| hanoi_ko_hard | 17 | 25 | 9 |
| inequality_en_easy | 85 | 85 | 85 |
| inequality_en_medium | 45 | 65 | — |
| inequality_en_hard | 28 | 40 | 3 |
| inequality_ko_easy | 82 | 84 | 80 |
| inequality_ko_medium | 54 | 71 | — |
| inequality_ko_hard | 39 | 50 | — |
| jamo_ko_easy | 16 | 37 | — |
| jamo_ko_medium | 4 | 6 | 0 |
| jamo_ko_hard | 2 | 2 | 0 |

> EXAONE-4.0-32B: inequality(en_medium·ko_medium·ko_hard)·jamo_ko_easy 미평가(—). 종합/평균은 평가된 태스크만 반영.
