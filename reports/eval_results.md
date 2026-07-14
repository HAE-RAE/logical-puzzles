# 추론 퍼즐 벤치마크 결과

태스크당 100문제 정확도. 로컬 vLLM, reasoning on, temp 0.6, top_p 0.95.

그래프: [`reports/eval_results.jpg`](eval_results.jpg) · 대시보드: [`reports/eval_results.html`](eval_results.html)

> **커버리지 주의:** gpt-oss-120b만 전체 18개 유형(93 태스크)을 평가했고, gemma·EXAONE는 8개 공통 유형(45 태스크)까지만 평가됨. 미평가는 `—`.

## 종합 · 난이도별 (%)

| Model | tasks | 종합 | easy | medium | hard |
|---|---|---|---|---|---|
| gpt-oss-120b | 93 | **39** | 60 | 37 | 20 |
| gemma-4-31b-it | 45 | **31** | 52 | 25 | 16 |
| EXAONE-4.0-32B | 45 | **12** | 28 | 4 | 3 |

## 유형별 정확도 (%, 양 언어·전 난이도 평균)

| Category | gpt-oss-120b | gemma-4-31b-it | EXAONE-4.0-32B |
|---|---|---|---|
| array_formula | 29 | 32 | 7 |
| causal_dag | 38 | 48 | 21 |
| cipher | 28 | 7 | 18 |
| cryptarithmetic | 24 | 38 | 6 |
| ferryman | 15 | 22 | 0 |
| hanoi | 42 | 13 | 3 |
| inequality | 56 | 66 | 56 |
| jamo | 7 | 15 | 0 |
| kinship | 6 | — | 5 |
| korean_units | 68 | — | — |
| logic_grid | 26 | — | 44 |
| minesweeper | 43 | — | — |
| number_baseball | 50 | — | — |
| saju | 14 | — | — |
| sat_puzzles | 30 | — | — |
| sudoku | 99 | — | — |
| time | 55 | — | — |
| yacht_dice | 52 | — | — |

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
| kinship_ko_easy | 7 | — | 12 |
| kinship_ko_medium | 4 | — | 1 |
| kinship_ko_hard | 6 | — | 3 |
| korean_units_ko_easy | 92 | — | — |
| korean_units_ko_medium | 84 | — | — |
| korean_units_ko_hard | 27 | — | — |
| logic_grid_en_easy | 54 | — | 44 |
| logic_grid_en_medium | 6 | — | — |
| logic_grid_en_hard | 12 | — | — |
| logic_grid_ko_easy | 59 | — | — |
| logic_grid_ko_medium | 9 | — | — |
| logic_grid_ko_hard | 14 | — | — |
| minesweeper_en_easy | 78 | — | — |
| minesweeper_en_medium | 47 | — | — |
| minesweeper_en_hard | 6 | — | — |
| minesweeper_ko_easy | 77 | — | — |
| minesweeper_ko_medium | 37 | — | — |
| minesweeper_ko_hard | 12 | — | — |
| number_baseball_en_easy | 75 | — | — |
| number_baseball_en_medium | 45 | — | — |
| number_baseball_en_hard | 24 | — | — |
| number_baseball_ko_easy | 82 | — | — |
| number_baseball_ko_medium | 56 | — | — |
| number_baseball_ko_hard | 20 | — | — |
| saju_ko_easy | 19 | — | — |
| saju_ko_medium | 13 | — | — |
| saju_ko_hard | 11 | — | — |
| sat_puzzles_en_easy | 57 | — | — |
| sat_puzzles_en_medium | 27 | — | — |
| sat_puzzles_en_hard | 5 | — | — |
| sat_puzzles_ko_easy | 53 | — | — |
| sat_puzzles_ko_medium | 29 | — | — |
| sat_puzzles_ko_hard | 6 | — | — |
| sudoku_en_easy | 99 | — | — |
| sudoku_en_medium | 100 | — | — |
| sudoku_en_hard | 98 | — | — |
| sudoku_ko_easy | 99 | — | — |
| sudoku_ko_medium | 100 | — | — |
| sudoku_ko_hard | 100 | — | — |
| time_ko_easy | 72 | — | — |
| time_ko_medium | 55 | — | — |
| time_ko_hard | 39 | — | — |
| yacht_dice_en_easy | 76 | — | — |
| yacht_dice_en_medium | 53 | — | — |
| yacht_dice_en_hard | 35 | — | — |
| yacht_dice_ko_easy | 73 | — | — |
| yacht_dice_ko_medium | 47 | — | — |
| yacht_dice_ko_hard | 25 | — | — |
