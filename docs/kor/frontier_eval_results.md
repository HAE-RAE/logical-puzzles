# Frontier 모델 평가 결과 (Claude-Opus-4.8 · GPT-5.5 · Gemini-3.1-Pro)

> **설정**: LiteLLM, `max_tokens=32768, reasoning_effort=medium`, 난이도 tier당 n=100.  
> **범위**: 32개 task(언어 분리) × 3 tier = 96 tier. `subway_ko`는 작업중 task라 참고용(논문 31-task 집계엔 제외).  
> **채점**: task별 결정론적 evaluator, 완전일치.

## 1. 전체 정확도

| 모델 | 전체 | easy | medium | hard |
|------|:---:|:---:|:---:|:---:|
| GPT-5.5 | 0.77 | 0.88 | 0.77 | 0.68 |
| Claude-Opus-4.8 | 0.71 | 0.85 | 0.68 | 0.60 |
| Gemini-3.1-Pro | 0.63 | 0.80 | 0.60 | 0.49 |

## 2. 카테고리별 평균 (EN→KO 전이 스펙트럼)

| 모델 | 직번역(11×2) | 언어특화(2×2) | 한국어전용(6) |
|------|:---:|:---:|:---:|
| GPT-5.5 | 0.74 | 0.96 | 0.77 |
| Claude-Opus-4.8 | 0.69 | 0.87 | 0.65 |
| Gemini-3.1-Pro | 0.60 | 0.79 | 0.65 |

## 3. EN vs KO (직번역+언어특화 13 task)

| 모델 | EN 평균 | KO 평균 | 격차(EN−KO) |
|------|:---:|:---:|:---:|
| GPT-5.5 | 0.77 | 0.77 | +0.002 |
| Claude-Opus-4.8 | 0.73 | 0.71 | +0.028 |
| Gemini-3.1-Pro | 0.65 | 0.60 | +0.054 |

## 4.1 Task별 정확도 — easy

| Task | 카테고리 | 언어 | GPT-5.5 | Claude-Opus-4.8 | Gemini-3.1-Pro |
|------|------|:--:|:--:|:--:|:--:|
| array_formula | 직번역 | EN | 0.83 | 0.80 | 0.80 |
| array_formula | 직번역 | KO | 0.76 | 0.78 | 0.80 |
| causal_dag | 직번역 | EN | 0.79 | 0.76 | 0.79 |
| causal_dag | 직번역 | KO | 0.78 | 0.75 | 0.79 |
| cipher | 언어특화 | EN | 0.95 | 0.98 | 0.91 |
| cipher | 언어특화 | KO | 0.97 | 0.87 | 0.95 |
| cryptarithmetic | 언어특화 | EN | 0.99 | 0.97 | 0.94 |
| cryptarithmetic | 언어특화 | KO | 0.99 | 0.99 | 0.99 |
| ferryman | 직번역 | EN | 0.55 | 0.68 | 0.56 |
| ferryman | 직번역 | KO | 0.66 | 0.69 | 0.33 |
| hanoi | 직번역 | EN | 0.95 | 0.72 | 0.28 |
| hanoi | 직번역 | KO | 0.94 | 0.78 | 0.26 |
| inequality | 직번역 | EN | 0.85 | 0.84 | 0.85 |
| inequality | 직번역 | KO | 0.84 | 0.83 | 0.84 |
| jamo | 한국어전용 | KO | 0.95 | 0.67 | 0.79 |
| kinship | 한국어전용 | KO | 0.68 | 0.59 | 0.74 |
| korean_units | 한국어전용 | KO | 0.99 | 0.95 | 0.54 |
| logic_grid | 직번역 | EN | 0.78 | 0.81 | 0.74 |
| logic_grid | 직번역 | KO | 0.69 | 0.67 | 0.64 |
| minesweeper | 직번역 | EN | 0.98 | 0.99 | 0.95 |
| minesweeper | 직번역 | KO | 0.96 | 0.99 | 0.91 |
| number_baseball | 직번역 | EN | 1.00 | 0.99 | 0.97 |
| number_baseball | 직번역 | KO | 0.99 | 1.00 | 0.93 |
| saju | 한국어전용 | KO | 0.99 | 0.94 | 0.90 |
| sat_puzzles | 직번역 | EN | 0.98 | 0.98 | 0.98 |
| sat_puzzles | 직번역 | KO | 0.92 | 0.95 | 0.98 |
| subway | 한국어전용 | KO | 0.78 | 0.74 | 0.82 |
| sudoku | 직번역 | EN | 1.00 | 1.00 | 1.00 |
| sudoku | 직번역 | KO | 1.00 | 1.00 | 1.00 |
| time | 한국어전용 | KO | 0.98 | 0.93 | 0.97 |
| yacht_dice | 직번역 | EN | 0.79 | 0.77 | 0.79 |
| yacht_dice | 직번역 | KO | 0.76 | 0.75 | 0.79 |

## 4.2 Task별 정확도 — medium

| Task | 카테고리 | 언어 | GPT-5.5 | Claude-Opus-4.8 | Gemini-3.1-Pro |
|------|------|:--:|:--:|:--:|:--:|
| array_formula | 직번역 | EN | 0.67 | 0.72 | 0.60 |
| array_formula | 직번역 | KO | 0.77 | 0.68 | 0.65 |
| causal_dag | 직번역 | EN | 0.61 | 0.57 | 0.61 |
| causal_dag | 직번역 | KO | 0.62 | 0.54 | 0.61 |
| cipher | 언어특화 | EN | 0.96 | 0.96 | 0.87 |
| cipher | 언어특화 | KO | 0.94 | 0.68 | 0.54 |
| cryptarithmetic | 언어특화 | EN | 0.95 | 0.91 | 0.89 |
| cryptarithmetic | 언어특화 | KO | 0.95 | 0.94 | 0.91 |
| ferryman | 직번역 | EN | 0.22 | 0.09 | 0.09 |
| ferryman | 직번역 | KO | 0.21 | 0.01 | 0.04 |
| hanoi | 직번역 | EN | 0.98 | 0.71 | 0.22 |
| hanoi | 직번역 | KO | 0.98 | 0.65 | 0.15 |
| inequality | 직번역 | EN | 0.94 | 0.88 | 0.56 |
| inequality | 직번역 | KO | 0.92 | 0.89 | 0.33 |
| jamo | 한국어전용 | KO | 0.88 | 0.37 | 0.56 |
| kinship | 한국어전용 | KO | 0.36 | 0.54 | 0.72 |
| korean_units | 한국어전용 | KO | 0.97 | 0.86 | 0.21 |
| logic_grid | 직번역 | EN | 0.12 | 0.23 | 0.06 |
| logic_grid | 직번역 | KO | 0.13 | 0.26 | 0.07 |
| minesweeper | 직번역 | EN | 1.00 | 0.97 | 0.95 |
| minesweeper | 직번역 | KO | 0.95 | 0.96 | 0.94 |
| number_baseball | 직번역 | EN | 0.99 | 0.95 | 0.74 |
| number_baseball | 직번역 | KO | 0.99 | 0.94 | 0.81 |
| saju | 한국어전용 | KO | 0.89 | 0.82 | 0.74 |
| sat_puzzles | 직번역 | EN | 0.94 | 0.80 | 0.93 |
| sat_puzzles | 직번역 | KO | 0.91 | 0.66 | 0.86 |
| subway | 한국어전용 | KO | 0.55 | 0.29 | 0.52 |
| sudoku | 직번역 | EN | 0.99 | 0.99 | 1.00 |
| sudoku | 직번역 | KO | 1.00 | 1.00 | 1.00 |
| time | 한국어전용 | KO | 0.94 | 0.76 | 0.94 |
| yacht_dice | 직번역 | EN | 0.58 | 0.57 | 0.62 |
| yacht_dice | 직번역 | KO | 0.60 | 0.54 | 0.54 |

## 4.3 Task별 정확도 — hard

| Task | 카테고리 | 언어 | GPT-5.5 | Claude-Opus-4.8 | Gemini-3.1-Pro |
|------|------|:--:|:--:|:--:|:--:|
| array_formula | 직번역 | EN | 0.60 | 0.58 | 0.46 |
| array_formula | 직번역 | KO | 0.50 | 0.60 | 0.41 |
| causal_dag | 직번역 | EN | 0.43 | 0.36 | 0.42 |
| causal_dag | 직번역 | KO | 0.41 | 0.38 | 0.34 |
| cipher | 언어특화 | EN | 0.95 | 0.85 | 0.76 |
| cipher | 언어특화 | KO | 0.90 | 0.53 | 0.40 |
| cryptarithmetic | 언어특화 | EN | 0.94 | 0.81 | 0.63 |
| cryptarithmetic | 언어특화 | KO | 0.98 | 0.92 | 0.74 |
| ferryman | 직번역 | EN | 0.43 | 0.20 | 0.44 |
| ferryman | 직번역 | KO | 0.27 | 0.31 | 0.10 |
| hanoi | 직번역 | EN | 0.34 | 0.27 | 0.27 |
| hanoi | 직번역 | KO | 0.41 | 0.29 | 0.29 |
| inequality | 직번역 | EN | 0.81 | 0.87 | 0.51 |
| inequality | 직번역 | KO | 0.93 | 0.87 | 0.32 |
| jamo | 한국어전용 | KO | 0.79 | 0.37 | 0.37 |
| kinship | 한국어전용 | KO | 0.33 | 0.77 | 0.88 |
| korean_units | 한국어전용 | KO | 0.96 | 0.85 | 0.27 |
| logic_grid | 직번역 | EN | 0.18 | 0.33 | 0.16 |
| logic_grid | 직번역 | KO | 0.32 | 0.33 | 0.18 |
| minesweeper | 직번역 | EN | 0.98 | 0.94 | 0.77 |
| minesweeper | 직번역 | KO | 0.94 | 0.92 | 0.73 |
| number_baseball | 직번역 | EN | 0.98 | 0.97 | 0.62 |
| number_baseball | 직번역 | KO | 0.97 | 0.96 | 0.57 |
| saju | 한국어전용 | KO | 0.83 | 0.56 | 0.71 |
| sat_puzzles | 직번역 | EN | 0.87 | 0.46 | 0.50 |
| sat_puzzles | 직번역 | KO | 0.80 | 0.28 | 0.33 |
| subway | 한국어전용 | KO | 0.10 | 0.05 | 0.12 |
| sudoku | 직번역 | EN | 0.94 | 0.99 | 0.88 |
| sudoku | 직번역 | KO | 1.00 | 1.00 | 0.96 |
| time | 한국어전용 | KO | 0.90 | 0.69 | 0.93 |
| yacht_dice | 직번역 | EN | 0.36 | 0.39 | 0.37 |
| yacht_dice | 직번역 | KO | 0.45 | 0.37 | 0.35 |

---

생성: `results/{model}/{task}/*.json` 파싱. accuracy = summary.overall.accuracy.
