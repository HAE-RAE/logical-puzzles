# Array Formula Puzzle Generator

Excel 배열 수식 기반 논리 퍼즐 생성기. 영어(`generation/array_formula.py`)와 한국어(`generation/array_formula_korean.py`) 두 버전을 지원합니다.

## 개요

- **난이도**: easy / medium / hard (각 200문항, 총 600문항)
- **문제 유형 4가지** (난이도별 50문항씩):
  - `lookup_query` — INDEX-MATCH, VLOOKUP 스타일 데이터 조회
  - `conditional_aggregation` — SUMIF, COUNTIF 스타일 조건부 집계
  - `array_computation` — SUMPRODUCT 스타일 배열 연산
  - `multi_condition` — SUMIFS, MAXIFS 스타일 다중 조건 문제
- **테이블**: 상품(Products), 주문(Orders), 고객(Customers) 3개 테이블
- **평가 모델**: Gemini 3 Flash Preview (temp=1.0)

## 난이도별 설정

| 설정 | Easy | Medium | Hard |
|------|------|--------|------|
| 상품 수 | 20-28 | 35-45 | 48-58 |
| 주문 수 | 25-40 | 50-70 | 75-110 |
| 고객 수 | 8-12 | 12-16 | 15-19 |
| 카테고리 수 | 5 | 7 | 8 |
| 지역 수 | 5 | 7 | 8 |
| 목표 정확도 | ~75% | ~50% | ~25% |

## 문제 난이도 설계

### Easy (~75%)
- 기본 조회, 필터링, 합계 계산 (2-3단계)
- 가중치 기반 선택: 기존 5개 템플릿(weight=1) + 신규 3개 템플릿(weight=2)
- 신규 템플릿: 할인가 계산, 중앙값 기반 필터, 지역별 할인 매출 등

> 예시: *"What is the total sales revenue for orders placed in 'Sejong'? (Look up price from Products table)"*

### Medium (~50%)
- 그룹화, 순위, 비율 계산 (4-6단계)
- 기존 쉬운 템플릿 2개 교체 + 신규 1개 추가
- 패턴: 고유 상품 수 순위, 평균 초과 필터 → 할인 매출, 카테고리별 매출 범위

> 예시: *"What is the discount rate of the product most ordered (by quantity) by 'Gold' membership customers?"*

### Hard (~25%)
- 다단계 추론, 피벗, 중첩 제외, 조건부 평균의 평균 (6-10+ 단계)
- 기존 템플릿 ~60% 교체
- 패턴:
  - **Pattern A**: 계산된 임계값을 필터로 사용 (중앙값 소비자 분할 등)
  - **Pattern B**: 중첩 제외 + 재순위 (상위 N 제외 후 지역별 평균)
  - **Pattern C**: 조건부 avg-of-avg (골드 전용 vs 전체 비교)
  - **Pattern D**: 변동계수(CV), 할인 보존율 범위, 등급×카테고리 피벗 비율

> 예시: *"For each region, compute total discounted revenue for ALL customers and for customers who joined in 2021 or later. What is the maximum difference (all - post-2021) across regions?"*

## 데이터 생성

```bash
# 영어 300문항 생성
python -c "
from generation.array_formula import generate_dataset, puzzle_to_prompt
import json
puzzles = generate_dataset(num_per_difficulty=100, seed=42)
with open('data/json/array_formula.jsonl', 'w') as f:
    for p in puzzles:
        record = {'id': p['id'], 'question': puzzle_to_prompt(p), 'answer': p['answer'],
                  'difficulty': p['difficulty'], 'solution': p.get('solution',''),
                  'type': p['type'], 'answer_type': p.get('answer_type','number'),
                  'tables': p['tables'], 'seed': p.get('seed')}
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
"

# 한국어 300문항 생성
python -c "
from generation.array_formula_korean import generate_dataset, puzzle_to_prompt
import json
puzzles = generate_dataset(num_per_difficulty=100, seed=42)
with open('data/json/array_formula_korean.jsonl', 'w') as f:
    for p in puzzles:
        record = {'id': p['id'], 'question': puzzle_to_prompt(p), 'answer': p['answer'],
                  'difficulty': p['difficulty'], 'solution': p.get('solution',''),
                  'type': p['type'], 'answer_type': p.get('answer_type','number'),
                  'tables': p['tables'], 'seed': p.get('seed')}
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
"
```

## 평가 실행

```bash
# 영어 + 한국어 동시 평가
python evaluation/run.py --tasks array_formula array_formula_korean --async --max-concurrent 30

# 특정 난이도만
python evaluation/run.py --tasks array_formula --difficulty hard --async
```

## 평가 결과 (Gemini 3 Flash Preview)

### 최종 결과 (v5 Quartile Calibration, 2026-03-26)

#### 절대 정확도

| Difficulty | Target | EN (200문항) | KR (200문항) |
|------------|--------|-------------|-------------|
| Easy | ~75% | 77.5% | 81.0% |
| Medium | ~50% | 55.0% | 50.5% |
| Hard | ≤25% | 24.0% | 26.5% |
| **Overall** | — | **52.2%** | **52.7%** |

#### 상대적 난이도 분석: 언어 간 난이도 일관성

절대 정확도와 별개로, 난이도 간 **상대적 하락 비율**이 언어에 걸쳐 일관되는지를 검증했다. 각 난이도의 정확도를 해당 언어의 easy 정확도로 나누어 상대 비율을 산출하고, 단계별 하락폭을 비교했다.

| Difficulty | EN 상대 비율 | KR 상대 비율 |
|------------|-------------|-------------|
| Easy (기준) | 1.00 | 1.00 |
| Medium | 0.71 (55.0/77.5) | 0.62 (50.5/81.0) |
| Hard | 0.31 (24.0/77.5) | 0.33 (26.5/81.0) |

| 구간 | EN 하락폭 | KR 하락폭 |
|------|----------|----------|
| easy → medium | −29% | −38% |
| medium → hard | −56% | −48% |

**관찰:**
1. **난이도 분리는 양 언어에서 충분하다.** 매 단계마다 최소 29% 이상의 상대적 하락이 발생하며, 단순한 우연이나 노이즈로는 설명되지 않는 수준의 난이도 구분이 확인된다.
2. **Hard 수준의 언어 간 일관성이 높다.** easy 대비 hard의 상대 비율이 EN 0.31, KR 0.33으로 거의 동일하다. 이는 hard 난이도에서의 추론 복잡도가 언어와 무관하게 유사한 수준으로 작동함을 시사한다.
3. **Medium 수준에서 한국어가 상대적으로 더 어렵다.** EN 0.71 vs KR 0.62로, 한국어 medium 템플릿이 영어 대비 약 9%p 더 큰 하락을 보인다. 원인에 대해서는 전체 태스크 데이터를 종합적으로 분석한 후 판단이 필요하다.
4. **동일 파라미터로 양 언어 커버 가능.** 절대 정확도 기준으로 EN과 KR 모두 목표 범위(±10%) 안에 있으며, 상대적 하락 패턴도 유사하므로 언어별 별도 파라미터 조정 없이 단일 설정으로 운용할 수 있다.

이 결과는 난이도 캘리브레이션이 특정 언어에 과적합(overfit)되지 않았음을 보여주며, 동일한 생성 파라미터를 다른 언어로 확장할 때에도 난이도 구조가 보존될 가능성을 시사한다.

### 이전 결과 (Iteration 2, 2026-02-07)

| Difficulty | EN | KR |
|------------|------|------|
| Easy | 83% | 77% |
| Medium | 49% | 38% |
| Hard | 29% | 24% |
| **Overall** | **53.7%** | **46.3%** |

## Change Log

### v5: Quartile Calibration
**Date:** 2026-03-26

**Changes:**
- 3분위 목표 정확도 기반 난이도 캘리브레이션 (easy ~75%, medium ~50%, hard ≤25%)
- 데이터 크기 재조정: easy(20-28), medium(35-45), hard(48-58) 상품
- 주문 수 재조정: easy(25-40), medium(50-70), hard(75-110)
- 카테고리/지역 수 조정: easy(5/5), medium(7/7), hard(8/8)
- 난이도당 200문항, 총 600문항으로 확대 (EN + KR 각 600문항)
- 캘리브레이션 모델: Gemini 3 Flash Preview (temp=1.0)
- 프로빙 기반 파라미터 탐색: 10개 설정 × 20문제로 사전 검증 후 최종 파라미터 결정
- EN/KR 동일 파라미터 적용, 언어 간 난이도 일관성 검증 완료

**Results:** EN Easy 77.5%, Medium 55.0%, Hard 24.0% / KR Easy 81.0%, Medium 50.5%, Hard 26.5%

### Iteration 2: Difficulty Increase
**Date:** 2026-02-07

**Changes:**
- 테이블 크기 증가: easy(28-35), medium(38-50), hard(50-65) 상품
- 주문 수 증가: easy(35-50), medium(55-80), hard(80-120)
- 상품명 40개 → 85개로 확장 (EN + KR)
- 가중치 기반 템플릿 선택 (`_weighted_choice`) 도입
- Easy: 문제 유형별 신규 3개 템플릿 추가 (weight=2)
- Medium: 쉬운 템플릿 2개 교체 + 신규 1개 추가
- Hard: ~60% 템플릿 교체 (중앙값 분할, 피벗 비율, 조건부 avg-of-avg, 변동계수 등)
- 헬퍼 함수 추가: `_group_distinct_count()`, `_std_dev()`, `_weighted_choice()`
- 검증: 2400문항(EN) + 2400문항(KR) 전수 생성 테스트 통과

### Iteration 1: Table Size + Template Redesign
**Date:** 2026-02-07

**Changes:**
- 테이블 크기 증가 (easy: 20-26, medium: 30-40, hard: 42-55)
- 주문 수 증가 (easy: 24-36, medium: 42-65, hard: 65-95)
- Easy: 할인가 주문 가치, 2위 조회, 카테고리 주문 통계 추가
- Medium: weighted-avg를 분기/지역 합계로 교체; conditional_agg에 비율/비교 도입
- Hard: 매출 가중 할인율, 피벗, 제외 후 합산; avg-of-avg 트랩, 변동계수, 3중 조건 비율; 5+ 단계 체인 조회
- 헬퍼 함수 추가: `_group_avg()`, `_median()`

**Results:** EN Easy 99%, Medium 84%, Hard 79%

## 파일 구조

```
generation/
  array_formula.py          # 영어 생성기
  array_formula_korean.py   # 한국어 생성기
evaluation/
  evaluators/array_formula.py  # 평가기 (text+number 답 처리)
  run.py                       # 평가 실행기
data/json/
  array_formula.jsonl          # 영어 데이터셋 (600문항)
  array_formula_korean.jsonl   # 한국어 데이터셋
results/                       # 평가 결과 저장
```
