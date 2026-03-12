# Array Formula Puzzle Generator

Excel 배열 수식 기반 논리 퍼즐 생성기. 영어(`generation/array_formula_en.py`)와 한국어(`generation/array_formula_ko.py`) 두 버전을 지원합니다.

## 개요

- **난이도**: easy / medium / hard (각 100문항, 총 300문항)
- **문제 유형 4가지** (난이도별 25문항씩):
  - `lookup_query` — INDEX-MATCH, VLOOKUP 스타일 데이터 조회
  - `conditional_aggregation` — SUMIF, COUNTIF 스타일 조건부 집계
  - `array_computation` — SUMPRODUCT 스타일 배열 연산
  - `multi_condition` — SUMIFS, MAXIFS 스타일 다중 조건 문제
- **테이블**: 상품(Products), 주문(Orders), 고객(Customers) 3개 테이블
- **평가 모델**: Gemini 3 Flash Preview (temp=1.0)

## 난이도별 설정

| 설정 | Easy | Medium | Hard |
|------|------|--------|------|
| 상품 수 | 28-35 | 38-50 | 50-65 |
| 주문 수 | 35-50 | 55-80 | 80-120 |
| 고객 수 | 10-14 | 14-18 | 16-20 |
| 카테고리 수 | 6 | 8 | 8 |
| 지역 수 | 6 | 7 | 8 |
| 추론 단계 | 2-3 | 4-6 | 6-10+ |

## 문제 난이도 설계

### Easy
- 기본 조회, 필터링, 합계 계산 (2-3단계)
- 가중치 기반 선택: 기존 5개 템플릿(weight=1) + 신규 3개 템플릿(weight=2)
- 신규 템플릿: 할인가 계산, 중앙값 기반 필터, 지역별 할인 매출 등

### Medium
- 그룹화, 순위, 비율 계산 (4-6단계)
- 기존 쉬운 템플릿 2개 교체 + 신규 1개 추가
- 패턴: 고유 상품 수 순위, 평균 초과 필터 → 할인 매출, 카테고리별 매출 범위

### Hard
- 다단계 추론, 피벗, 중첩 제외, 조건부 평균의 평균 (6-10+ 단계)
- 기존 템플릿 ~60% 교체
- 패턴:
  - **Pattern A**: 계산된 임계값을 필터로 사용 (중앙값 소비자 분할 등)
  - **Pattern B**: 중첩 제외 + 재순위 (상위 N 제외 후 지역별 평균)
  - **Pattern C**: 조건부 avg-of-avg (골드 전용 vs 전체 비교)
  - **Pattern D**: 변동계수(CV), 할인 보존율 범위, 등급×카테고리 피벗 비율

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
from generation.array_formula_ko import generate_dataset, puzzle_to_prompt
import json
puzzles = generate_dataset(num_per_difficulty=100, seed=42)
with open('data/json/array_formula_ko.jsonl', 'w') as f:
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
python evaluation/run.py --tasks array_formula_en array_formula_ko --async --max-concurrent 30

# 특정 난이도만
python evaluation/run.py --tasks array_formula --difficulty hard --async
```

## 평가 결과 (Gemini 3 Flash Preview)

### 최종 결과 (Iteration 2, 2026-02-07)

| Difficulty | EN | KR |
|------------|------|------|
| Easy | 83% | 77% |
| Medium | 49% | 38% |
| Hard | 29% | 24% |
| **Overall** | **53.7%** | **46.3%** |



## Change Log

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
  array_formula_en.py       # 영어 생성기
  array_formula_ko.py       # 한국어 생성기
evaluation/
  evaluators/array_formula.py  # 평가기 (text+number 답 처리)
  run.py                       # 평가 실행기
data/json/
  array_formula.jsonl          # 영어 데이터셋 (300문항)
  array_formula_ko.jsonl       # 한국어 데이터셋 (300문항)
results/                       # 평가 결과 저장
```
