# Array Formula Puzzle (배열 수식 퍼즐)

엑셀/스프레드시트의 배열 수식과 조건부 함수를 활용한 LLM 논리 추론 평가 데이터셋입니다.

## 개요

실제 비즈니스 환경에서 자주 사용되는 스프레드시트 데이터 분석 문제를 기반으로 합니다. LLM이 테이블 데이터를 이해하고, 적절한 계산과 조건부 로직을 적용하여 정확한 답을 도출하는 능력을 평가합니다.

## 문제 유형

### 1. Lookup Query (조회 문제)

VLOOKUP, INDEX-MATCH, XLOOKUP 스타일의 데이터 조회 문제입니다.

**예시:**
```
[제품 테이블]
| id | 제품명 | 카테고리 | 단가 | 재고 |
|----|--------|----------|------|------|
| 1  | 사과   | 과일     | 1500 | 100  |
| 2  | 우유   | 유제품   | 2500 | 50   |
| 3  | 배     | 과일     | 2000 | 80   |

질문: '우유'의 단가는 얼마인가요?
정답: 2500
```

**난이도별 차이:**
- Easy: 단순 값 조회
- Medium: 조회 후 계산 (할인 적용가 등)
- Hard: 조건부 최대/최소값 찾기 (INDEX-MATCH + MAX/MIN)

### 2. Conditional Aggregation (조건부 집계)

SUMIF, COUNTIF, AVERAGEIF 스타일의 조건부 집계 문제입니다.

**예시:**
```
질문: '과일' 카테고리의 총 재고 수량은?
계산: 100 + 80 = 180
정답: 180
```

**난이도별 차이:**
- Easy: 단일 조건 COUNT/SUM
- Medium: 평균 계산, 재고 가치 합계
- Hard: 다중 조건 (COUNTIFS, SUMIFS)

### 3. Array Computation (배열 연산)

SUMPRODUCT 스타일의 복합 배열 연산 문제입니다.

**예시:**
```
질문: 모든 제품의 총 재고 가치(단가×재고의 합)는?
계산: 1500×100 + 2500×50 + 2000×80 = 150000 + 125000 + 160000 = 435000
정답: 435000
```

**난이도별 차이:**
- Easy: 단순 SUMPRODUCT
- Medium: 할인율 적용 계산
- Hard: 다중 테이블 참조 (주문 테이블과 제품 테이블 조인)

### 4. Multi-Condition (복합 조건)

SUMIFS, COUNTIFS, MAXIFS, MINIFS 스타일의 복합 조건 문제입니다.

**예시:**
```
질문: '과일' 카테고리 중 재고가 50개 이상인 제품의 최고 단가는?
필터: 과일 AND 재고>=50 → 사과(1500), 배(2000)
정답: 2000
```

**난이도별 차이:**
- Easy: 2개 조건 COUNT/SUM
- Medium: 2개 조건 + MAX/MIN
- Hard: 3개 이상 조건 + 다중 테이블

## 데이터 형식

### 퍼즐 JSON 구조

```json
{
  "id": "af_medium_lookup_query_a1b2c3d4",
  "type": "lookup_query",
  "difficulty": "medium",
  "seed": 12345,
  "tables": {
    "제품": {
      "columns": ["id", "제품명", "카테고리", "단가", "재고", "할인율"],
      "data": [
        {"id": 1, "제품명": "사과", "카테고리": "과일", "단가": 1500, "재고": 100, "할인율": 10},
        ...
      ]
    },
    "주문": {
      "columns": ["주문번호", "제품명", "지역", "수량", "분기"],
      "data": [...]
    }
  },
  "question": "'사과'의 할인 적용 단가는 얼마인가요?",
  "formula_hint": "VLOOKUP과 산술 연산을 조합하세요.",
  "answer": 1350,
  "answer_type": "number"
}
```

### 필드 설명

| 필드 | 설명 |
|------|------|
| `id` | 고유 식별자 (난이도_유형_해시) |
| `type` | 문제 유형 |
| `difficulty` | 난이도 (easy/medium/hard) |
| `seed` | 재현용 랜덤 시드 |
| `tables` | 테이블 데이터 (이름 → 컬럼/데이터) |
| `question` | 질문 텍스트 |
| `formula_hint` | 엑셀 수식 힌트 (선택적 사용) |
| `answer` | 정답 |
| `answer_type` | 답변 유형 (number/text) |

## 사용법

### 문제 생성

```python
from array_formula import generate_puzzle, generate_dataset, puzzle_to_prompt

# 단일 문제 생성
puzzle = generate_puzzle(
    difficulty='medium',
    problem_type='conditional_aggregation',
    seed=42
)

# 프롬프트 변환
prompt = puzzle_to_prompt(puzzle, include_hint=False)
print(prompt)
print(f"정답: {puzzle['answer']}")

# 데이터셋 생성 (난이도×유형별 각 10개)
puzzles = generate_dataset(num_puzzles_per_config=10, seed=2025)
```

### CLI로 생성

```bash
# 데이터셋 생성
cd guess
python array_formula.py --num 10 --seed 2025

# 데모 출력
python array_formula.py --demo
```

### 평가 실행

```bash
cd evaluation

# GPT-4o로 평가
python eval_array_formula.py --model gpt-4o

# Claude로 평가
python eval_array_formula.py --model claude-3-5-sonnet-20241022

# 특정 난이도/유형만 평가
python eval_array_formula.py --model gpt-4o --difficulty hard --type multi_condition

# 힌트 포함 평가
python eval_array_formula.py --model gpt-4o --hint
```

### 평가 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--model` | 평가 모델 | gpt-4o |
| `--dataset` | 데이터셋 경로 | ../evaluation_data/array_formula/array_formula_dataset.json |
| `--limit` | 평가 문제 수 제한 | 전체 |
| `--difficulty` | 특정 난이도만 평가 | 전체 |
| `--type` | 특정 유형만 평가 | 전체 |
| `--hint` | 수식 힌트 포함 | False |
| `--delay` | API 호출 간 딜레이(초) | 0.5 |

## 평가 메트릭

### Exact Match
정답과 정확히 일치하는 비율입니다.

### Numeric Close (수치 근접도)
숫자 답변의 경우 1% 오차 허용 기준 정확도입니다. 반올림/버림 차이로 인한 오류를 보정합니다.

### 결과 예시

```
============================================================
평가 결과 요약 - gpt-4o
============================================================
전체 정확도: 95/120 (79.2%)
수치 근접 정확도: 82.5%
평균 응답 시간: 2340ms

난이도별 정확도:
  easy: 38/40 (95.0%)
  medium: 32/40 (80.0%)
  hard: 25/40 (62.5%)

유형별 정확도:
  lookup_query: 28/30 (93.3%)
  conditional_aggregation: 26/30 (86.7%)
  array_computation: 22/30 (73.3%)
  multi_condition: 19/30 (63.3%)
============================================================
```

## 난이도 설계

| 난이도 | 테이블 크기 | 조건 수 | 테이블 수 | 예상 정확도 |
|--------|------------|---------|-----------|------------|
| Easy | 5-6행 | 1개 | 1개 | 90%+ |
| Medium | 7-9행 | 2개 | 1개 | 70-85% |
| Hard | 10-14행 | 3개+ | 1-2개 | 50-70% |

## 특징

### 1. 유일해 보장
모든 문제는 하나의 정답만 존재합니다. 테이블 데이터와 질문 조건이 모호함 없이 설계됩니다.

### 2. 자동 평가
정답이 숫자 또는 정확한 텍스트이므로 자동 평가가 가능합니다.

### 3. 재현 가능
시드 기반 생성으로 동일한 문제를 재생성할 수 있습니다.

### 4. 실용성
실제 비즈니스 데이터 분석에서 사용되는 패턴을 반영합니다.

## 확장 가능성

### 추가 예정 문제 유형
- **Formula Debugging**: 잘못된 수식 찾기
- **Spreadsheet Inference**: 숨겨진 셀 값 추론
- **Pivot Table**: 피벗 테이블 결과 예측

### 다국어 지원
현재 한국어 데이터만 지원하며, 영어/일본어 버전 추가 예정입니다.

## 라이선스

MIT License
