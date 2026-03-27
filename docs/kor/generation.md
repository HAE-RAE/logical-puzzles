# Logical-Puzzles 데이터셋 생성 가이드

28개 논리 퍼즐 생성 스크립트(영·한 병행 작업 포함) 모음입니다.

## 📋 지원 작업

| 작업 | 스크립트 | 난이도 | 기본 개수 |
|------|--------|------------|---------------|
| Array Formula (EN) | `array_formula_en.py` | Easy, Medium, Hard | 100 |
| Array Formula (KO) | `array_formula_ko.py` | Easy, Medium, Hard | 100 |
| Causal DAG (KO) | `causal_dag_ko.py` | Easy, Medium, Hard | 300 |
| Causal DAG (EN) | `causal_dag_en.py` | Easy, Medium, Hard | 300 |
| Cipher (KO) | `cipher_ko.py` | Easy, Medium, Hard | 100 |
| Cipher (EN) | `cipher_en.py` | Easy, Medium, Hard | 100 |
| Cryptarithmetic (EN) | `cryptarithmetic_en.py` | Easy, Medium, Hard | 400 |
| Cryptarithmetic (KO) | `cryptarithmetic_ko.py` | Easy, Medium, Hard | 400 |
| Ferryman (EN) | `ferryman_en.py` | Easy, Medium, Hard | 100 |
| Ferryman (KO) | `ferryman_ko.py` | Easy, Medium, Hard | 100 |
| Hanoi (EN) | `hanoi_en.py` | - | 100 |
| Hanoi (KO) | `hanoi_ko.py` | - | 100 |
| Inequality (EN) | `inequality_en.py` | Easy, Medium, Hard | 400 |
| Inequality (KO) | `inequality_ko.py` | Easy, Medium, Hard | 400 |
| Kinship | `kinship.py` | - | 100 |
| Kinship Vision | `kinship_vision.py` | - | 100 |
| Logic Grid (KO) | `logic_grid_ko.py` | Easy, Medium, Hard | 300 |
| Logic Grid (EN) | `logic_grid_en.py` | Easy, Medium, Hard | 300 |
| Minesweeper (EN) | `minesweeper_en.py` | Easy, Medium, Hard | - |
| Minesweeper (KO) | `minesweeper_ko.py` | Easy, Medium, Hard | - |
| Number Baseball (EN) | `number_baseball_en.py` | Easy, Medium, Hard | 400 |
| Number Baseball (KO) | `number_baseball_ko.py` | Easy, Medium, Hard | 400 |
| SAT Puzzle (KO) | `sat_puzzle_ko.py` | Easy, Medium, Hard | 300 |
| SAT Puzzle (EN) | `sat_puzzle_en.py` | Easy, Medium, Hard | 300 |
| Sudoku (EN) | `sudoku_en.py` | Easy, Medium, Hard | - |
| Sudoku (KO) | `sudoku_ko.py` | Easy, Medium, Hard | - |
| Yacht Dice (EN) | `yacht_dice.py` | - | 100 |
| Yacht Dice (KO) | `yacht_dice_ko.py` | - | 100 |

## 🚀 사용법

### 개별 작업 생성

```bash
# 프로젝트 루트에서 실행
cd logical-puzzles

# 기본 개수로 생성
python generation/kinship.py

# 개수 지정
python generation/kinship.py --num 200

# 다른 작업 예시
python generation/cipher_en.py --num 100
python generation/logic_grid_en.py --num-samples 300
```

### 배치 생성

```bash
# 모든 작업을 한 번에 생성
bash scripts/gen_data.sh
```

**참고**: `gen_data.sh`의 일부 작업은 주석 처리되어 있을 수 있습니다. 필요에 따라 수정하세요.

## 📁 출력 형식

생성된 데이터는 두 가지 형식으로 저장됩니다:

### 1. CSV 형식 (`data/csv/`)
- 평가를 위한 간단한 형식
- 열: `id`, `question`, `answer`, `difficulty`, `type` 등

### 2. JSONL 형식 (`data/json/`)
- 평가 시스템에서 사용
- 각 줄은 JSON 객체
- 추가 메타데이터 포함 (예: `choices`, `solution` 등)

**참고:** 난이도 레벨은 평가 시스템과의 일관성을 위해 소문자(`easy`, `medium`, `hard`)로 저장됩니다.

## 🔧 스크립트별 옵션

### 공통 옵션

대부분의 스크립트는 다음 옵션을 지원합니다:

- `--num`: 생성할 문제 수 (난이도별 또는 전체)
- `--num-samples`: 샘플 수 (일부 스크립트)

### 특수 옵션

일부 스크립트는 추가 옵션을 지원할 수 있습니다. 각 스크립트의 `--help` 옵션을 확인하세요:

```bash
python generation/kinship.py --help
python generation/cipher_en.py --help
```

## 📊 생성 통계

생성 후 각 스크립트는 다음 정보를 출력합니다:

- 생성된 문제의 총 개수
- 난이도별 분포
- 저장된 파일 경로

**출력 예시:**
```
Generated 100 questions
Difficulty breakdown:
easy      34
medium    33
hard      33

CSV file created! -> data/csv/kinship.csv
JSONL file created! -> data/json/kinship.jsonl
```

## ⚙️ 설정 및 사용자 정의

### 난이도 조정

각 스크립트는 내부적으로 난이도별 생성 비율을 설정할 수 있습니다. 스크립트 내의 난이도 설정을 수정하세요.

### 생성 개수 조정

`scripts/gen_data.sh` 파일에서 각 작업의 생성 개수를 조정할 수 있습니다:

```bash
# 예시: kinship 문제 개수 변경
python generation/kinship.py --num 200  # 100 → 200
```

## 🔍 데이터 검증

생성된 데이터는 다음 위치에서 확인할 수 있습니다:

```bash
# CSV 파일 확인
head data/csv/kinship.csv

# JSONL 파일 확인
head data/json/kinship.jsonl | python -m json.tool
```

## 📝 참고사항

1. **반복 실행**: 동일한 스크립트를 여러 번 실행하면 기존 파일을 덮어씁니다.
2. **생성 시간**: 일부 복잡한 작업은 생성에 시간이 걸릴 수 있습니다.
3. **메모리**: 대량 생성 시 메모리 사용량을 확인하세요.

## 📚 추가 정보

- 평가 시스템 사용법: [evaluation.md](evaluation.md)
- 작업별 상세 정보: [puzzles/](puzzles/)
- 프로젝트 구조: [../README.md](../README.md)

## 🔗 관련 파일

- 배치 생성 스크립트: `../scripts/gen_data.sh`
- 데이터 저장 위치: `../data/`
- 평가 데이터: `../eval_data/`