# logical-puzzles

논리 퍼즐 평가 데이터셋 및 LLM 평가 파이프라인

## 구조

```
logical-puzzles/
├── guess/              # 문제 생성 코드
│   ├── ferryman.py     # 뱃사공 문제 생성
│   ├── yacht_dice.py   # Yacht Dice 문제 생성
│   ├── sudoku.py       # 스도쿠 문제 생성
│   └── minesweeper.py  # 지뢰찾기 문제 생성
├── evaluation/         # 평가 코드
│   ├── eval_ferryman.py    # 뱃사공 평가
│   ├── eval_yacht_dice.py  # Yacht Dice 평가
│   ├── eval_sudoku.py      # 스도쿠 평가
│   └── eval_minesweeper.py # 지뢰찾기 평가
└── data/              # 생성된 데이터 (로컬에만 저장)
    ├── csv/
    ├── json/
    ├── sudoku/
    └── minesweeper/
```

## 퍼즐 유형

### 1. Ferryman (뱃사공)
뱃사공의 물품 운송 문제로, 다양한 운항 규정과 제약 조건을 고려하여 총 소요 시간을 계산하는 문제

**특징:**
- 속도 제한 구역 (A구역, B구역)
- 화물 무게에 따른 속도 감소
- 의무 휴식 시간
- 복잡한 조건 추론 필요

### 2. Yacht Dice
12개의 주사위 결과를 12개의 카테고리에 최적으로 할당하여 총점을 최대화하는 조합 최적화 문제

**특징:**
- 12! = 479,001,600 가지의 가능한 할당
- 헝가리안 알고리즘을 사용한 최적해 계산
- 보너스 점수 계산
- 복잡한 점수 규칙 이해 필요
- **다양한 룰 변경 지원** (보너스, 점수, 최적화 목표)

### 3. Sudoku (스도쿠)
9×9 스도쿠 퍼즐 생성 및 난이도별 평가 데이터셋

**특징:**
- **유일해 보장**: 모든 퍼즐이 정확히 하나의 해를 가짐
- **난이도 평가**: Easy, Medium, Hard, Expert, Extreme 자동 분류
- **Spot-check 평가**: HMAC 기반 K-셀 선택으로 LLM 평가 지원
- **대칭 지원**: 회전/반사 대칭으로 미적 품질 향상
- **재현 가능**: 시드 고정으로 동일한 퍼즐 재생성

### 4. Minesweeper (지뢰찾기)
제약 충족 문제(CSP)로 설계된 지뢰찾기 퍼즐 - LLM의 논리적 추론 능력 평가

**특징:**
- **유일해 보장**: 백트래킹 솔버로 유일해 검증
- **최소 힌트**: 유일해를 유지하면서 힌트 최소화
- **다양한 난이도**: Easy (6×6), Medium (8×8), Hard (10×10)
- **좌표 기반 평가**: 지뢰 위치를 (r,c) 형식으로 출력
- **부분 점수**: Exact Match, Precision, Recall, F1 Score

## 사용법

### 문제 생성

```bash
# Ferryman 문제 생성
cd guess
python ferryman.py

# Yacht Dice 문제 생성
python yacht_dice.py

# Sudoku 문제 생성 (5개 난이도별 데이터셋)
python sudoku.py

# Minesweeper 문제 생성 (난이도별 데이터셋)
python minesweeper.py
```

### 평가 실행

```bash
# Ferryman 평가
cd evaluation
python eval_ferryman.py

# Yacht Dice 평가 (기본 룰)
python eval_yacht_dice.py --model gpt-4o

# Yacht Dice 평가 (커스텀 룰)
python eval_yacht_dice.py \
  --model gpt-4o \
  --bonus-threshold 70 \
  --bonus-points 50 \
  --yacht-points 100 \
  --recalculate

# Sudoku 평가
python eval_sudoku.py --model gpt-4o
python eval_sudoku.py --model gpt-4o-mini
python eval_sudoku.py --model o1

# Minesweeper 평가
python eval_minesweeper.py --model gpt-4o
python eval_minesweeper.py --model o1
```

### Yacht Dice 룰 변경

Yacht Dice는 다양한 게임 규칙을 변경할 수 있습니다:

```python
from yacht_dice import YachtDiceConfig

# 기본 룰
config1 = YachtDiceConfig()

# 보너스 변경
config2 = YachtDiceConfig(bonus_threshold=70, bonus_points=50)

# 하단 항목 점수 변경
config3 = YachtDiceConfig(full_house_points=50, yacht_points=100)

# 최적화 목표 변경
config4 = YachtDiceConfig(optimization_goal="minimize")
```

자세한 내용은 [YACHT_DICE_USAGE.md](YACHT_DICE_USAGE.md) 참고

### Sudoku 사용 예시

```python
from sudoku import generate_puzzle, generate_5diff_dataset

# 단일 퍼즐 생성
puzzle = generate_puzzle(difficulty='Hard', symmetry='rot180', seed=42, k=6)

# 5개 난이도별 데이터셋 생성
puzzles = generate_5diff_dataset(k=6, seed=2025)
```

**생성 옵션:**
- `difficulty`: 'Easy', 'Medium', 'Hard', 'Expert', 'Extreme', 'Any'
- `symmetry`: 'none', 'rot180' (180도 회전 대칭)
- `k`: spot-check 좌표 개수 (기본값: 6)
- `seed`: 랜덤 시드 (재현성)

**평가 데이터 형식:**
```json
{
  "id": "s9_002000",
  "puzzle": "39.........8..15.6...",
  "solution": "391654782278391546...",
  "difficulty": {"label": "Easy", "search_nodes": 0},
  "spotcheck": {
    "k": 6,
    "positions": ["r2c6", "r4c9", "r8c9", "r5c9", "r7c6", "r4c6"],
    "code": "191796"
  }
}
```

### Minesweeper 사용 예시

```python
from minesweeper import generate_puzzle, generate_dataset

# 단일 퍼즐 생성
puzzle = generate_puzzle(rows=6, cols=6, num_mines=8, difficulty='easy', seed=42)

# 난이도별 데이터셋 생성
puzzles = generate_dataset(num_puzzles_per_level=5, seed=2025)
```

**생성 옵션:**
- `rows, cols`: 그리드 크기 (6×6, 8×8, 10×10 등)
- `num_mines`: 지뢰 개수 (총 셀의 12-20% 권장)
- `difficulty`: 'easy', 'medium', 'hard'
- `seed`: 랜덤 시드 (재현성)

**평가 데이터 형식:**
```json
{
  "id": "easy_6x6_2025",
  "rows": 6,
  "cols": 6,
  "mines": 8,
  "difficulty": "easy",
  "puzzle": ["2#3##1", "2##442", "1#####", "##2#3#", "######", "#0##0#"],
  "solution": ["010110", "010000", "001101", "000001", "000000", "000000"]
}
```

**프롬프트 형식:**
```
You are solving a Minesweeper puzzle. The grid is 6x6 with 8 mines total.

Grid notation:
- '#' represents a hidden cell
- Numbers (0-8) show count of adjacent mines

Puzzle grid:
2#3##1
2##442
1#####
##2#3#
######
#0##0#

Output format: (r,c) (r,c) ... (0-based indexing)
Final answer: (0,1) (0,3) (1,2) ...
```

## 주의사항

- `data/` 디렉터리는 `.gitignore`에 포함되어 있어 레포지토리에 업로드되지 않습니다
- 평가 결과는 로컬에만 저장됩니다
- 공용 레포지토리이므로 API 키나 민감한 정보를 커밋하지 마세요