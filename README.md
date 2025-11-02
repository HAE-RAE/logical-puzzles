# logical-puzzles

논리 퍼즐 평가 데이터셋 및 LLM 평가 파이프라인

## 구조

```
logical-puzzles/
├── guess/              # 문제 생성 코드
│   ├── ferryman.py     # 뱃사공 문제 생성
│   └── yacht_dice.py   # Yacht Dice 문제 생성
├── evaluation/         # 평가 코드
│   ├── eval_ferryman.py    # 뱃사공 평가
│   └── eval_yacht_dice.py  # Yacht Dice 평가
└── data/              # 생성된 데이터 (로컬에만 저장)
    ├── csv/
    └── json/
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

## 사용법

### 문제 생성

```bash
# Ferryman 문제 생성
cd guess
python ferryman.py

# Yacht Dice 문제 생성
python yacht_dice.py
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

## 주의사항

- `data/` 디렉터리는 `.gitignore`에 포함되어 있어 레포지토리에 업로드되지 않습니다
- 평가 결과는 로컬에만 저장됩니다
- 공용 레포지토리이므로 API 키나 민감한 정보를 커밋하지 마세요