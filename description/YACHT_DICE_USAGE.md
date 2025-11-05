# Yacht Dice 룰 변경 및 평가 가이드

## 개요

Yacht Dice는 다양한 게임 규칙을 변경할 수 있으며, 각 규칙에 맞게 프롬프트와 정답이 자동으로 조정됩니다.

## 변경 가능한 룰

### 1. 보너스 설정
- `bonus_threshold`: 보너스를 받기 위한 경계 점수 (기본값: 63)
- `bonus_points`: 보너스 점수 (기본값: 35)

### 2. 하단 항목 점수
- `full_house_points`: Full House 점수 (기본값: 25)
- `small_straight_points`: Small Straight 점수 (기본값: 30)
- `large_straight_points`: Large Straight 점수 (기본값: 40)
- `yacht_points`: Yacht 점수 (기본값: 50)

### 3. 최적화 목표
- `optimization_goal`: "maximize" 또는 "minimize" (기본값: maximize)

## 사용 예시

### 1. 문제 생성 (기본 룰)

```bash
cd guess
python yacht_dice.py
```

### 2. 커스텀 룰로 평가

```bash
cd evaluation

# 예시 1: 보너스 변경 (70점 이상 → 50점)
python eval_yacht_dice.py \
  --model gpt-4o \
  --bonus-threshold 70 \
  --bonus-points 50 \
  --recalculate

# 예시 2: 하단 항목 점수 대폭 증가
python eval_yacht_dice.py \
  --model claude-sonnet-4 \
  --full-house-points 50 \
  --yacht-points 100 \
  --recalculate

# 예시 3: 최소화 모드
python eval_yacht_dice.py \
  --model gpt-4o \
  --optimization-goal minimize \
  --recalculate

# 예시 4: 복합 커스텀 룰
python eval_yacht_dice.py \
  --model gpt-4o \
  --bonus-threshold 80 \
  --bonus-points 100 \
  --full-house-points 100 \
  --yacht-points 200 \
  --recalculate
```

## 룰 변경 시 동작

### 1. 프롬프트 자동 변경

`YachtDiceConfig`의 설정에 따라 시스템 프롬프트가 자동으로 생성됩니다:

**기본 룰 (63/35):**
```
...
(이 상단 6개 항목의 합이 63점 이상이면 35점 보너스를 받음)
...
9. Full House: 동일한 눈이 각각 3개, 2개일 경우, 고정 25점
...
```

**커스텀 룰 (70/50, Full House 100점):**
```
...
(이 상단 6개 항목의 합이 70점 이상이면 50점 보너스를 받음)
...
9. Full House: 동일한 눈이 각각 3개, 2개일 경우, 고정 100점
...
```

### 2. 정답 자동 재계산

`--recalculate` 플래그를 사용하면:
- 저장된 `dice_results`를 불러옴
- 새로운 `YachtDiceConfig`로 최적해를 다시 계산
- 변경된 정답으로 평가 수행

**예시:**
```
원래 정답 (기본 룰): 105점
새로운 정답 (Yacht 200점): 150점
```

### 3. 최적화 알고리즘

- **maximize 모드**: 헝가리안 알고리즘으로 최대 점수 찾기
- **minimize 모드**: 헝가리안 알고리즘으로 최소 점수 찾기

## 테스트 코드

```python
from yacht_dice import YachtDiceConfig, generate_random_dice, solve_yacht_dice

# 주사위 생성
dice = generate_random_dice(seed=42)

# 기본 룰
config1 = YachtDiceConfig()
score1, _ = solve_yacht_dice(dice, config1)
prompt1 = config1.get_system_prompt()
print(f"기본 점수: {score1}")

# 커스텀 룰
config2 = YachtDiceConfig(
    bonus_threshold=70,
    bonus_points=50,
    yacht_points=100
)
score2, _ = solve_yacht_dice(dice, config2)
prompt2 = config2.get_system_prompt()
print(f"커스텀 점수: {score2}")

# 프롬프트 확인
print("70점/50점 보너스:", "70점 이상이면 50점" in prompt2)
print("Yacht 100점:", "고정 100점" in prompt2 and "Yacht" in prompt2)
```

## 검증된 기능

✓ 프롬프트가 config에 따라 동적으로 생성됨
✓ 점수 계산이 config에 따라 정확하게 변경됨
✓ 최적화 목표(maximize/minimize) 정상 작동
✓ 헝가리안 알고리즘으로 최적해 보장
✓ 정답 재계산 기능 정상 작동

## 주의사항

- `--recalculate` 사용 시 JSONL 파일에 `dice_results`와 `seed`가 포함되어 있어야 함
- 데이터를 새로 생성한 경우, 해당 버전에 맞는 JSONL 파일을 사용해야 함
- 룰을 크게 변경하면 정답이 크게 달라질 수 있음
