# Logical-Puzzles 통합 평가 시스템

18개 논리 퍼즐 작업을 위한 통합 평가 시스템입니다.

## 지원 작업

현재 18개 작업을 지원합니다 (배치 평가에서 sudoku와 minesweeper 제외):

- `kinship`: 한국어 친족 관계 (다중 선택 A-E)
- `kinship_vision`: 이미지 기반 친족 관계 (동일한 평가자 사용)
- `cipher_en`: 영어 암호 해독
- `cipher_ko`: 한국어 암호 해독 (동일한 평가자 사용)
- `hanoi_en`: 하노이 탑 (디스크, 출발지, 목적지)
- `ferryman_en`: 뱃사공 항해 (X hours Y minutes)
- `ferryman_ko`: 뱃사공 항해 (X시간 Y분)
- `array_formula_en`: 배열 공식 계산 (영어)
- `array_formula_ko`: 배열 공식 계산 (한국어, 동일한 평가자 사용)
- `causal_dag_en`: 인과 관계 DAG 추론 (영어)
- `causal_dag_ko`: 인과 관계 DAG 추론 (한국어, 동일한 평가자 사용)
- `cryptarithmetic`: 암호 산술 퍼즐
- `inequality`: 부등식 제약 조건 만족
- `logic_grid_en`: 로직 그리드 퍼즐 (영어)
- `logic_grid_ko`: 로직 그리드 퍼즐 (한국어, 동일한 평가자 사용)
- `number_baseball`: 숫자 야구 (스트라이크/볼)
- `sat_puzzles_en`: SAT 퍼즐 풀이 (영어)
- `sat_puzzles_ko`: SAT 퍼즐 풀이 (한국어, 동일한 평가자 사용)
- `yacht_dice`: 야트 다이스 최적화

## 설치

```bash
# 필요한 패키지 설치
pip install litellm python-dotenv
```

## 환경 설정

프로젝트 루트에 `.env` 파일을 생성하고 API 키를 설정하세요:

```bash
# .env 파일 예시
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

시스템은 자동으로 `.env` 파일을 로드하여 API 키를 사용합니다.

## 사용법

### 기본 사용법

```bash
# 프로젝트 루트에서 실행
cd logical-puzzles

# 모든 작업 평가 (config.yaml 설정 사용)
python evaluation/run.py
python -m evaluation.run

# 특정 작업 평가
python evaluation/run.py --tasks kinship cipher hanoi

# 다른 모델 사용
python evaluation/run.py --model gemini/gemini-3-flash-preview
python evaluation/run.py --model gemini/gemini-2.5-flash
python evaluation/run.py --model gpt-4o
python evaluation/run.py --model claude-3-5-sonnet-20241022

# 난이도 및 제한 필터링
python evaluation/run.py --difficulty easy --limit 10
```

### 비동기 모드

비동기 모드는 `evaluation/config.yaml`에서 제어됩니다 (기본값: `use_async: true`):

```bash
# 비동기 모드 평가 (config.yaml에서 기본값)
python evaluation/run.py

# 명시적으로 비동기 모드 활성화 (config.yaml에 use_async: true가 있으면 기본값과 동일)
python evaluation/run.py --async

# 동시 실행 수 조정 (config.yaml에서 기본값: 30)
python evaluation/run.py --max-concurrent 50
```

**참고:** 현재 `--async` 플래그는 `config.yaml`이 이미 `use_async: true`를 기본값으로 설정하고 있기 때문에 효과가 없습니다. 이 플래그는 config.yaml에서 `false` 설정을 재정의하려는 경우에 유용합니다.

### 고급 옵션

```bash
python evaluation/run.py \
    --model gemini/gemini-3-flash-preview \
    --tasks kinship cipher hanoi \
    --difficulty medium \
    --limit 20 \
    --output-dir results/my_test \
    --async \
    --max-concurrent 50 \
    --quiet
```

### 셸 스크립트 (17개 작업 배치 평가)

배치 평가를 위한 두 가지 스크립트가 제공됩니다:

1. **순차 실행** (`evaluate_all.sh`):
   ```bash
   # 17개 작업을 하나씩 평가 (안정적, 느림)
   bash scripts/evaluate_all.sh
   ```
   - 작업을 순차적으로 실행 (한 번에 하나씩)
   - 더 안정적이고 디버깅이 쉬움
   - 낮은 리소스 사용
   - 더 명확한 로그 출력

2. **병렬 실행** (`evaluate_all_parallel.sh`) ⭐ **권장**:
   ```bash
   # 17개 작업을 병렬로 평가 (한 번에 5개, 빠름)
   bash scripts/evaluate_all_parallel.sh
   ```
   - 최대 5개 작업을 동시에 실행
   - 훨씬 빠름 (약 3-5배 속도 향상)
   - 높은 리소스 사용
   - 두 스크립트 모두 17개 작업을 평가합니다 (sudoku와 minesweeper 제외)
   - **병렬 처리로 인해 각 작업별로 별도의 로그 파일이 생성됩니다**
     - 로그 파일 위치: `results/log/{task_name}.log`
     - 각 작업의 시작/종료 시간, 상태(SUCCESS/FAILED)가 기록됩니다
     - 실시간으로 로그를 확인하면서 진행 상황을 모니터링할 수 있습니다

### 실행 중인 평가 모니터링

```bash
# 간단한 테이블 보기 (기본값)
bash scripts/monitor_eval.sh

# 전체 정보가 포함된 상세 보기
bash scripts/monitor_eval.sh detailed

# 도움말 표시
bash scripts/monitor_eval.sh help
```

모니터링 스크립트는 다음을 표시합니다:
- 실행 중인 평가 프로세스 (PID, 모델, 작업)
- 로그 파일의 진행 정보
- 정확도 (사용 가능한 경우)
- 로그 파일 위치

### 결과 시각화

```bash
# Jupyter 노트북으로 결과 시각화
jupyter notebook scripts/visualize_results.ipynb
# 또는
jupyter lab scripts/visualize_results.ipynb
```

시각화 노트북은 다음을 제공합니다:
- 작업별 전체 정확도
- 작업 및 난이도별 정확도 (그룹 막대 그래프)
- 작업 및 난이도별 정확도 히트맵
- 작업별 평균 지연 시간
- 정확도 vs 지연 시간 산점도

## API 키 우선순위

시스템은 다음 순서로 API 키를 검색합니다:

1. `.env` 파일 (프로젝트 루트)
2. 환경 변수 (`export GEMINI_API_KEY=...`)
3. LiteLLM 기본 설정

**권장사항**: 모든 API 키를 `.env` 파일에 저장하세요.

## 출력 결과

결과는 **모델/작업 폴더 구조**로 `results/` 디렉토리에 저장됩니다.

```
results/
├── gemini_gemini-3-flash-preview/
│   ├── kinship/
│   │   ├── gemini_gemini-3-flash-preview_kinship_2026-02-02T13-46-22__0.50.csv
│   │   └── gemini_gemini-3-flash-preview_kinship_2026-02-02T13-46-22__0.50.json
│   └── cipher/
│       ├── gemini_gemini-3-flash-preview_cipher_2026-02-02T14-00-00__0.60.csv
│       └── gemini_gemini-3-flash-preview_cipher_2026-02-02T14-00-00__0.60.json
└── gpt-4o/
    └── kinship/
        └── ...
```

**파일 형식:**
- **CSV**: 상세 결과 (`{model}_{task}_{timestamp}__{accuracy}.csv`)
  - 열: `id`, `question`, `answer`, `resps` (원본 LLM 응답), `filtered_resps` (파싱된 답변), `exact_match` (0 또는 1), `difficulty` (소문자)
  - 각 행은 하나의 퍼즐 평가를 나타냅니다
- **JSON**: 난이도별 요약 (`{model}_{task}_{timestamp}__{accuracy}.json`)
  - `summary.overall`: 전체 정확도, 정답 수, 전체 수, 평균 지연 시간 (ms)
  - `summary.by_difficulty`: 난이도별 통계 (키는 소문자: `easy`, `medium`, `hard`, `expert`)

**장점:**
- 모델별로 구성되어 비교가 쉬움
- 작업별로 분리되어 관리가 쉬움
- 상세 분석은 CSV, 요약은 JSON

### CSV 파일 구조

```csv
id,question,answer,resps,filtered_resps,exact_match,difficulty
kinship_0,"Question content...",A,"Model raw response",A,1,easy
kinship_1,"Question content...",B,"Model raw response",C,0,medium
```

**참고:** 난이도 값은 일관성을 위해 소문자(`easy`, `medium`, `hard`, `expert`)로 저장됩니다.

### JSON 파일 구조 (난이도별 요약)

```json
{
  "metadata": {
    "task": "kinship",
    "model": "gemini/gemini-3-flash-preview",
    "timestamp": "2026-02-02T13-46-22",
    "total_puzzles": 300
  },
  "summary": {
    "overall": {
      "accuracy": 0.5566666666666666,
      "correct_count": 167,
      "total_count": 300,
      "avg_latency_ms": 95893.96
    },
    "by_difficulty": {
      "easy": {
        "total": 100,
        "correct": 60,
        "accuracy": 0.60
      },
      "medium": {
        "total": 100,
        "correct": 50,
        "accuracy": 0.50
      },
      "hard": {
        "total": 100,
        "correct": 40,
        "accuracy": 0.40
      }
    }
  }
}
```

## 설정 파일 (config.yaml)

`evaluation/config.yaml`에서 기본 설정을 관리할 수 있습니다:

```yaml
llm:
  model: gemini/gemini-3-flash-preview
  temperature: 1.0
  max_tokens: 65536  # 긴 응답이 필요한 작업을 위해 증가 (예: yacht_dice)
  top_p: 0.95
  top_k: 64
  # reasoning_effort: medium  # 선택사항, 현재 비활성화됨
  timeout: 600.0     # 타임아웃 (초)

data_dir: data/json
output_dir: results

evaluation:
  use_async: true      # 비동기 모드 기본 활성화
  max_concurrent: 30    # 최대 동시 실행 수

tasks:
  - kinship
  - kinship_vision
  - cipher_en
  - cipher_ko
  - hanoi_en
  - ferryman_en
  - ferryman_ko
  - array_formula_en
  - array_formula_ko
  - causal_dag_en
  - causal_dag_ko
  - cryptarithmetic
  - inequality
  - logic_grid_en
  - logic_grid_ko
  - number_baseball
  - sat_puzzles_en
  - sat_puzzles_ko
  - yacht_dice

difficulties:
  - easy
  - medium
  - hard
```

**설정 우선순위:**
1. 명령줄 인수 (최우선)
2. `config.yaml` 설정
3. 기본값 (최하위)

## 구조

```
evaluation/
├── core/                     # 핵심 구성 요소
│   ├── base.py               # 기본 데이터 구조
│   ├── llm_client.py         # LiteLLM 래퍼 (.env 자동 로드)
│   └── result_handler.py     # 결과 저장
├── evaluators/               # 작업별 평가자
│   ├── __init__.py           # 레지스트리
│   ├── kinship.py
│   ├── cipher.py
│   ├── hanoi.py
│   ├── ferryman.py
│   ├── array_formula.py
│   ├── causal_dag.py
│   ├── cryptarithmetic.py
│   ├── inequality.py
│   ├── logic_grid.py
│   ├── number_baseball.py
│   ├── sat_puzzle.py
│   ├── yacht_dice.py
│   └── ... (더 많은 평가자)
├── legacy/                 # 레거시 평가 스크립트 (참고용)
│   ├── README.md
│   └── eval_*.py
├── eval_data/              # 정적 평가 데이터
│   ├── kinship_vision/
│   │   └── kinship.jpg
│   └── minesweeper/
│       ├── eval_metadata.jsonl
│       ├── eval_puzzles.jsonl
│       ├── eval_solutions.jsonl
│       └── solution.md
├── run.py                   # 메인 실행 스크립트 (.env 자동 로드)
├── config.yaml              # 설정 파일
└── README.md                # 이 문서
```

## 새 작업 추가

1. `evaluators/`에 새 평가자 파일 생성
2. `_parse_answer()` 및 `_check_answer()` 메서드 구현
3. `evaluators/__init__.py`의 `EVALUATOR_REGISTRY`에 등록

예제:
```python
# evaluators/my_task.py
from ..core.base import BaseEvaluator

class MyTaskEvaluator(BaseEvaluator):
    SYSTEM_PROMPT = "..."
    
    def _parse_answer(self, response: str) -> Any:
        # 응답에서 답변 파싱
        pass
        
    def _check_answer(self, expected: Any, predicted: Any) -> Tuple[bool, float]:
        # 답변이 정확한지 확인
        # 반환: (is_correct, score)
        # 참고: 모든 평가자는 이진 점수 체계를 사용합니다 (정답: 1.0, 오답: 0.0)
        # 일관성을 위해 부분 점수는 제거되었습니다
        return correct, 1.0 if correct else 0.0
```

## 지원 모델

LiteLLM을 통해 다양한 모델을 사용할 수 있습니다:

**Google Gemini** (GEMINI_API_KEY 필요, 2026년 2월 기준):
- `gemini/gemini-3-flash-preview` ⭐ (기본값, 최신, 강력함)
- `gemini/gemini-2.5-flash` (안정적, 빠름)

> **참고**: LiteLLM은 자동으로 `.env`에서 `GEMINI_API_KEY`를 사용합니다.

**OpenAI** (OPENAI_API_KEY 필요):
- `gpt-4o`
- `gpt-4o-mini`
- `gpt-4-turbo`

**Anthropic** (ANTHROPIC_API_KEY 필요):
- `claude-3-5-sonnet-20241022`
- `claude-3-opus-20240229`