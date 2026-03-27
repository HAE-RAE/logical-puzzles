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

평가 시스템은 두 가지 모델 라우터를 지원합니다:

- **liteLLM**: liteLLM 라이브러리를 통해 클라우드 API 호출 (Gemini, OpenAI, Anthropic 등)
- **remote**: OpenAI 호환 API로 자체 호스팅 서버 호출 (예: Colab vLLM)

모든 설정은 CLI 인수로 전달합니다 (설정 파일 불필요).

### liteLLM 모드 (클라우드 API)

```bash
# 기본 사용법
python evaluation/run.py \
    --model gemini/gemini-3-flash-preview \
    --model_router litellm \
    --gen-kwargs "temperature=1.0,max_tokens=65536,top_p=0.95,top_k=64" \
    --tasks kinship --async

# 난이도 필터 및 제한
python evaluation/run.py \
    --model gemini/gemini-3-flash-preview \
    --model_router litellm \
    --gen-kwargs "temperature=1.0,max_tokens=65536" \
    --tasks kinship cipher hanoi \
    --difficulty medium --limit 20 \
    --async --max-concurrent 50
```

### Remote 모드 (자체 호스팅 vLLM 등)

```bash
python evaluation/run.py \
    --model Qwen/Qwen3-0.6B \
    --model_router remote \
    --remote_url "https://xxxx.ngrok-free.app" \
    --gen-kwargs "temperature=0.6,max_tokens=16384,top_p=0.95,top_k=20,reasoning=on" \
    --tasks kinship --async --max-concurrent 30
```

### CLI 인수

| 인수 | 필수 | 설명 |
|------|------|------|
| `--model` | O | 모델 이름 (예: `gemini/gemini-3-flash-preview`, `Qwen/Qwen3-0.6B`) |
| `--model_router` | O | `litellm` 또는 `remote` |
| `--remote_url` | remote일 때 | 원격 서버 URL |
| `--gen-kwargs` | X | 생성 파라미터 (`key=value,key=value` 형식) |
| `--timeout` | X | 요청 타임아웃 (초, 기본값: 600) |
| `--tasks` | X | 평가할 작업 목록 (미지정 시 전체) |
| `--async` | X | 비동기 모드 활성화 |
| `--max-concurrent` | X | 최대 동시 요청 수 (기본값: 30) |
| `--difficulty` | X | 난이도 필터 |
| `--limit` | X | 작업당 최대 퍼즐 수 |
| `--quiet` | X | 출력 최소화 |

### 셸 스크립트 (배치 평가)

| 스크립트 | 모드 | 실행 방식 |
|---------|------|----------|
| `eval_litellm.sh` | liteLLM | 순차 실행 |
| `eval_litellm_parallel.sh` | liteLLM | 병렬 실행 (5개 동시) |
| `eval_remote.sh` | Remote | 순차 실행 |
| `eval_remote_parallel.sh` | Remote | 병렬 실행 (5개 동시) |

```bash
# liteLLM (Gemini 등)
bash scripts/eval_litellm.sh
bash scripts/eval_litellm_parallel.sh     # 권장

# Remote (Colab vLLM 등)
bash scripts/eval_remote.sh
bash scripts/eval_remote_parallel.sh      # 권장
```

병렬 스크립트는 작업별 로그를 `results/{model_name}/log/{task}.log`에 저장합니다.

### 실행 중인 평가 모니터링

```bash
bash scripts/monitor.sh              # 간단한 테이블 보기
bash scripts/monitor.sh detailed     # 상세 보기
bash scripts/monitor.sh help         # 도움말
```

### 결과 시각화

```bash
jupyter notebook scripts/viz_results.ipynb
# 또는
jupyter lab scripts/viz_results.ipynb
```

## API 키 우선순위 (liteLLM 모드)

시스템은 다음 순서로 API 키를 검색합니다:

1. `.env` 파일 (프로젝트 루트)
2. 환경 변수 (`export GEMINI_API_KEY=...`)
3. LiteLLM 기본 설정

**권장사항**: 모든 API 키를 `.env` 파일에 저장하세요.

> **참고**: Remote 모드는 `.env`에 API 키가 필요 없습니다. 서버 URL은 `--remote_url`로 전달합니다.

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
  - `summary.by_difficulty`: 난이도별 통계 (키는 소문자: `easy`, `medium`, `hard`)

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

**참고:** 난이도 값은 일관성을 위해 소문자(`easy`, `medium`, `hard`)로 저장됩니다.

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

## 생성 파라미터 (`--gen-kwargs`)

쉼표로 구분된 `key=value` 쌍으로 전달합니다:

```bash
# liteLLM (Gemini)
--gen-kwargs "temperature=1.0,max_tokens=65536,top_p=0.95,top_k=64"

# Remote (Qwen3 사고 모드)
--gen-kwargs "temperature=0.6,max_tokens=16384,top_p=0.95,top_k=20,reasoning=on"
```

특수 키:
- `reasoning=on`: 사고 모드 활성화 (remote 모드에서 `enable_thinking: true` 추가)
- 숫자 값은 자동으로 `int` 또는 `float`으로 변환

## 구조

```
evaluation/
├── core/                     # 핵심 구성 요소
│   ├── base.py               # 기본 데이터 구조
│   └── result_handler.py     # 결과 저장
├── model/                    # LLM 클라이언트 패키지
│   ├── __init__.py           # create_client() 팩토리
│   ├── base.py               # BaseLLMClient (ABC)
│   ├── litellm.py            # LiteLLMClient
│   └── remote.py             # RemoteLLMClient (OpenAI 호환)
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
│       └── ...
└── run.py                   # 메인 실행 스크립트
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

### liteLLM 모드

LiteLLM을 통해 다양한 모델을 사용할 수 있습니다:

**Google Gemini** (GEMINI_API_KEY 필요):
- `gemini/gemini-3-flash-preview` (최신, 강력함)
- `gemini/gemini-2.5-flash` (안정적, 빠름)

**OpenAI** (OPENAI_API_KEY 필요):
- `gpt-4o`
- `gpt-4o-mini`

**Anthropic** (ANTHROPIC_API_KEY 필요):
- `claude-3-5-sonnet-20241022`
- `claude-3-opus-20240229`

### Remote 모드

OpenAI 호환 API로 서빙되는 모든 모델 (예: vLLM):
- `Qwen/Qwen3-0.6B`
- `Qwen/Qwen3-1.7B`
- vLLM이 지원하는 모든 HuggingFace 모델