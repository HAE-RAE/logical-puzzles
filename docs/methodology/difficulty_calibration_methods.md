# Task별 난이도 조절 기법 정리

`generation/` 폴더의 17개 task가 easy / medium / hard 3단계 난이도를 어떻게
구현·보정했는지 정리한 문서입니다. (논문 작성용.)

task는 **EN→KO 전이 가능성**에 따라 세 카테고리로 나뉜다 — 직번역(11) /
언어 특화(2) / 한국어 전용(4). 자세한 정의·소속 task는 §0의 카테고리 표 참조.

> `subway`, `water_jug`는 현재 작업 중이라 본 문서에서 제외한다.

## 0. 공통 보정 프레임워크

- **목표 모델**: `gemini-3-flash-preview` (reasoning=medium)

- **목표 정답률 및 허용 기준** — 각 task의 세 난이도를 아래 밴드에 맞춘다:

  | 난이도 | Target | 허용 범위 (Acceptable) |
  |--------|--------|------------------------|
  | Easy   | 75%    | 65% – 85% |
  | Medium | 50%    | 40% – 60% |
  | Hard   | 25%    | 15% – 35% |

  - **허용 오차**: 각 난이도 정답률을 Target 기준 **±10%p 이내**(위 허용 범위 안)로
    관리한다.
  - **난이도 분리(separation)**: 인접한 난이도(Easy–Medium, Medium–Hard) 간
    정답률 격차를 **최소 10%p 이상** 확보한다 — 세 난이도가 실제로 구분되도록
    하기 위함이다. (Target대로면 격차가 각 25%p이므로 충분하나, 측정값이 허용
    범위의 같은 쪽 경계로 몰리면 격차가 좁아질 수 있어 별도로 점검한다.)

- **카테고리 분류 (EN→KO 전이 가능성 스펙트럼)** — task가 한국어/문화에
  얼마나 의존하는지에 따라 셋으로 나뉘며, 보정·평가 해석의 기준이 된다:

  | 카테고리 | 정의 | task (개수) |
  |----------|------|-------------|
  | **직번역 (Language-neutral)** | 추론이 언어와 무관. KO = EN의 충실한 번역(동일 알고리즘·동일 난이도 파라미터, 차이는 프롬프트/주석 언어뿐). EN–KO 성능 격차 ≈ 0 기대. | array_formula, causal_dag, ferryman, hanoi, inequality, logic_grid, minesweeper, number_baseball, sat_puzzles, sudoku, yacht_dice **(11)** |
  | **언어 특화 (Script-adapted)** | EN/KO 둘 다 존재하나 KO가 **한글 자모(초·중·종성) 구조**에 적응 → 두 언어의 인스턴스가 번역-동치가 아님. | cipher, cryptarithmetic **(2)** |
  | **한국어 전용 (Korean-exclusive)** | EN 대응이 없음. 한국 *문화/지식*(kinship·saju·time) 또는 *문자체계*(jamo) 의존. | kinship, saju, time, jamo **(4)** |

  > **EN/KO 보정 기준**: 직번역·언어 특화는 **EN을 기준으로 보정**하고 KO는 동일
  > config를 미러링한다(언어 특화는 자모 적응분만 다름). 한국어 전용은 EN이 없으므로
  > KO 자체가 기준이다.

## 요약표

| Task | 그룹 | 핵심 난이도 레버 | easy → medium → hard |
|------|------|------------------|----------------------|
| kinship | 한국어전용 | 방해 대화 수 | noise 36 → 40 → 112 |
| saju | 한국어전용 | 문제 유형 혼합비 | 쉬운유형 위주 → 혼합 → 어려운유형 위주 |
| time | 한국어전용 | from-scratch 일진(60갑자) 비율 | ganji 0.34 → 0.60 → 0.91 |
| jamo | 한국어전용 | 음절 수 + 받침 복잡도 | 2음절·무받침 → 2음절·단받침 → 3음절·겹받침 |
| cipher | 언어특화 | 암호 적층 깊이 + 키 유도 방식 | (EN/KO 알고리즘 상이, 아래 참고) |
| cryptarithmetic | 언어특화 | 피연산자 수(자모 그룹별) | 2 → 3 → 4 operands |
| array_formula | 직번역 | 템플릿 난이도 혼합 + 테이블 규모 | 45%E+55%M → 50%M+50%H → H+방해컬럼 |
| ferryman | 직번역 | 휴식 횟수 + 시뮬레이션 규모 | rest 10–11 → 14–16 → 30 |
| hanoi | 직번역 | 원반 수 n + 질의 이동시점 k | n5–7 → n7–9 → n12–15 |
| causal_dag | 직번역 | 이벤트 수 + 간선 밀도 + AND 비율 | 25–31 → 40–46 → 46–58 events |
| logic_grid | 직번역 | 격자 크기 + 직접 앵커 수 | 5인 → 6인 → 8인 |
| sat_puzzles | 직번역 | 변수 수 + 절 수 | 9var → 11var → 14var |
| minesweeper | 직번역 | 지뢰 밀도 + 공개 셀 비율 | 혼합블록 → 9×9/14 → 9×9/18 |
| sudoku | 직번역 | givens(주어진 칸) 수 | 41 → 38 → 33 givens |
| inequality | 직번역 | 퍼즐류 전환 + 격자 크기 | 1D 사슬 → Futoshiki 5/6 → Futoshiki 6 위주 |
| number_baseball | 직번역 | 자릿수 + 공개(reveal) | 7D(set 5공개) → 8D(위치0.6) → 8D(무공개) |
| yacht_dice | 직번역 | 굴림유형 풀 + 채점 라운드 K | 고정점수풀·K2 → 혼합·K2 → normal·K6 |

---

## 1. 한국어 전용 (Korean-exclusive)

EN 대응이 없는 task. 성격이 둘로 나뉜다:
- **문화/지식 의존**: kinship(호칭 체계), saju(사주명리), time(음력·일진) — 한국
  *문화·지식*이 있어야 풀 수 있음.
- **문자체계 의존**: jamo(한글 자모 분해·합성) — 한국 *문자(Hangul) 구조* 자체에
  의존(영어로 옮기면 단순 시저 암호로 붕괴).

### kinship — 가족 호칭 추론
관계 사슬(대화 로그 속 인물 관계)을 추적해 최종 호칭을 맞히는 task.

- **고정 조건(전 난이도 공통)**: `num_choices=26`, `reverse_prob=1.0`(역방향 추론),
  `shuffle_mode="full"`, `ask_intermediate_prob=0.0`.
- **유일한 난이도 레버 — `num_noise_dialogues`(방해 대화 수)**:

  | | easy | medium | hard |
  |--|--|--|--|
  | num_noise_dialogues | 36 | 40 | 112 |

- **메커니즘**: 정답 추론과 무관한 노이즈 대화를 늘려 *정보 검색 부담·문맥 길이*를
  키움. easy/medium은 거의 동일(36 vs 40)하고 hard에서 노이즈를 약 3배로 급증시켜
  관련 단서를 찾는 난이도를 높인다.

### saju — 사주(四柱) 명리 계산
생년월일시로부터 사주 기둥(년/월/일/시주) 간지를 계산하는 결정론적 달력 task.

- **난이도 레버 — 문제 유형 혼합비(recipe mix)**. 측정된 유형별 정답률을 이용:
  - 년주 단독 ≈ **97%**
  - 일간이 주어진 시주(`m_hour_pillar_given_day`) ≈ **87%**
  - 일주/시주 원시 계산(`h_day_pillar`/`h_hour_pillar`) ≈ **24%**

  | | 구성(5~9문항 레시피) |
  |--|--|
  | easy | 시주(given-day)×4 + 일주×1 → 대부분 쉬운 유형 |
  | medium | 시주(given-day)×2 + 어려운 유형×3 → 혼합 |
  | hard | 어려운 유형(일주/시주)×8 + 쉬운 유형×1 |

- **메커니즘**: 평가기를 바꾸지 않고, **정답률이 알려진 하위 문제 유형을 비율로 섞어**
  목표 밴드를 합성. 단일 유형(일주 24%)만으로는 중간 밴드를 만들 수 없어 혼합으로 해결.

### time — 한국 달력(양력·음력) 추론
양력↔음력 변환, 상대일(오늘/내일/어제), 일수 가산, 요일, 그리고 **일진(日辰, 연속
60갑자)** 을 계산하는 결정론적 달력 task. saju와 같은 *recipe-mix* 방식으로 보정한다.

- **난이도 레버 — `ganji`(from-scratch 일진) 문제의 혼합 비율**. 일진을 처음부터
  계산하는 문제가 어렵고(만세력 지식 필요), 날짜/요일 계산은 쉽다:

  | | easy | medium | hard |
  |--|--|--|--|
  | ganji(일진) 비율 | 0.34 | 0.60 | 0.91 |
  | 나머지 유형 | date 0.66 | weekday 0.40 | date 0.09 |

- **메커니즘**: saju와 동일 — *정답률이 알려진 하위 유형*을 비율로 섞어 밴드를 합성한다.
  (측정: medium 0.60 → ≈51%, hard 0.91 → ≈24%; easy는 0.29→87 / 0.40→63 사이를 보간해
  0.34 채택, ≈75% 예상.)

### jamo — 한글 자모 합성
글자를 (초성·중성·종성) 분해 → 초성을 정해진 칸수만큼 순환 이동 → 재합성하는 task.
영어로 옮기면 단순 시저 암호로 붕괴하므로 번역 불가(진짜 언어 의존).

- **난이도 레버 2축 — 음절 수(n) + 받침(종성) 복잡도**:

  | | easy | medium | hard |
  |--|--|--|--|
  | 음절 수 n | 2 | 2 | 3 |
  | 받침 | light(거의 없음) | single(단받침) | mixed(겹받침 포함) |

- **메커니즘**: 음절이 늘수록 변환 단계 증가, **겹받침(`ㄳ ㄵ ㄺ`…)은 분해/재조합이
  까다로움**. 모델이 자모 조작에 전반적으로 약해 전 구간이 아래로 쏠리므로, 받침을
  제거해 easy를 확보(측정상 n2-single ≈ 45%, n3-mixed ≈ hard 밴드).

---

## 2. 언어 특화 (Script-adapted)

EN/KO 둘 다 존재하지만 KO 버전이 **한글 자모(초·중·종성) 구조**에 적응해 있어,
두 언어의 문제 인스턴스가 단순 번역으로 대응되지 않는다.

### cipher — 암호 해독 (EN/KO 알고리즘 상이)
EN/KO가 **서로 다른 암호 체계**를 쓰는 유일한 task.

**EN — 고전 암호(Vigenère / 전치)**

| 레버 | easy | medium | hard |
|--|--|--|--|
| cipher_stack(적층) | [vigenere] | [vigenere] | [transposition, vigenere] (2단) |
| keyword_logic(키 유도) | direct | extraction | positional |
| hint_count | 1 | 2 | 10 |
| answer_length | 20–24 | 20–24 | 6–10 |

**KO — 한글 자모 구조 암호(초성 시프트/중성 치환/종성 시프트/역순)**

| 레버 | easy | medium | hard |
|--|--|--|--|
| cipher_stack(적층) | 6단 | 12단 자모 적층 | 12단 자모 적층 |
| keyword_logic(키 유도) | positional | positional | extraction |
| hint_count | 0 | 0 | 0 |
| answer_length | 8 | 7–9 | 8–10 |

- **메커니즘**: 두 언어 모두 ① **암호 적층 깊이**, ② **키 유도 방식**
  (direct→extraction→positional, 키를 직접 주는지/추출·위치로 유도하는지),
  ③ EN은 힌트 수(1→2→10)로 보정. KO는 힌트 없이 자모 변환 단계(6→12)로 난이도를 줌.
- **핵심 차이**: EN은 알파벳 시프트/치환·전치 계열, KO는 초·중·종성으로 분해 후
  자모 단위로 변환 → **번역 시 동치가 성립하지 않는 진짜 언어 특화**.

### cryptarithmetic — 복면산
글자→숫자(0–9) 유일 대응, 첫 글자≠0, 덧셈식 제약을 만족시키는 task.

**EN/KO 표기 체계가 다르다** (그래서 언어 특화):
- **EN**: 라틴 글자 A–Z를 숫자에 대응(그룹 A–I / J–R / S–Z의 세 독립 표).
- **KO**: 한글 음절을 **초성/중성/종성으로 분해**해 **초성표·중성표·종성표를 각각
  독립적으로** 숫자에 대응(자모 그룹별 매핑). → 두 언어 인스턴스는 번역-동치가 아니다.

난이도 레버(피연산자 수)는 양쪽에 동일하게 적용되며, 보정은 EN 기준으로 수행했다.
아래 표는 EN 기준 값이다.

- **난이도 레버 — 피연산자 수(`num_operands`)**: 강한 모델에서 *지배적* 레버.

  | | easy | medium | hard |
  |--|--|--|--|
  | num_operands | 2 | 3 | 4 |
  | 자릿수 | 5 | 6 | 5 |
  | min_carries(올림 바닥) | 4 | 5 | 5 |
  | target_letters | 10 | 10 | 10 |

- **측정/교훈**: 2op ≈ 85–100% / 3op ≈ 50% / 4op ≈ 20%. **자릿수 폭·올림 수는
  거의 무영향(inert)**, 유일 글자 수는 10개(0–9)가 상한이라 더 못 밀음 → 결국
  *피연산자 개수*가 거의 유일하게 유효한 레버.

---

## 3. 직번역 (Language-neutral · EN/KO 동일 알고리즘 · 동일 파라미터)

> 아래 11개 task는 EN/KO가 같은 생성 코드·같은 난이도 config를 공유한다(차이는
> 프롬프트/주석 언어뿐). 보정은 주로 EN 기준으로 수행되었다.

### array_formula — 엑셀 배열 수식
INDEX-MATCH/SUMIF/SUMPRODUCT/SUMIFS 류 표 계산 문제.

- **레버 ① 템플릿 난이도 혼합 + ② 테이블 규모 + ③ 방해 컬럼**:

  | | easy | medium | hard |
  |--|--|--|--|
  | 템플릿 혼합 | easy 45% + medium 55% | medium 50% + hard 50% | hard 전용 + distractor 컬럼 |
  | rows | 24–32 | 55–65 | 80–85 |
  | categories/regions | 6 / 6 | 8 / 8 | 8 / 8 |
  | orders 수 | 35–50 | 120–170 | 220–300 |
  | customers 수 | 10–14 | 18–20 | 20 |

- **메커니즘**: ① 문제 유형(템플릿)의 본질 난이도, ② 데이터 규모(스캔량),
  ③ 방해 컬럼(무관 데이터)의 3축을 함께 키움.

### ferryman — 운송 스케줄 시뮬레이션
규정·휴식·혼잡을 반영해 도착 시간 등을 계산하는 다단계 시뮬레이션.

- **레버 — 휴식 횟수(`rest_count`) 중심 + 동반 물리 파라미터 스케일**:

  | | easy | medium | hard |
  |--|--|--|--|
  | rest_count | 10–11 | 14–16 | 30 |
  | distance(km) | 75–110 | 155–200 | 380–460 |
  | max_segments | 45 | 100 | 240 |
  | 혼잡 지속/감속폭 | 작음 | 중간 | 큼 |

- **메커니즘**: 시뮬레이션 이벤트(휴식·혼잡) 수를 늘려 *누적 계산 단계*를 증가시킴.
  거리·구간 수·감속폭이 함께 커져 한 문제 안의 산술 단계가 늘어난다.

### hanoi — 하노이 탑
지수적 이동 수열에서 *k번째 이동* 또는 그 시점의 상태를 묻는 문제.

- **레버 — 원반 수(n) + 질의 이동시점(k)**:

  | | easy | medium | hard |
  |--|--|--|--|
  | n (원반 수) | 5–7 | 7–9 | 12–15 |
  | k (질의 이동 index) | 25–30 | 31–37 | 1 ~ total(전 구간) |
  | 빌더 | triple-hash | triple-hash | dual-hash + simulation |

- **메커니즘**: n이 이동 수열 길이($2^n-1$)를, k가 추적 깊이를 결정.
  지수 모델($C=0.00541, a=0.1292$)로 k→정확도를 보정해 밴드를 맞췄다.

### causal_dag — 인과 DAG 시간 추론
이벤트 인과 그래프에서 발생 시각/시간차/특정 시점 이전 발생 수 등을 추론.

- **레버 — 그래프 크기·밀도·논리 게이트·질의 유형**:

  | | easy | medium | hard |
  |--|--|--|--|
  | num_events | 25–31 | 40–46 | 46–58 |
  | edge_density | 0.62 | 0.88 | 0.95 |
  | max_out_degree | 3 | 4 | 5 |
  | and_probability | 0.47 | 0.63 | 0.64 |
  | 주 질의 유형 | occurrence_time 중심 | count_by_target_time 0.75 | count_by_target_time 0.80 |
  | shuffle_edges | True | True | True |

- **메커니즘**: 크기·밀도로 전파 경로 길이를 늘리고, **AND 게이트**는 다중 선행
  조건의 동시 만족을 요구해 추론을 어렵게 함. 질의 유형이 필요한 *추론 폭*을 결정
  (단일 경로 vs 광역 전파). 간선 순서 셔플로 표면 단서도 제거.

### logic_grid — 논리 격자(Zebra류)
사람×속성 격자를 단서로 채우는 제약 만족 문제.

- **레버 — 격자 크기 + 직접 앵커(direct constraint) 수**:

  | | easy | medium | hard |
  |--|--|--|--|
  | num_people | 5 | 6 | 8 |
  | num_categories | 5 | 5 | 7 |
  | 총 단서 수 | 18–20 | 18–20 | 45–48 |
  | 직접 앵커 수 | 6 | 4 | 11 |

- **메커니즘**: 격자 크기(=탐색 공간)와 **직접 단서 비율**로 조절. medium은 easy와
  단서 총수는 같지만 직접 앵커를 6→4로 줄여 더 많은 간접 추론을 요구한다.

### sat_puzzles — SAT/CNF 충족
변수 진리값 배정으로 모든 절을 만족시키는 문제(유일 해 보장).

- **레버 — 변수 수 + 절 수**:

  | | easy | medium | hard |
  |--|--|--|--|
  | num_vars | 9 | 11 | 14 |
  | clauses | 36–54 | 56–88 | 115–170 |
  | clause_length | 3–4 | 3–4 | 3–4 |
  | negation_ratio | 0.51 | 0.54 | 0.57 |

- **메커니즘**: 변수 수가 탐색 공간($2^n$)을, 절 수가 제약 밀도를 결정.
  부정 비율은 미세 조정용.

### minesweeper — 지뢰찾기
공개된 숫자 단서로 전체 지뢰 위치를 특정(유일 해).

- **레버 — 격자 크기 + 지뢰 밀도 + 공개 셀 비율 + 해 형식**:

  | | easy | medium | hard |
  |--|--|--|--|
  | 구성 | medium/hard 블록 혼합(7×7~12×12) | 9×9 / 14 지뢰 | 9×9 / 18 지뢰 |
  | reveal 모드 | fixed | until_unique (0.25→0.45) | until_unique (0.25→0.58) |
  | solution_format | forcing | constrain | force_global |

- **메커니즘**: 지뢰 밀도↑·공개 셀↓일수록 유일 해 추론이 어려움. 공개 비율을
  *"유일 해가 될 때까지(until_unique)"* 동적으로 결정한다.
  (hard는 낮은 유일성 수락률로 생성이 느려 `--only`/`--workers` 병렬화 도입.)

### sudoku — 스도쿠
9×9 표준 스도쿠 채우기.

- **레버 — givens(주어진 숫자 칸) 수 단일 레버**:

  | | easy | medium | hard |
  |--|--|--|--|
  | target_givens | 41 | 38 | 33 |
  | spotcheck_k | 4 | 5 | 6 |
  | minimal 플래그 | off | off | off |
  | 대칭 | rot180 | rot180 | rot180 |

- **메커니즘**: givens가 적을수록 고급 추론 필요. 41 givens ≈ naked-single만으로 풀림
  (≈75%)→38→33으로 점진 하향. **minimal 퍼즐은 90%→7% '절벽'을 만들어** 비활성화하고,
  givens 수만으로 매끄러운 난이도 곡선을 확보했다.

### inequality — 대소 비교 / Futoshiki
easy와 medium/hard가 **서로 다른 퍼즐류**를 쓴다.

- **easy — 1D 부등식 사슬**: 크기 밴드(8–9 또는 16), 숨김 수(`num_to_hide` 0–1).
- **medium/hard — Futoshiki 격자**:

  | | medium | hard |
  |--|--|--|
  | 격자 크기 가중 | 5×5 62% / 6×6 38% | 5×5 20% / 6×6 80% |
  | backtrack_ratio | 0.82 | 0.92 |
  | givens (5×5 / 6×6) | 2–6 / 3–7 | 2–6 / 3–7 |

- **메커니즘**: easy는 선형 사슬(쉬움), medium/hard는 격자. **격자 크기 + "실제로
  백트래킹이 필요한 비율"**로 조절(전파 규칙만으로 풀리는지 vs 탐색 필요).

### number_baseball — 숫자야구(Bulls & Cows)
스트라이크/볼 힌트로 비밀 숫자를 추론.

- **레버 — 자릿수(num_digits) + 공개(reveal)**:

  | | easy | medium | hard |
  |--|--|--|--|
  | num_digits | 7 | 8 | 8 |
  | revealed_digits(숫자 집합 일부 공개) | 5 | 0 | 0 |
  | revealed_positions(위치 고정, 기댓값) | – | 0.6 | 0 |

- **측정/교훈**: 자릿수는 *절벽형* 레버(5D≈96 / 6D≈89 / 7D≈62 / 8D≈28%)로 너무 거칠어
  목표(50/25%)가 자릿수 간 틈에 빠짐 → **+1자리로 과도하게 어렵게 한 뒤 reveal로 완화**.
  8D에서는 *숫자 집합 공개가 무효*(병목이 위치 배열)이고 **위치 공개가 강력(+28%p/위치)**.
  위치 정수 공개는 너무 거칠어 `revealed_positions`를 **float(기대 위치 수)로 혼합**해
  medium 50% 착지(0.6 → 60% 퍼즐만 1칸 공개).

### yacht_dice — 요트 주사위
12라운드 카테고리 배정을 최적화하는 문제(부분 합 채점).

- **레버 — 굴림 유형 풀 구성 + 채점 라운드 수(K) + 불일치 밴드**:

  | | easy | medium | hard |
  |--|--|--|--|
  | roll_types 풀 | 고정점수(yacht/straight/full_house) 위주 | four_kind/full_house 확대 | normal 75% |
  | SPOTCHECK_K(채점 합산 라운드) | 2 | 2 | 6 |
  | greedy_gap 밴드 | 0–8 | 7–30 | 30+ |
  | decision_complexity | 0–3 | 3.5–7 | 8+ |

- **메커니즘**: 굴림이 *고정점수* 유형일수록 카테고리 배정이 결정적(쉬움), `normal`이
  많을수록 12!개 배정 탐색이 어려워짐. K가 클수록 채점 합산 부담↑.
  easy는 각 스팟 라운드에 *다른 라운드가 못 따라가는 고가치 카테고리*를 심어 전역 최적
  배정을 강제(건설적 생성). `greedy_gap`·`decision_complexity` 밴드로 극단 이상값 필터.

---

## 부록: task별 핵심 레버 한눈에

- **크기/규모형** (탐색 공간·데이터량을 키움): sat_puzzles(변수·절), logic_grid(인원),
  causal_dag(이벤트·밀도), array_formula(테이블), hanoi(원반 수), ferryman(시뮬 단계).
- **희소성형** (단서/정보를 줄임): sudoku(givens), minesweeper(공개 셀), inequality(givens),
  logic_grid(직접 앵커), number_baseball(reveal 역방향).
- **유형 혼합형** (난이도 알려진 하위 유형 비율 조합): saju, time(일진 비율), array_formula(템플릿), hanoi(질의 시점).
- **구조 적층형** (변환/제약 단계 적층): cipher(암호 적층), jamo(음절·받침), cryptarithmetic(피연산자).
- **노이즈형** (무관 정보 추가): kinship(방해 대화), causal_dag(간선 셔플), array_formula(방해 컬럼).
