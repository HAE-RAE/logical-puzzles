# Legacy Evaluation Scripts

## ⚠️ 주의

이 폴더의 파일들은 **참고용으로만 보관**되는 기존 평가 스크립트입니다.

## 📌 현황

- **현재 사용 중인 시스템**: `evaluation/` 폴더의 통합 평가 시스템
- **이 폴더의 용도**: 기존 구현 참고, 비교, 마이그레이션 검증

## 🔄 마이그레이션 상태

| Task | Legacy Script | 새 Evaluator | 상태 |
|------|--------------|-------------|------|
| kinship | `eval_kinship.py` | `evaluators/kinship.py` | ✅ 완료 |
| cipher | `eval_cipher.py` | `evaluators/cipher.py` | ✅ 완료 |
| hanoi | `eval_hanoi.py` | `evaluators/hanoi.py` | ✅ 완료 |
| ferryman | `eval_ferryman.py` | `evaluators/ferryman.py` | ✅ 완료 |
| array_formula | `eval_array_formula.py` | - | 🔄 예정 |
| logic_grid | `eval_logic_grid.py` | - | 🔄 예정 |
| sat_puzzles | `eval_sat_puzzle.py` | - | 🔄 예정 |
| causal_dag | `eval_causal_dag.py` | - | 🔄 예정 |
| cryptarithmetic | `eval_cryptarithmetic.py` | - | 🔄 예정 |
| inequality | `eval_inequality.py` | - | 🔄 예정 |
| number_baseball | `eval_number_baseball.py` | - | 🔄 예정 |
| minesweeper | `eval_minesweeper.py` | - | 🔄 예정 |
| sudoku | `eval_sudoku.py` | - | 🔄 예정 |
| yacht_dice | `eval_yacht_dice.py` | - | 🔄 예정 |

## 🚀 새 평가 시스템 사용하기

```bash
# 통합 평가 시스템 사용
python evaluation/run.py --tasks kinship cipher --limit 10

# 또는
python -m evaluation.run --tasks kinship cipher

# 다양한 모델 지원
python evaluation/run.py --model gemini/gemini-3-flash-preview
python evaluation/run.py --model gpt-4o
```

## 📝 주요 개선사항

통합 평가 시스템의 장점:

1. **통일된 인터페이스**: 모든 task가 동일한 방식으로 평가
2. **모델 전환 용이**: LiteLLM으로 쉬운 모델 변경
3. **확장 가능**: 새 task 추가가 간단
4. **결과 관리**: JSON 형식으로 일관된 결과 저장
5. **환경 변수 지원**: `.env` 파일로 API 키 관리

## 🔗 참고

- 새 평가 시스템 가이드: [../README.md](../README.md)
- Evaluator 추가 방법: [../evaluators/README.md](../evaluators/README.md)

---

**이 스크립트들을 직접 실행하지 마세요.** 대신 새로운 통합 평가 시스템을 사용하세요.
