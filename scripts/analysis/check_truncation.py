"""결과 CSV에서 '잘림(truncation)' 의심 태스크를 찾는다.

max_tokens 상한에 걸려 답을 내기 전에 잘리면 → 최종 답 추출 실패(빈 응답)가
길게 생성된 케이스에서 몰린다. finish_reason 컬럼이 없어 이 두 신호로 추정.

  python scripts/analysis/check_truncation.py [results_dir]
      [--model SUBSTR] [--task SUBSTR] [--min-empty 0.1]

예)
  # 기존 결과에서 잘림 의심 태스크 목록 (재실행 대상 뽑기)
  python scripts/analysis/check_truncation.py results --min-empty 0.15
  # 100k 재실행분에서 hanoi 가 안 잘렸는지 확인
  python scripts/analysis/check_truncation.py results_maxtok --task hanoi
"""
import csv, glob, os, sys, argparse

csv.field_size_limit(10**8)

def is_empty_answer(s):
    s = (s or "").strip()
    return s in ("", "[]", "['']", '[""]', "None")

def outlen(r):
    return len(r.get("thinking_content") or "") + len(r.get("resps") or "")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default="results")
    ap.add_argument("--model", default="")
    ap.add_argument("--task", default="")
    ap.add_argument("--min-empty", type=float, default=0.10,
                    help="이 비율 이상 빈 응답이면 재실행 권장 (기본 0.10)")
    ap.add_argument("--long-char", type=int, default=40000,
                    help="이 길이(문자) 이상을 '장문'으로 간주 (기본 40000)")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.results_dir, "*", "*", "*.csv")))
    rows_out = []
    for f in files:
        parts = f.replace("\\", "/").split("/")
        model, task = parts[-3], parts[-2]
        if a.model and a.model not in model:
            continue
        if a.task and a.task not in task:
            continue
        try:
            data = list(csv.DictReader(open(f, encoding="utf-8")))
        except Exception:
            continue
        if not data:
            continue
        n = len(data)
        lens = [(outlen(r), is_empty_answer(r.get("filtered_resps")),
                 int(float(r.get("exact_match", 0) or 0))) for r in data]
        empty = sum(e for _, e, _ in lens)
        # 장문인데 답 없음 = 잘림 강한 신호
        cut = sum(1 for L, e, _ in lens if e and L >= a.long_char)
        acc = sum(m for _, _, m in lens) / n
        rows_out.append({
            "model": model.replace("_", "/", 1), "task": task, "n": n,
            "acc": acc, "empty": empty, "empty_rate": empty / n,
            "cut_long": cut, "maxlen": max(L for L, _, _ in lens),
        })

    rows_out.sort(key=lambda r: (-r["empty_rate"], -r["cut_long"]))
    print(f"{'model':28} {'task':26} {'n':>3} {'acc':>5} {'empty':>6} "
          f"{'empty%':>7} {'cut(long)':>9} {'maxlen':>7}  flag")
    print("-" * 108)
    rerun = []
    for r in rows_out:
        flag = ""
        if r["empty_rate"] >= a.min_empty:
            flag = "*** 잘림의심 → 재실행"
            rerun.append(f"{r['model']}::{r['task']}")
        print(f"{r['model'][:28]:28} {r['task'][:26]:26} {r['n']:3d} "
              f"{r['acc']:5.2f} {r['empty']:6d} {r['empty_rate']*100:6.1f}% "
              f"{r['cut_long']:9d} {r['maxlen']:7d}  {flag}")

    print("\n" + "=" * 60)
    print(f"잘림 의심(빈응답 ≥ {a.min_empty*100:.0f}%): {len(rerun)} 태스크")
    # 모델별로 재실행할 task 목록을 한 줄로 (eval_maxtokens.sh 의 ONLY_TASKS 용)
    by_model = {}
    for x in rerun:
        m, t = x.split("::")
        by_model.setdefault(m, []).append(t)
    for m, ts in by_model.items():
        key = {"openai/gpt-oss-120b": "gpt-oss", "google/gemma-4-31b-it": "gemma",
               "LGAI-EXAONE/EXAONE-4.0-32B": "exaone"}.get(m, m)
        print(f'\n  ONLY_TASKS="{" ".join(sorted(set(ts)))}" bash run/eval/eval_maxtokens.sh {key}')

if __name__ == "__main__":
    main()
