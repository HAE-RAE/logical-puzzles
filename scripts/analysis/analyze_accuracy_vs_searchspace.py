#!/usr/bin/env python3
"""
Regression / correlation analysis: gemini-3-flash accuracy vs (raw) search space.

Tests the Part 4 hypothesis with the MEASURED accuracy in accuracy_per_task.json:
  H: raw search-space size predicts accuracy CONSISTENTLY across tasks.

Inputs:
  - accuracy_per_task.json : {task_en: {easy,medium,hard: accuracy}}  (gemini-3-flash, EN)
  - raw search-space (log10) per (task, difficulty), hard-coded from the analysis doc.
    Only Type A (scaling) + Type B (constant) have a finite answer-candidate space.
    Type C (array_formula/hanoi/causal_dag/ferryman) has NO finite answer space -> excluded
    from the raw-space regression (no defined predictor).

Caveats (printed): raw_space for inequality/minesweeper/easy-blocks are order-of-magnitude
estimates; logic_grid accuracy is unreliable (non-unique ~100%, single-answer grading).
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]   # scripts/analysis/ -> project root
acc = json.load(open(ROOT / "data/accuracy/accuracy_per_task.json"))
TIERS = ["easy", "medium", "hard"]

# log10(raw answer-candidate space) per task/tier.  None = Type C (no finite space).
# Values sourced from docs/methodology/module_algorithm_search_space.md.
LOG_RAW = {
    # Type A (scaling)
    "number_baseball_en": [2.857, 4.481, 5.180],     # P(10,N): 720 / 30240 / 151200
    "sat_puzzles_en":     [2.86, 3.31, 4.21],         # 2^V: ~724 / 2048 / 16384
    "inequality_en":      [5.0, 6.6, 8.2],            # ~est: 1D size! / Latin-sq counts (approx)
    "minesweeper_en":     [12.0, 13.7, 16.2],         # ~est: C(unrevealed,mines); easy heterogen.
    "logic_grid_en":      [8.32, 11.43, 27.63],       # (N!)^(K-1)  [ACCURACY UNRELIABLE: non-unique]
    # Type B (constant across tiers)
    "cipher_en":          [0.95, 1.11, 0.95],         # keyword candidates ~9 (med up to 18)
    "yacht_dice_en":      [8.68, 8.68, 8.68],         # 12! constant
    # Type C (no finite answer space) -> excluded from raw-space regression
    "array_formula_en":   [None, None, None],
    "hanoi_en":           [None, None, None],
    "causal_dag_en":      [None, None, None],
    "ferryman_en":        [None, None, None],
}
TYPE = {
    "number_baseball_en": "A", "sat_puzzles_en": "A", "inequality_en": "A",
    "minesweeper_en": "A", "logic_grid_en": "A(unreliable)",
    "cipher_en": "B", "yacht_dice_en": "B",
    "array_formula_en": "C", "hanoi_en": "C", "causal_dag_en": "C", "ferryman_en": "C",
}

# The json holds all tasks (EN/KO/Korean-exclusive); this analysis is EN-only and only
# covers the tasks defined in LOG_RAW/TYPE above, so restrict acc to those.
acc = {k: v for k, v in acc.items() if k in LOG_RAW}


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    def rank(a):
        a = np.asarray(a, float)
        order = a.argsort()
        r = np.empty(len(a), float)
        r[order] = np.arange(len(a))
        # average ties
        for v in np.unique(a):
            m = a == v
            r[m] = r[m].mean()
        return r
    return pearson(rank(x), rank(y))


print("=" * 72)
print("1) tier별 정확도 분포 (전 task) — 보정(calibration) 확인")
print("=" * 72)
tier_means = {}
for t in TIERS:
    vals = [acc[k][t] for k in acc]
    tier_means[t] = np.mean(vals)
    print(f"  {t:7}: mean={np.mean(vals):.3f}  sd={np.std(vals):.3f}  "
          f"range=[{min(vals):.2f}, {max(vals):.2f}]  (target {{'easy':0.75,'medium':0.5,'hard':0.25}}[t])")

print()
print("=" * 72)
print("2) 분산 분해 — tier(범주) vs raw_space(연속) 설명력")
print("=" * 72)
# all 33 points
all_acc = np.array([acc[k][t] for k in acc for t in TIERS])
tot_var = np.var(all_acc)
# residual after subtracting tier mean
resid_tier = np.array([acc[k][t] - tier_means[t] for k in acc for t in TIERS])
r2_tier = 1 - np.var(resid_tier) / tot_var
print(f"  전체 정확도 분산           : {tot_var:.5f}")
print(f"  tier(easy/med/hard)만으로 R²: {r2_tier:.4f}  -> tier 가 분산의 {r2_tier*100:.1f}% 설명")
print(f"  tier 차감 후 잔차 표준편차  : {np.std(resid_tier):.4f}  (task별 차이는 이만큼만 남음)")

print()
print("=" * 72)
print("3) raw_space ↔ 정확도 상관 (Type A+B, logic_grid 제외) — 가설 H 검정")
print("=" * 72)
# within each tier, across tasks
for t in TIERS:
    xs, ys, labels = [], [], []
    for k in acc:
        lr = LOG_RAW[k][TIERS.index(t)]
        if lr is None:                      # Type C
            continue
        if k == "logic_grid_en":            # accuracy unreliable
            continue
        xs.append(lr); ys.append(acc[k][t]); labels.append(k)
    pr, sr = pearson(xs, ys), spearman(xs, ys)
    print(f"  [{t:6}] n={len(xs)}  Pearson(log_raw, acc)={pr:+.3f}  Spearman={sr:+.3f}")
    print(f"           log_raw 범위=[{min(xs):.2f}, {max(xs):.2f}] ({max(xs)-min(xs):.1f} 자릿수)  "
          f"acc 범위=[{min(ys):.2f}, {max(ys):.2f}]")

print()
print("=" * 72)
print("4) Type B 직접 반례 (raw_space 상수인데 정확도 변함)")
print("=" * 72)
for k in ["cipher_en", "yacht_dice_en"]:
    print(f"  {k:18} raw≈상수(log {LOG_RAW[k][0]:.1f})  acc easy/med/hard = "
          f"{acc[k]['easy']:.2f}/{acc[k]['medium']:.2f}/{acc[k]['hard']:.2f}  -> tier로만 변함")

print()
print("=" * 72)
print("5) 같은 정확도(easy)에 걸친 raw_space 범위 — 'raw_space가 정확도 결정 아님'")
print("=" * 72)
easy_clean = [(k, LOG_RAW[k][0], acc[k]["easy"]) for k in acc
              if LOG_RAW[k][0] is not None and k != "logic_grid_en"]
easy_clean.sort(key=lambda r: r[1])
for k, lr, a in easy_clean:
    print(f"    {k:18} log_raw={lr:5.2f}  easy_acc={a:.2f}  [type {TYPE[k]}]")
print(f"  -> easy 정확도는 {min(a for _,_,a in easy_clean):.2f}~{max(a for _,_,a in easy_clean):.2f} "
      f"(폭 {max(a for _,_,a in easy_clean)-min(a for _,_,a in easy_clean):.2f}) 인데 "
      f"raw_space 는 {max(lr for _,lr,_ in easy_clean)-min(lr for _,lr,_ in easy_clean):.1f} 자릿수 변동")

print()
print("NOTE: raw_space(inequality/minesweeper/easy-blocks)는 자릿수 추정. "
      "logic_grid 정확도는 비유일(~100%)로 신뢰 불가 → 제외. n 작음(task=11) 유의.")
