"""Tiny helper for SFT qualitative comparison stats. Not part of the pipeline."""
import json
import statistics
from collections import defaultdict


def load(d):
    return [json.loads(l) for l in open(f"results/grpo_vs_sft/{d}/per_puzzle.jsonl")]


base = load("baseline")
n3 = load("sft_naive_ep3")
g3 = load("sft_guided_ep3")

by_b = {p["puzzle_id"]: p for p in base}
by_n = {p["puzzle_id"]: p for p in n3}
by_g = {p["puzzle_id"]: p for p in g3}

diff_stat = defaultdict(lambda: {"base": [], "naive": [], "guided": []})
ptype_stat = defaultdict(lambda: {"base": [], "naive": [], "guided": []})
for pid in by_b:
    d = by_b[pid]["difficulty"]
    t = by_b[pid]["problem_type"]
    diff_stat[d]["base"].append(by_b[pid]["pass_at_1"])
    diff_stat[d]["naive"].append(by_n[pid]["pass_at_1"])
    diff_stat[d]["guided"].append(by_g[pid]["pass_at_1"])
    ptype_stat[t]["base"].append(by_b[pid]["pass_at_1"])
    ptype_stat[t]["naive"].append(by_n[pid]["pass_at_1"])
    ptype_stat[t]["guided"].append(by_g[pid]["pass_at_1"])

print("--- difficulty pass@1 ---")
for d in ["easy", "medium", "hard"]:
    s = diff_stat[d]
    avg = lambda v: sum(v) / len(v) if v else 0
    print(f"{d:>6}: base={avg(s['base']):.3f}  naive={avg(s['naive']):.3f}  guided={avg(s['guided']):.3f}")

print()
print("--- problem_type pass@1 ---")
for t in ["lookup_query", "conditional_aggregation", "array_computation", "multi_condition"]:
    s = ptype_stat[t]
    avg = lambda v: sum(v) / len(v) if v else 0
    print(f"{t:>25}: base={avg(s['base']):.3f}  naive={avg(s['naive']):.3f}  guided={avg(s['guided']):.3f}")


def len_stat(rows):
    L = [s["completion_len_tokens"] for r in rows for s in r["samples"]]
    return statistics.mean(L), statistics.median(L), max(L)


print()
print("--- completion length (mean / median / max) ---")
for tag, rows in [("base", base), ("naive", n3), ("guided", g3)]:
    m, md, mx = len_stat(rows)
    print(f"{tag:>6}: mean={m:.0f}  median={md:.0f}  max={mx}")

print()
print("--- 정성 비교용 puzzle (base 정답=1.0 / sft 모두 오답=0.0) ---")
selected = {}
for d in ["easy", "medium", "hard"]:
    cands = [
        p for p in base
        if p["difficulty"] == d
        and p["pass_at_1"] == 1.0
        and by_n[p["puzzle_id"]]["pass_at_1"] == 0.0
        and by_g[p["puzzle_id"]]["pass_at_1"] == 0.0
    ]
    if cands:
        selected[d] = cands[0]
        print(f"{d}: {cands[0]['puzzle_id']}  expected={cands[0]['expected']}")

# Save selected puzzle response samples to a file for the md
out = {"diff_stat": {d: {k: sum(v) / len(v) for k, v in diff_stat[d].items()} for d in diff_stat},
       "ptype_stat": {t: {k: sum(v) / len(v) for k, v in ptype_stat[t].items()} for t in ptype_stat},
       "len_stat": {tag: dict(zip(["mean", "median", "max"], len_stat(r))) for tag, r in [("base", base), ("naive", n3), ("guided", g3)]},
       "selected": {}}
for d, c in selected.items():
    pid = c["puzzle_id"]
    out["selected"][d] = {
        "puzzle_id": pid,
        "difficulty": d,
        "problem_type": c["problem_type"],
        "expected": c["expected"],
        "base_response": c["samples"][0]["raw_response"],
        "naive_response": by_n[pid]["samples"][0]["raw_response"],
        "guided_response": by_g[pid]["samples"][0]["raw_response"],
        "base_predicted": [s["predicted"] for s in c["samples"]],
        "naive_predicted": [s["predicted"] for s in by_n[pid]["samples"]],
        "guided_predicted": [s["predicted"] for s in by_g[pid]["samples"]],
    }
import json as _j
open("results/grpo_vs_sft/qualitative_selected.json", "w").write(_j.dumps(out, indent=2, ensure_ascii=False))
print("saved → results/grpo_vs_sft/qualitative_selected.json")
