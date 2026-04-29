"""SFT/GRPO 4-way 비교 plot 생성.

Outputs (docs/figures/):
  - sft_pass_trajectory.png       SFT epoch별 pass@1 (naive vs guided + baseline)
  - sft_completion_length.png     baseline / naive / guided 응답 길이 분포 (3-way)
  - sft_by_difficulty.png         SFT 난이도별 pass@1 grouped bar
  - sft_by_problem_type.png       SFT problem_type별 pass@1 grouped bar
  - 4way_overall.png              4-way overall pass@1 bar
  - 4way_pass_trajectory.png      epoch축에 SFT trajectory + GRPO 점
  - 4way_completion_length.png    4 모델 응답 길이 violin
  - 4way_by_difficulty.png        4-way 난이도별 grouped bar
  - 4way_by_problem_type.png      4-way problem_type별 grouped bar
  - grpo_learning_curve.png       step별 correctness + mean_length
  - grpo_reward_trajectory.png    step별 correctness/format/total reward (raw + MA)
  - grpo_length_vs_reward.png     응답 길이 vs reward 산점도
"""

import json
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "docs" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# 색상 통일
C_BASE = "#7f7f7f"
C_NAIVE = "#d62728"
C_GUIDED = "#1f77b4"
C_GRPO = "#2ca02c"


def load_summary(name):
    return json.load(open(ROOT / f"results/grpo_vs_sft/{name}/summary.json"))


def load_per_puzzle(name):
    return [
        json.loads(line)
        for line in (ROOT / f"results/grpo_vs_sft/{name}/per_puzzle.jsonl").read_text().splitlines()
        if line.strip()
    ]


def grpo_log_rows():
    log = sorted((ROOT / "logs").glob("grpo_full_*.log"), key=lambda p: p.stat().st_mtime)[-1].read_text()
    rows = []
    for m in re.finditer(r"\{['\"]loss['\"][^}]+\}", log):
        s = m.group(0)
        def get(k):
            mm = re.search(rf"['\"]{k}['\"]:\s*([-\d.eE]+)", s)
            return float(mm.group(1)) if mm else None
        rows.append(dict(
            loss=get("loss"),
            reward=get("reward"),
            correctness=get("rewards/correctness_reward/mean"),
            fmt=get("rewards/format_reward/mean"),
            mean_length=get("completions/mean_length"),
            clipped=get("completions/clipped_ratio"),
            kl=get("kl"),
            step_time=get("step_time"),
        ))
    return rows


def lens(name):
    return [s["completion_len_tokens"] for r in load_per_puzzle(name) for s in r["samples"]]


# ─── SFT-only figures (legacy) ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
epochs = [0, 1, 2, 3]
naive_y = [0.450] + [load_summary(f"sft_naive_ep{e}")["overall_pass_at_1"] for e in [1, 2, 3]]
guided_y = [0.450] + [load_summary(f"sft_guided_ep{e}")["overall_pass_at_1"] for e in [1, 2, 3]]
ax.axhline(0.450, ls="--", c="gray", lw=1, label="baseline (raw)")
ax.plot(epochs, naive_y, "o-", c=C_NAIVE, lw=2, ms=7, label="SFT-naive")
ax.plot(epochs, guided_y, "s-", c=C_GUIDED, lw=2, ms=7, label="SFT-guided")
for e, y in zip(epochs, naive_y):
    ax.annotate(f"{y:.3f}", (e, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color=C_NAIVE)
for e, y in zip(epochs, guided_y):
    ax.annotate(f"{y:.3f}", (e, y), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color=C_GUIDED)
ax.set_xlabel("epoch"); ax.set_ylabel("pass@1")
ax.set_title("SFT pass@1 trajectory")
ax.set_xticks(epochs); ax.set_ylim(0, 0.55); ax.legend(); ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig(FIGDIR / "sft_pass_trajectory.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7, 4.5))
b_lens, n_lens, g_lens = lens("baseline"), lens("sft_naive_ep3"), lens("sft_guided_ep3")
parts = ax.violinplot([b_lens, n_lens, g_lens], showmeans=False, showmedians=True, widths=0.8)
for pc, c in zip(parts["bodies"], [C_BASE, C_NAIVE, C_GUIDED]):
    pc.set_facecolor(c); pc.set_alpha(0.55)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels([f"baseline\n(mean {statistics.mean(b_lens):.0f})",
                    f"SFT-naive ep3\n(mean {statistics.mean(n_lens):.0f})",
                    f"SFT-guided ep3\n(mean {statistics.mean(g_lens):.0f})"])
ax.set_ylabel("completion length (tokens)")
ax.set_title("SFT response length collapse"); ax.set_yscale("log"); ax.grid(alpha=0.25, axis="y")
plt.tight_layout(); plt.savefig(FIGDIR / "sft_completion_length.png", dpi=150); plt.close()

# SFT grouped bars (3-way) — keep legacy
for fname, key, xlabels, xname in [
    ("sft_by_difficulty", "by_difficulty", ["easy", "medium", "hard"], None),
    ("sft_by_problem_type", "by_problem_type",
     ["lookup_query", "conditional_aggregation", "array_computation", "multi_condition"],
     ["lookup", "cond_aggr", "arr_comp", "multi_cond"]),
]:
    fig, ax = plt.subplots(figsize=(8 if "problem" in fname else 7, 4.5))
    keys = xlabels
    b = [load_summary("baseline")[key][k]["pass_at_1"] for k in keys]
    n = [load_summary("sft_naive_ep3")[key][k]["pass_at_1"] for k in keys]
    g = [load_summary("sft_guided_ep3")[key][k]["pass_at_1"] for k in keys]
    x = np.arange(len(keys)); w = 0.27
    ax.bar(x - w, b, w, label="baseline", color=C_BASE)
    ax.bar(x, n, w, label="SFT-naive ep3", color=C_NAIVE)
    ax.bar(x + w, g, w, label="SFT-guided ep3", color=C_GUIDED)
    for i, (bi, ni, gi) in enumerate(zip(b, n, g)):
        ax.text(i - w, bi + 0.01, f"{bi:.2f}", ha="center", fontsize=8)
        ax.text(i, ni + 0.01, f"{ni:.2f}", ha="center", fontsize=8)
        ax.text(i + w, gi + 0.01, f"{gi:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(xname or keys); ax.set_ylabel("pass@1")
    ax.set_title(f"SFT pass@1 — {fname.split('_', 2)[-1]}"); ax.set_ylim(0, 0.85)
    ax.legend(); ax.grid(alpha=0.25, axis="y")
    plt.tight_layout(); plt.savefig(FIGDIR / f"{fname}.png", dpi=150); plt.close()

# ─── 4-way overall bar ────────────────────────────────────────────────
labels = ["baseline\n(raw)", "SFT-naive\nep3", "SFT-guided\nep3", "GRPO\nep1"]
vals = [
    load_summary("baseline")["overall_pass_at_1"],
    load_summary("sft_naive_ep3")["overall_pass_at_1"],
    load_summary("sft_guided_ep3")["overall_pass_at_1"],
    load_summary("grpo_ep1")["overall_pass_at_1"],
]
colors = [C_BASE, C_NAIVE, C_GUIDED, C_GRPO]

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.5)
ax.axhline(vals[0], ls="--", c="gray", lw=1, alpha=0.6)
for b, v in zip(bars, vals):
    diff = v - vals[0]
    sign = "+" if diff >= 0 else ""
    ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}\n({sign}{diff:+.3f})",
            ha="center", fontsize=9, color="black" if v >= vals[0] else "#a01010")
ax.set_ylabel("pass@1 (eval_60, N=4 sampling)")
ax.set_title("4-way overall pass@1 — GRPO recovers + improves; SFT collapses")
ax.set_ylim(0, max(vals) * 1.18)
ax.grid(alpha=0.25, axis="y")
plt.tight_layout(); plt.savefig(FIGDIR / "4way_overall.png", dpi=150); plt.close()
print("saved 4way_overall.png")

# ─── 4-way trajectory (epoch axis) ────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.axhline(0.450, ls="--", c="gray", lw=1, label="baseline (raw, ep0)")
ax.plot(epochs, naive_y, "o-", c=C_NAIVE, lw=2, ms=7, label="SFT-naive (3 ep)")
ax.plot(epochs, guided_y, "s-", c=C_GUIDED, lw=2, ms=7, label="SFT-guided (3 ep)")
ax.plot([1], [vals[3]], "D", c=C_GRPO, ms=10, label="GRPO (1 ep)")
ax.annotate(f"{vals[3]:.3f}", (1, vals[3]), textcoords="offset points", xytext=(10, 0), va="center", color=C_GRPO, fontsize=9, weight="bold")
ax.annotate(f"{0.450:.3f}", (0, 0.450), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color="black")
for e, y in zip(epochs[1:], naive_y[1:]):
    ax.annotate(f"{y:.3f}", (e, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color=C_NAIVE)
for e, y in zip(epochs[1:], guided_y[1:]):
    ax.annotate(f"{y:.3f}", (e, y), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color=C_GUIDED)
ax.set_xlabel("epoch")
ax.set_ylabel("pass@1")
ax.set_title("Trajectory by epoch — SFT monotone↓ vs GRPO baseline↑")
ax.set_xticks(epochs)
ax.set_ylim(0, 0.6)
ax.legend(loc="center right")
ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig(FIGDIR / "4way_pass_trajectory.png", dpi=150); plt.close()
print("saved 4way_pass_trajectory.png")

# ─── 4-way completion length ──────────────────────────────────────────
r_lens = lens("grpo_ep1")
fig, ax = plt.subplots(figsize=(8, 4.8))
data = [b_lens, n_lens, g_lens, r_lens]
parts = ax.violinplot(data, showmeans=False, showmedians=True, widths=0.85)
for pc, c in zip(parts["bodies"], [C_BASE, C_NAIVE, C_GUIDED, C_GRPO]):
    pc.set_facecolor(c); pc.set_alpha(0.6)
means = [statistics.mean(d) for d in data]
ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels([f"baseline\n(mean {means[0]:.0f})",
                    f"SFT-naive ep3\n(mean {means[1]:.0f})",
                    f"SFT-guided ep3\n(mean {means[2]:.0f})",
                    f"GRPO ep1\n(mean {means[3]:.0f})"])
ax.set_ylabel("completion length (tokens, log scale)")
ax.set_yscale("log")
ax.set_title("Response length — SFT collapses (1/6~1/20), GRPO preserves base length")
ax.grid(alpha=0.25, axis="y")
plt.tight_layout(); plt.savefig(FIGDIR / "4way_completion_length.png", dpi=150); plt.close()
print("saved 4way_completion_length.png")

# ─── 4-way grouped bar — by_difficulty ────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.8))
diffs = ["easy", "medium", "hard"]
b = [load_summary("baseline")["by_difficulty"][d]["pass_at_1"] for d in diffs]
n = [load_summary("sft_naive_ep3")["by_difficulty"][d]["pass_at_1"] for d in diffs]
g = [load_summary("sft_guided_ep3")["by_difficulty"][d]["pass_at_1"] for d in diffs]
r = [load_summary("grpo_ep1")["by_difficulty"][d]["pass_at_1"] for d in diffs]
x = np.arange(len(diffs)); w = 0.20
ax.bar(x - 1.5 * w, b, w, label="baseline", color=C_BASE)
ax.bar(x - 0.5 * w, n, w, label="SFT-naive ep3", color=C_NAIVE)
ax.bar(x + 0.5 * w, g, w, label="SFT-guided ep3", color=C_GUIDED)
ax.bar(x + 1.5 * w, r, w, label="GRPO ep1", color=C_GRPO)
for i, (bi, ni, gi, ri) in enumerate(zip(b, n, g, r)):
    for off, val, col in [(-1.5, bi, "black"), (-0.5, ni, "black"),
                          (0.5, gi, "black"), (1.5, ri, "black")]:
        ax.text(i + off * w, val + 0.015, f"{val:.2f}", ha="center", fontsize=7.5, color=col)
ax.set_xticks(x); ax.set_xticklabels(diffs); ax.set_ylabel("pass@1")
ax.set_title("4-way pass@1 by difficulty")
ax.set_ylim(0, 0.85); ax.legend(loc="upper right", ncol=2, fontsize=9); ax.grid(alpha=0.25, axis="y")
plt.tight_layout(); plt.savefig(FIGDIR / "4way_by_difficulty.png", dpi=150); plt.close()
print("saved 4way_by_difficulty.png")

# ─── 4-way grouped bar — by_problem_type ──────────────────────────────
fig, ax = plt.subplots(figsize=(9.5, 5))
ptypes = ["lookup_query", "conditional_aggregation", "array_computation", "multi_condition"]
short = ["lookup", "cond_aggr", "arr_comp", "multi_cond"]
b = [load_summary("baseline")["by_problem_type"][p]["pass_at_1"] for p in ptypes]
n = [load_summary("sft_naive_ep3")["by_problem_type"][p]["pass_at_1"] for p in ptypes]
g = [load_summary("sft_guided_ep3")["by_problem_type"][p]["pass_at_1"] for p in ptypes]
r = [load_summary("grpo_ep1")["by_problem_type"][p]["pass_at_1"] for p in ptypes]
x = np.arange(len(ptypes)); w = 0.20
ax.bar(x - 1.5 * w, b, w, label="baseline", color=C_BASE)
ax.bar(x - 0.5 * w, n, w, label="SFT-naive ep3", color=C_NAIVE)
ax.bar(x + 0.5 * w, g, w, label="SFT-guided ep3", color=C_GUIDED)
ax.bar(x + 1.5 * w, r, w, label="GRPO ep1", color=C_GRPO)
for i, (bi, ni, gi, ri) in enumerate(zip(b, n, g, r)):
    for off, val in [(-1.5, bi), (-0.5, ni), (0.5, gi), (1.5, ri)]:
        ax.text(i + off * w, val + 0.012, f"{val:.2f}", ha="center", fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(short); ax.set_ylabel("pass@1")
ax.set_title("4-way pass@1 by problem type — GRPO improves on every category")
ax.set_ylim(0, 0.78); ax.legend(loc="upper right", ncol=2, fontsize=9); ax.grid(alpha=0.25, axis="y")
plt.tight_layout(); plt.savefig(FIGDIR / "4way_by_problem_type.png", dpi=150); plt.close()
print("saved 4way_by_problem_type.png")

# ─── GRPO learning curves (full 1065 step) ────────────────────────────
rows = grpo_log_rows()
nbl = len(rows)
steps = np.array([(i + 1) * 10 for i in range(nbl)])
correct = np.array([r["correctness"] or 0 for r in rows])
mean_len = np.array([r["mean_length"] or 0 for r in rows])
fmt = np.array([r["fmt"] or 0 for r in rows])
total = np.array([r["reward"] or 0 for r in rows])


def mavg(arr, w=10):
    return np.convolve(arr, np.ones(w) / w, mode="valid") if len(arr) >= w else arr


# Reward trajectory
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(steps, total, ".", c="#ff7f0e", alpha=0.30, ms=4, label="total reward (raw)")
ax.plot(steps, correct, ".", c=C_GRPO, alpha=0.30, ms=4, label="correctness (raw)")
ax.plot(steps, fmt, ".", c="#9467bd", alpha=0.30, ms=4, label="format (raw)")
WIN = 10
ax.plot(steps[WIN - 1:], mavg(total, WIN), c="#ff7f0e", lw=2.5, label="total (10-block MA)")
ax.plot(steps[WIN - 1:], mavg(correct, WIN), c=C_GRPO, lw=2.5, label="correctness (10-block MA)")
ax.plot(steps[WIN - 1:], mavg(fmt, WIN), c="#9467bd", lw=2.5, label="format (10-block MA)")
ax.axhline(0.45, ls="--", c="black", lw=1.2, alpha=0.7, label="baseline pass@1 = 0.45")
for f, lbl in [(355, "⅓ ep"), (710, "⅔ ep"), (1065, "1 ep end")]:
    ax.axvline(f, ls=":", c="gray", lw=0.8, alpha=0.5)
    ax.annotate(lbl, (f, 0.95), fontsize=8, color="gray", ha="center")
ax.set_xlabel("training step (out of 1 065)"); ax.set_ylabel("reward")
ax.set_title("GRPO reward trajectory (1 epoch on 1 065 paired prompts, G=4)")
ax.set_ylim(-0.02, 1.02); ax.legend(loc="upper left", ncol=2, fontsize=9); ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig(FIGDIR / "grpo_reward_trajectory.png", dpi=150); plt.close()
print("saved grpo_reward_trajectory.png")

# learning curve (correctness + mean_length 이중축)
fig, axL = plt.subplots(figsize=(9, 5))
axR = axL.twinx()
axL.plot(steps, correct, ".", c=C_GRPO, alpha=0.35, ms=4)
axL.plot(steps[WIN - 1:], mavg(correct, WIN), c=C_GRPO, lw=2, label="correctness (10-block MA)")
axL.axhline(0.45, ls="--", c="black", lw=1, label="baseline pass@1 = 0.45")
axL.set_ylabel("correctness reward / mean", color=C_GRPO); axL.tick_params(axis="y", labelcolor=C_GRPO)
axL.set_ylim(0, 1.0)
axR.plot(steps, mean_len, "x", c="#9467bd", alpha=0.4, ms=5)
axR.plot(steps[WIN - 1:], mavg(mean_len, WIN), c="#9467bd", lw=2, label="mean_length (10-block MA)")
axR.set_ylabel("completion mean_length (tokens)", color="#9467bd"); axR.tick_params(axis="y", labelcolor="#9467bd")
axR.set_ylim(6000, 12500)
axL.set_xlabel("training step (out of 1 065)")
axL.set_title("GRPO learning curve — correctness ↑ then plateau, mean_length stable ~9k")
axL.grid(alpha=0.2)
h1, l1 = axL.get_legend_handles_labels(); h2, l2 = axR.get_legend_handles_labels()
axL.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=9)
plt.tight_layout(); plt.savefig(FIGDIR / "grpo_learning_curve.png", dpi=150); plt.close()
print("saved grpo_learning_curve.png")

# length vs reward 산점도
fig, ax = plt.subplots(figsize=(6.5, 5))
sc = ax.scatter(mean_len, correct, c=steps, cmap="viridis", s=45, alpha=0.85, edgecolors="w", lw=0.5)
cbar = plt.colorbar(sc, ax=ax); cbar.set_label("training step")
if len(mean_len) >= 3:
    pearson = float(np.corrcoef(mean_len, correct)[0, 1])
    ax.text(0.04, 0.96, f"Pearson r = {pearson:.3f}", transform=ax.transAxes,
            va="top", fontsize=10, bbox=dict(boxstyle="round", fc="white", alpha=0.7))
ax.axhline(0.45, ls="--", c="gray", lw=1, alpha=0.7)
ax.set_xlabel("mean_length (tokens)"); ax.set_ylabel("correctness reward (mean)")
ax.set_title("GRPO: correctness vs response length"); ax.grid(alpha=0.2)
plt.tight_layout(); plt.savefig(FIGDIR / "grpo_length_vs_reward.png", dpi=150); plt.close()
print("saved grpo_length_vs_reward.png")
print("done")
