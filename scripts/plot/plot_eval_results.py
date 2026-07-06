"""reports/eval_results.csv → reports/eval_results.jpg

모델별(세로 3단) × 태스크별(가로 45개) 정확도 막대그래프.
색 = 모델, 농도 = 난이도(easy>medium>hard), 빗금 = 미평가.

    python scripts/plot/plot_eval_results.py
"""
import csv
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

CSV = "reports/eval_results.csv"
OUT = "reports/eval_results.jpg"
MODELS = [("gpt-oss-120b", "#2a78d6"), ("gemma-4-31b-it", "#eb6834"),
          ("EXAONE-4.0-32B", "#4a3aa7")]
ALPHA = {"easy": 1.0, "medium": 0.62, "hard": 0.34}

# 한글 폰트 (Windows: Malgun Gothic). 다른 OS면 존재하는 한글 폰트 경로로 교체.
for fp in ["C:/Windows/Fonts/malgun.ttf",
           "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]:
    try:
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
        break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False

rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
cats = []
for r in rows:
    if r["category"] not in cats:
        cats.append(r["category"])

# 카테고리 클러스터마다 한 칸 띄운 x 위치
xpos, x, prev = [], 0, None
for r in rows:
    if prev is not None and r["category"] != prev:
        x += 1.0
    xpos.append(x); x += 1; prev = r["category"]
centers = {c: sum(xpos[i] for i, r in enumerate(rows) if r["category"] == c)
              / sum(1 for r in rows if r["category"] == c) for c in cats}

fig, axes = plt.subplots(3, 1, figsize=(13, 9.2), dpi=150, sharex=True)
fig.patch.set_facecolor("white")

def acc(r, key):
    v = r[key]
    return None if v == "" else float(v) * 100

for ax, (mname, color) in zip(axes, MODELS):
    vals = [acc(r, mname) for r in rows]
    for xi, r, v in zip(xpos, rows, vals):
        if v is None:
            ax.bar(xi, 100, width=0.9, color="#f0f1f4", hatch="////",
                   edgecolor="#d5d8de", linewidth=0)
        else:
            ax.bar(xi, v, width=0.9, color=color, alpha=ALPHA[r["difficulty"]],
                   edgecolor="none")
    ov = [v for v in vals if v is not None]
    ax.set_ylim(0, 100); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("정확도 (%)", fontsize=9, color="#535a68")
    ax.tick_params(axis="y", labelsize=8, colors="#8b93a2")
    ax.set_axisbelow(True); ax.grid(axis="y", color="#e2e5ea", linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#e2e5ea")
    for c in cats[1:]:
        left = min(xpos[i] for i, r in enumerate(rows) if r["category"] == c)
        ax.axvline(left - 1, color="#e2e5ea", linewidth=0.8, linestyle=(0, (2, 2)))
    ax.text(0.004, 0.93, mname, transform=ax.transAxes, fontsize=13,
            fontweight="bold", color=color, va="top")
    ax.text(0.004, 0.80, f"종합 {sum(ov)/len(ov):.0f}%", transform=ax.transAxes,
            fontsize=10, color="#535a68", va="top")

axes[-1].set_xticks([centers[c] for c in cats])
axes[-1].set_xticklabels(cats, fontsize=9, color="#535a68")
axes[-1].tick_params(axis="x", length=0)

leg = [Patch(facecolor="#555", alpha=ALPHA[d], label=d) for d in ("easy", "medium", "hard")]
leg.append(Patch(facecolor="#f0f1f4", hatch="////", edgecolor="#d5d8de", label="미평가"))
axes[0].legend(handles=leg, loc="upper right", fontsize=8.5, ncol=4,
               frameon=False, bbox_to_anchor=(1.0, 1.30))

fig.suptitle("추론 퍼즐 벤치마크 — 태스크별 정확도  (태스크당 100문제, reasoning on)",
             fontsize=14, fontweight="bold", color="#131720", y=0.99)
fig.text(0.5, 0.008,
         "8개 유형 × 한/영 × easy·medium·hard = 45 tasks · 색 농도 = 난이도",
         ha="center", fontsize=9, color="#8b93a2")
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
fig.savefig(OUT, format="jpg", dpi=150, facecolor="white", pil_kwargs={"quality": 92})
print("saved", OUT)
