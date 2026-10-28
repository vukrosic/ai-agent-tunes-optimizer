"""Final validation loss of every run in the 5 supervision setups -> setups.png"""
import json, os, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SETUPS = [("before", "writes the optimizer once,\nnever sees training"), ("free", "may change it any time,\ndecides when"),
          ("look", "training pauses at 100/200/300,\nmay change it"), ("live", "watches training live,\nmay edit it mid-run"),
          ("forced", "training pauses at 100/200/300,\nmust change it")]
def final(name):
    r = [json.loads(l) for l in open(os.path.join(HERE, "runs", name, "log.jsonl"))][-1]; return r.get("final_val", r.get("val"))
def changed(name):
    return len({open(os.path.join(HERE, "runs", name, "history", f)).read() for f in os.listdir(os.path.join(HERE, "runs", name, "history"))}) > 1
fig, ax = plt.subplots(figsize=(8, 5.4), dpi=150)
for x, c, lab in [(1.953, "#9aa0a6", "default settings 1.953"), (1.878, "#1a73e8", "best of 19 sweep runs 1.878"), (1.864, "#188038", "sweep's lrs + late decay 1.864")]:
    ax.axvline(x, color=c, lw=1.6, ls="--", label=lab)
for y, (k, desc) in enumerate(SETUPS):
    names = [f"{k}_{n}" for n in range(1, 6)]; v = [final(n) for n in names]; ch = [changed(n) for n in names]
    ax.scatter([a for a, c in zip(v, ch) if not c], [y] * ch.count(False), s=70, color="#9334e6", zorder=3, edgecolor="white", lw=1.5)
    ax.scatter([a for a, c in zip(v, ch) if c], [y] * ch.count(True), s=70, facecolor="white", edgecolor="#9334e6", lw=2, zorder=3)
    m = st.mean(v); ax.plot([m, m], [y - 0.28, y + 0.28], color="#202124", lw=2.5, zorder=4)
    ax.text(m + 0.0012, y - 0.2, f"average {m:.3f}", ha="left", va="center", fontsize=8.5, color="#202124")
ax.scatter([], [], s=70, color="#9334e6", label="one Opus run: optimizer set before step 0 only")
ax.scatter([], [], s=70, facecolor="white", edgecolor="#9334e6", lw=2, label="one Opus run: optimizer changed during training")
ax.set_yticks(range(len(SETUPS))); ax.set_yticklabels([d for _, d in SETUPS], fontsize=9); ax.invert_yaxis(); ax.set_ylim(len(SETUPS) - 0.4, -0.6)
ax.set_xlim(1.855, 1.96); ax.set_xlabel("final validation loss (lower is better)")
ax.spines[["top", "right", "left"]].set_visible(False); ax.tick_params(axis="y", length=0); ax.grid(axis="x", alpha=0.25)
ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(0.4, -0.13))
ax.set_title("How closely Opus supervises training barely matters (5 runs per setup)", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "setups.png"))
