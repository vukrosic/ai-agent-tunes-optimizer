"""Validation loss curves: default settings, best tuned fixed settings, the agent-controlled runs, tuned lrs with the agent schedule -> curves.png"""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
rows = {r["run"]: r for r in map(json.loads, open(os.path.join(HERE, "baselines.jsonl")))}
runs = {n: [[r["step"], r["val"]] for r in map(json.loads, open(os.path.join(HERE, "runs", n, "log.jsonl")))] for n in ("sonnet", "opus1", "opus2", "opus3")}
lines = [("default settings (fixed)", rows["static lr 0.02"]["curve"], "#9aa0a6"),
         ("best of 19 tuned fixed settings (linear decay)", rows["static lr 0.02 adam 0.03 decay"]["curve"], "#1a73e8"),
         ("Sonnet changes the settings every 50 steps", runs["sonnet"], "#e8710a"),
         ("Opus, run 1", runs["opus1"], "#9334e6"), ("Opus, run 2", runs["opus2"], "#a142f4"), ("Opus, run 3", runs["opus3"], "#c58af9"),
         ("tuned lrs + Sonnet's late decay", rows["static lr 0.02 adam 0.03 decay from 250"]["curve"], "#188038")]
fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
for name, c, col in lines:
    ax.plot([0] + [s for s, _ in c], [5.55] + [v for _, v in c], color=col, lw=2.2, label=f"{name}: {c[-1][1]:.3f}")
ax.set_ylim(1.8, 2.8); ax.set_xlim(40, 405); ax.set_xlabel("training step"); ax.set_ylabel("validation loss (lower is better)")
ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=9)
ax.set_title("Tiny LLM: an AI agent tuning the optimizer during training")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "curves.png"))
