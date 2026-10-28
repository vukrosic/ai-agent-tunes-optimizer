"""Validation loss curves: default settings, best tuned fixed settings, the Sonnet-controlled run, tuned lrs with its schedule -> curves.png"""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
rows = {r["run"]: r for r in map(json.loads, open(os.path.join(HERE, "baselines.jsonl")))}
agent = [json.loads(l) for l in open(os.path.join(HERE, "run_log.jsonl"))]
lines = [("default settings (fixed)", rows["static lr 0.02"]["curve"], "#9aa0a6"),
         ("best of 19 tuned fixed settings (linear decay)", rows["static lr 0.02 adam 0.03 decay"]["curve"], "#1a73e8"),
         ("Sonnet changes the settings every 50 steps", [[r["step"], r["val"]] for r in agent], "#d93025"),
         ("tuned lrs + Sonnet's late decay", rows["static lr 0.02 adam 0.03 decay from 250"]["curve"], "#188038")]
fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
for name, c, col in lines:
    ax.plot([0] + [s for s, _ in c], [5.55] + [v for _, v in c], color=col, lw=2.5, label=f"{name}: {c[-1][1]:.3f}")
ax.set_ylim(1.8, 2.8); ax.set_xlim(40, 405); ax.set_xlabel("training step"); ax.set_ylabel("validation loss (lower is better)")
ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.25); ax.legend(frameon=False)
ax.set_title("Tiny LLM: an AI agent tuning the optimizer during training")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "curves.png"))
