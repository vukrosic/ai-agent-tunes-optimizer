"""Price the laptop and Opus, then compare: buy more training runs, or buy Opus decisions? -> cost.png and a printed table"""
import json, os
from math import comb
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OPUS = dict(input=4, cache_write=5, cache_read=0.2, output=20)   # $ per million tokens, Opus 5.5 list price
LAPTOP = 2000 / (3 * 365 * 86400)                                  # $2,000 MacBook used for 3 years, $ per second
H100 = 510 / 74 / 3600                                             # $ per second, AWS on-demand (the MIT paper's figure)
RUN_SEC = 8.4                                                      # one 400-step training run on the laptop

rows = [json.loads(l) for l in open(os.path.join(HERE, "baselines.jsonl"))]
sweep = sorted(r["val"] for r in rows[:19])                        # the 19-run learning-rate sweep
tok = {r["run"]: r for r in map(json.loads, open(os.path.join(HERE, "agent_tokens.jsonl")))}
final = lambda n: json.loads(open(os.path.join(HERE, "runs", n, "log.jsonl")).readlines()[-1])["val"]


def best_of(n):
    """Expected best loss when you train n settings picked at random from the sweep."""
    return sum(sweep[i] * comb(19 - 1 - i, n - 1) for i in range(19)) / comb(19, n)


def opus_cost(n):
    t = tok[n]; return sum(t[k] * OPUS[k] / 1e6 for k in OPUS) + t["wall_sec"] * LAPTOP


run_cost = RUN_SEC * LAPTOP
agents = [("opus_d1", 1), ("opus_d3", 3), ("opus1", 7), ("opus2", 7), ("opus3", 7)]
print(f"one training run on the laptop: ${run_cost:.5f}")
for n, d in agents: print(f"Opus, {d} decisions: ${opus_cost(n):.3f}, loss {final(n):.4f}")
mean7 = sum(final(n) for n, d in agents if d == 7) / 3
cost7 = sum(opus_cost(n) for n, d in agents if d == 7) / 3
same = next(k for k in range(1, 20) if best_of(k) <= mean7)
print(f"random sweep runs needed to match Opus with 7 decisions ({mean7:.4f}): {same}")
breakeven = cost7 / (same - 1)
print(f"Opus pays off when one training run costs more than ${breakeven:.3f} = {breakeven / LAPTOP / 3600:.1f} laptop-hours = {breakeven / H100:.0f} H100-seconds")
print(f"all 5 Opus runs at list price: ${sum(opus_cost(n) for n, _ in agents):.2f}")

fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
for price, label, col in ((run_cost, "random sweep, 9 s laptop runs", "#1a73e8"), (60 * H100, "random sweep, if each run took 1 min of H100", "#9aa0a6")):
    ax.plot([k * price for k in range(1, 20)], [best_of(k) for k in range(1, 20)], color=col, lw=2, label=label)
    for k in (1, 19): ax.annotate(f"{k} run{'s' if k > 1 else ''}", (k * price, best_of(k)), textcoords="offset points", xytext=(4, 6), fontsize=8, color="#5f6368")
for n, d in agents:
    ax.scatter(opus_cost(n), final(n), s=50, color="#9334e6", edgecolor="white", linewidth=2, zorder=3)
for d, xy in ((1, "opus_d1"), (3, "opus_d3"), (7, "opus2")):
    ax.annotate(f"Opus, {d} decision{'s' if d > 1 else ''}", (opus_cost(xy), final(xy)), textcoords="offset points", xytext=(8, -3), fontsize=8, color="#202124")
ax.scatter([], [], s=50, color="#9334e6", label="one training run, Opus changes the settings")
ax.set_xscale("log"); ax.set_xlabel("money spent ($, log scale)"); ax.set_ylabel("best validation loss (lower is better)")
ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.45, 0.99))
ax.set_title("Same loss for less money: more training runs or Opus decisions?")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "cost.png"))
