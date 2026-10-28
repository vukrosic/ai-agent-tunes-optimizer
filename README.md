# Let an AI agent tune the optimizer while your LLM trains

## Abstract

**An AI agent (Claude) watched a language model train and changed the optimizer settings every 50 training steps.** It could not restart training or undo a change.

**In one run, the agent got close to the best of 19 hand-tuned runs.** Three repeated Claude Opus runs ended almost the same.

**The agent found a better learning-rate schedule than the sweep, but missed the best learning rates.** Putting the two together gave the best model.

This is a small version of [Reactive Optimizers (OPAL)](https://xingjianbai.com/reactive-optimizers/), the MIT paper where an agent rewrites the optimizer during training.

> [!IMPORTANT]
> Validation loss (how wrong the model's next-letter guesses are, lower is better): default settings **1.953**, Claude Opus **1.890 / 1.888 / 1.886** in 3 runs, best of 19 tuned runs **1.878**, tuned learning rates + the agent's schedule **1.864**.

![curves](curves.png)

---

## Training is repeatable, so differences come from the agent

Every run starts from the same weights and sees the same data in the same order. The default settings, run 3 times, gave **1.9526** every time.

> **Any difference between agent runs comes only from the agent's choices.**

## Opus gets close to a 19-run sweep in one run, every time

| who sets the optimizer | runs needed | final validation loss |
|---|---|---|
| nobody (default settings) | 1 | 1.953 |
| Claude Sonnet | 1 | 1.900 |
| Claude Opus, 3 separate runs | 1 each | 1.890, 1.888, 1.886 |
| best of 19 fixed settings (learning-rate sweep) | 19 | 1.878 |
| **sweep's learning rates + the agent's schedule** | 1 | **1.864** |

The 3 Opus runs ended within 0.004 of each other and closed **87%** of the gap between the default and the sweep. Sonnet closed 70%.

## Keep the learning rate high, then drop it fast at the end

Every agent run did the same thing: raise both learning rates early, hold them, then cut them over the last 150–200 steps. That is close to the warmup-stable-decay schedule used to train large LLMs.

With the default learning rates, this late drop improves the loss by **0.046**. A straight-line decay from the start improves it by only 0.006.

## The agent missed the biggest win: the embedding learning rate

The model has two learning rates: one for the hidden weight matrices (trained with **Muon**) and one for the **embedding table**, which turns each letter into a vector (trained with **Adam**).

The best setting needs the embedding learning rate **10x** the default. Every agent moved both learning rates together, so the embedding one only went up about 2x.

> **The sweep found better learning rates and the agents found a better schedule. Using both gave 1.864.**

---

## Where it stops working

> [!WARNING]
> One small model, 400 steps, 8 decisions per run. The agent could change only 4 numbers (two learning rates, momentum, weight decay); OPAL's agent rewrites the optimizer code and makes about 100 decisions. The agent saw only losses; it might find the learning-rate balance if it saw per-layer statistics.

## Setup

Byte-level transformer on TinyShakespeare: 3 layers, width 64, RoPE, QK-norm, RMSNorm, SwiGLU, no biases. Muon on hidden matrices, Adam on the embedding and norm gains. 400 steps, batch 16, 64 tokens of context, about 9 seconds per run on one CPU thread.

Short paper with all numbers: [paper.md](paper.md). Each agent's reasons for its changes: `runs/<name>/decisions.md`.

```bash
python3 train.py static --lr 0.02 --adam 0.03 --decay --decay_from 250   # one fixed run
python3 train.py reset --run NAME                                         # start an agent run in runs/NAME/
python3 train.py segment --run NAME                                       # train 50 steps with runs/NAME/config.json; the agent edits it between segments
python3 plot.py
```
