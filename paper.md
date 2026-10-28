# An AI agent tuning the optimizer during training: one run vs a 19-run sweep

October 28, 2026

## Abstract

An AI agent (Claude) controlled the optimizer of a tiny language model during training. Every 50 steps it read the loss log and chose new settings. Training is deterministic: the same fixed settings give the same loss every time, so different agent runs differ only by the agent's choices.

| controller | final validation loss |
|---|---|
| default settings | 1.953 |
| Sonnet, 1 run | 1.900 |
| Claude Opus, 3 separate runs | 1.888 ± 0.002 |
| best of a 19-run learning-rate sweep | 1.878 |

All four agent runs followed the same plan:
- raise the learning rates early
- lower them toward the end
- leave momentum and weight decay alone
- keep the two learning rates in their default ratio

That ratio is the main thing the agents got wrong. The sweep's best setting uses an embedding learning rate 10x the default.

The agents' schedule shape was better than the sweep's. Combining the sweep's learning rates with Sonnet's schedule gives 1.864, the best result.

## 1. Question

Reactive Optimizers (OPAL, Bai et al., MIT, 2026) let an AI agent change the optimizer while a model trains. On a 162M GPT, the agent beat the official recipe.

We ask a small version of that question. Starting from untuned default settings, how close does an agent get to a tuned sweep in one run? And does the agent give the same result when the run is repeated?

## 2. Setup

**Model.** A byte-level transformer:
- 3 layers, width 64, 4 heads
- RoPE, QK-norm, RMSNorm, SwiGLU, no biases, tied embeddings
- trained on TinyShakespeare
- 400 steps, batch 16, 64 tokens of context
- about 9 seconds on one CPU thread

Every run starts from the same weights and sees the same batches in the same order.

**Optimizer.** Muon (Newton-Schulz, Nesterov momentum) trains the hidden weight matrices. Adam trains the embedding table and the norm gains.

**What the agent controls.** Before each of the 8 segments of 50 steps, the agent sets four values:
- Muon learning rate (default 0.02)
- Muon momentum (default 0.95)
- Adam learning rate (default 0.003)
- weight decay (default 0)

After each segment, the agent sees four numbers:
- the train loss at the start of the segment
- the mean train loss over the last 10 steps
- the highest train loss in the segment
- the validation loss

The agent cannot restart or undo a segment. It never saw the sweep results or the other agents' runs.

**Agents.** Claude Sonnet ran once. Claude Opus ran 3 times, each with the same prompt in its own folder.

**Baseline.** 19 fixed-setting runs:
- Muon learning rate from 0.01 to 0.16
- Adam learning rate from 0.001 to 0.1
- learning rate either constant or decaying linearly to 0

## 3. Results

**Training is repeatable.** The default settings, run 3 times, gave 1.9526 each time.

| run | runs | final validation loss |
|---|---|---|
| default settings | 1 | 1.953 |
| default settings, linear decay | 1 | 1.947 |
| default settings, late decay (constant until step 250, then linear to 0) | 1 | 1.907 |
| Sonnet controls the settings | 1 | 1.900 |
| Opus controls the settings, run 1 | 1 | 1.890 |
| Opus controls the settings, run 2 | 1 | 1.888 |
| Opus controls the settings, run 3 | 1 | 1.886 |
| best of the sweep (Muon 0.02, Adam 0.03, linear decay) | 19 | 1.878 |
| sweep's learning rates + late decay | 1 | **1.864** |

What each agent did, as Muon learning rate / Adam learning rate:

| run | highest rates | lowered from | last segment | Adam / Muon ratio |
|---|---|---|---|---|
| Sonnet | 0.025 / 0.004 | step 250 | 0.005 / 0.001 | 0.15–0.2 |
| Opus 1 | 0.04 / 0.006 | step 150 | 0.004 / 0.0006 | 0.15 |
| Opus 2 | 0.04 / 0.006 | step 200 | 0.006 / 0.0009 | 0.14–0.15 |
| Opus 3 | 0.05 / 0.008 | step 200 | 0.006 / 0.001 | 0.15–0.17 |
| best fixed setting | 0.02 / 0.03 | step 0 (linear) | — | 1.5 |

No agent changed momentum or weight decay.

## 4. What we learned

1. **Opus is consistent and better than Sonnet.** The three Opus runs ended within 0.004 of each other. On average they closed 87% of the gap between the default settings and the 19-run sweep. Sonnet's single run closed 70%.
2. **Agents tune the size of the learning rate but not the balance between the two.** Every agent scaled both learning rates by about the same factor, keeping the default Adam-to-Muon ratio of 0.15. The best fixed setting has a ratio of 1.5. That is the biggest single gain in the sweep, and no agent found it.
3. **The agents' timing beats a linear decay.** Holding the learning rate up and then dropping it in the last 100–150 steps is close to the warmup-stable-decay schedules used to train large LLMs. With the default learning rates, Sonnet's timing gains 0.046, while a linear decay gains only 0.006.
4. **A sweep and an agent find different things.** The sweep finds the right learning rates. The agent finds the right schedule shape. Together they give 1.864, which beats both.

## 5. Limits

- One model, 400 steps.
- The agents made 8 decisions per run. OPAL's agent makes about 100.
- Agents could change only four numbers. OPAL's agent can rewrite the optimizer code itself.
- The sweep did not include late-decay schedules. With them, it would have found 1.864 itself.
- The agent sees only losses. It might find the learning-rate ratio if it saw per-layer statistics, such as how fast the embeddings change compared with the hidden matrices.

## Files

- `train.py`: model, optimizer, fixed runs and agent segments
- `baselines.jsonl`: all fixed runs
- `runs/<name>/log.jsonl` and `runs/<name>/decisions.md`: each agent run and its reasons
- `plot.py` and `curves.png`: the chart
