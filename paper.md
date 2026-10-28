# An AI agent tuning the optimizer during training: one run vs a 19-run sweep

October 28, 2026

## Abstract

An AI agent (Claude Sonnet) controlled the optimizer of a tiny language model during training. Every 50 steps it read the loss log and chose new settings. In one run it reached a validation loss of 1.900. The default settings reach 1.953, and the best of a 19-run learning-rate sweep reaches 1.878. The agent's schedule shape was better than the sweep's. It held the learning rate high, then dropped it fast at the end. Its learning rates were worse than the sweep's. Combining the sweep's learning rates with the agent's schedule gives 1.864, the best result.

## 1. Question

Reactive Optimizers (OPAL, Bai et al., MIT, 2026) let an AI agent change the optimizer while a model trains. On a 162M GPT, the agent beat the official recipe. The agent started from a learning-rate-swept recipe and gained a further 0.16 bits per sentence.

We ask a small version of that question. Starting from untuned default settings, how close does an agent get to a tuned sweep, in one run?

## 2. Setup

**Model.** A byte-level transformer:
- 3 layers, width 64, 4 heads
- RoPE, QK-norm, RMSNorm, SwiGLU, no biases, tied embeddings
- trained on TinyShakespeare
- 400 steps, batch 16, 64 tokens of context
- about 9 seconds on one CPU thread

Every run sees the same batches in the same order, so the optimizer is the only difference.

**Optimizer.** Muon (Newton-Schulz, Nesterov momentum) trains the hidden weight matrices. Adam trains the embedding table and the norm gains.

**What the agent controls.** The agent sets four values before each of the 8 segments of 50 steps:
- Muon learning rate (default 0.02)
- Muon momentum (default 0.95)
- Adam learning rate (default 0.003)
- weight decay (default 0)

After each segment, the agent sees:
- the train loss at the segment's start
- the mean train loss over the segment's last 10 steps
- the segment's highest train loss
- the validation loss

The agent cannot restart or undo a segment. It never saw the sweep results.

**Baseline.** 19 fixed-setting runs:
- Muon learning rate 0.01 to 0.16
- Adam learning rate 0.001 to 0.1
- learning rate either constant or decaying linearly to 0

## 3. Results

| run | runs | final validation loss |
|---|---|---|
| default settings | 1 | 1.953 |
| default settings, linear decay | 1 | 1.947 |
| **Sonnet controls the settings** | 1 | **1.900** |
| default settings, Sonnet-style late decay (constant until step 250, then linear to 0) | 1 | 1.907 |
| best of the sweep (Muon 0.02, Adam 0.03, linear decay) | 19 | 1.878 |
| sweep's learning rates + late decay | 1 | **1.864** |

Sonnet made these changes:
- At step 50, it raised the Adam learning rate from 0.003 to 0.004.
- At step 100, it raised the Muon learning rate from 0.02 to 0.025.
- It held those settings until step 250.
- It then cut both learning rates in three steps, to 0.005 for Muon and 0.001 for Adam by the last segment.
- It never touched momentum or weight decay.

## 4. What we learned

1. **One agent run closed 70% of the gap to a 19-run sweep.** The gap is 0.053 of 0.075.
2. **Almost all of the agent's gain came from when it lowered the learning rate.** With the default learning rates, a linear decay gains only 0.006. Holding the rate until step 250 and then dropping it gains 0.046. That second schedule is close to the warmup-stable-decay schedules used in large LLM training.
3. **The agent was too careful with the size of the learning rate.** The best Adam learning rate is 10x the default, and that is the largest single gain in the sweep. The agent moved it only from 0.003 to 0.004. By step 50 the sweep's best run is already 0.22 ahead in validation loss (2.44 vs 2.66), and the agent never caught up.
4. **The two methods find different things.** The sweep finds the size of the learning rate. The agent finds the shape of the schedule. Together they give 1.864, which beats both.

## 5. Limits

- One agent run, one model, 400 steps. A different agent run may choose differently.
- The agent made 8 decisions per run. OPAL makes about 100.
- The agent changed only four numbers. OPAL's agent can rewrite the optimizer code itself, for example switching weight matrices to Muon.
- The sweep did not include late-decay schedules. With them, the sweep would have found 1.864 itself.

## Files

- `train.py`: model, optimizer, fixed runs and agent segments
- `baselines.jsonl`: all fixed runs
- `run_log.jsonl`, `decisions.md`: the agent run and its reasons
- `plot.py`, `curves.png`: the chart
