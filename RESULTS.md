# Results: an AI agent tuning the optimizer while a small LLM trains

October 28, 2026. All numbers are the validation loss after step 400 (lower is better).

## Setup

**Model.** A byte-level transformer:
- 3 layers, width 64, 4 heads
- RoPE, QK-norm, RMSNorm, SwiGLU, no biases, tied embeddings
- trained on TinyShakespeare
- 400 steps, batch 16, 64 tokens of context
- about 9 seconds on one CPU thread

Every run starts from the same weights and sees the same batches in the same order. The default settings, run 3 times, gave 1.9526 each time. Any difference between agent runs comes only from the agent's choices.

**Optimizer.** Muon (Newton-Schulz, Nesterov momentum) trains the hidden weight matrices. Adam trains the embedding table and the norm gains. Defaults: Muon learning rate 0.02, momentum 0.95, Adam learning rate 0.003, no schedule.

**Agent.** Claude Opus 5.5 in Claude Code, one run at a time, each in its own folder.

**Sweep baseline.** 19 fixed-setting runs:
- Muon learning rate from 0.01 to 0.16
- Adam learning rate from 0.001 to 0.1
- learning rate either constant or decaying linearly to 0

---

## Part 1: the agent sets 4 numbers every 50 steps

Training ran in 8 segments of 50 steps. The first segment used the default settings. Before each of the other 7 segments, the agent set the Muon learning rate, Muon momentum, Adam learning rate and weight decay. After each segment it saw the train loss at the start of the segment, the mean train loss over its last 10 steps, the highest train loss in it, and the validation loss.

| run | runs | final validation loss |
|---|---|---|
| default settings | 1 | 1.953 |
| default settings, linear decay | 1 | 1.947 |
| default settings, late decay (constant until step 250, then linear to 0) | 1 | 1.907 |
| Opus controls the settings, 1 decision (at step 200) | 1 | 1.948 |
| Opus controls the settings, 3 decisions | 1 | 1.894 |
| Opus controls the settings, 7 decisions, run 1 | 1 | 1.890 |
| Opus controls the settings, 7 decisions, run 2 | 1 | 1.888 |
| Opus controls the settings, 7 decisions, run 3 | 1 | 1.886 |
| best of the sweep (Muon 0.02, Adam 0.03, linear decay) | 19 | 1.878 |
| sweep's learning rates + late decay from step 250 | 1 | **1.864** |
| sweep's learning rates + late decay from step 200 (Opus's timing) | 1 | **1.864** |

What each 7-decision Opus run did, as Muon learning rate / Adam learning rate:

| run | highest rates | lowered from | last segment | Adam / Muon ratio |
|---|---|---|---|---|
| Opus 1 | 0.04 / 0.006 | step 150 | 0.004 / 0.0006 | 0.15 |
| Opus 2 | 0.04 / 0.006 | step 200 | 0.006 / 0.0009 | 0.14–0.15 |
| Opus 3 | 0.05 / 0.008 | step 200 | 0.006 / 0.001 | 0.15–0.17 |
| best fixed setting | 0.02 / 0.03 | step 0 (linear) | — | 1.5 |

No agent changed momentum or weight decay.

1. **Opus is consistent.** The three 7-decision runs ended within 0.004 of each other and closed 87% of the gap between the default settings and the 19-run sweep.
2. **Agents tune the size of the learning rates but not the balance between them.** Every agent scaled both learning rates by about the same factor, keeping the default Adam-to-Muon ratio of 0.15. The best fixed setting has a ratio of 1.5.
3. **The agents' timing beats a linear decay.** Holding the learning rate up and then dropping it in the last 150–200 steps is close to the warmup-stable-decay schedules used to train large LLMs. With the default learning rates, a late decay from step 250 gains 0.046, while a linear decay gains only 0.006.
4. **The sweep and the agents found different things.** The sweep found better learning rates and the agents found a better schedule. Using both gives 1.864, lower than either alone.

Files: `train.py` (model, optimizer, fixed runs, agent segments), `baselines.jsonl` (all fixed runs), `runs/opus*/log.jsonl` and `runs/opus*/decisions.md` (each run and its reasons), `plot.py` → `curves.png`.

---

## Part 2: how closely the agent supervises training

Here the agent gets the whole optimizer code (`setups/optimizer.py`) and may rewrite anything in it. Five setups differ only in when the agent can change it. Exact prompts: [setups/PROMPTS.md](setups/PROMPTS.md). Training scripts: `setups/<setup>/train.py`.

| setup | how it works |
|---|---|
| **before**: writes once | The agent edits `optimizer.py`, then runs `train.py` once. It trains all 400 steps in one go and prints only the final loss. |
| **free**: decides when | `train.py --steps N` trains N more steps from the checkpoint and prints the loss. The agent picks every N and may edit `optimizer.py` between calls. |
| **look**: forced to look | Like free, but training also pauses at steps 100, 200 and 300. The next call continues whether or not the code changed. |
| **live**: watches live | `train.py` runs all 400 steps without pausing, slowed to about 0.25 s per step (2 minutes). It logs the train loss every step and the validation loss every 25 steps. Before every step it checks `optimizer.py` and loads a new version, keeping the optimizer state. |
| **forced**: forced to change | Like look, but at steps 100, 200 and 300 `train.py` refuses to continue until `optimizer.py` has changed. |

### All 25 runs

| setup | run 1 | run 2 | run 3 | run 4 | run 5 | average | spread (sd) |
|---|---|---|---|---|---|---|---|
| before | 1.8765 | 1.8817 | 1.8827 | 1.8903 | 1.9037 | 1.8870 | 0.0106 |
| free | 1.8900 | 1.8876 | 1.8732 | 1.8969 | 1.8898 | 1.8875 | 0.0087 |
| look | 1.8923 | 1.8827 | 1.8827 | 1.8857 | 1.8815 | **1.8850** | 0.0044 |
| live | 1.8907 | 1.8764 | 1.8831 | 1.8958 | 1.8824 | 1.8857 | 0.0076 |
| forced | 1.9052 | 1.9012 | 1.8757 | 1.8952 | 1.9017 | 1.8958 | 0.0118 |

Each number was checked by reloading the run's final model with an untouched `fixed.py` and recomputing the validation loss (`setups/check.py`).

**Are the setups different?** A permutation test that shuffles the 25 numbers between setups gives p = 0.35 for "some setup differs from the rest". Forced against the other 20 runs pooled gives p = 0.02 (one-sided). That second comparison was chosen after seeing that forced looked worst, so it is a lead, not a result.

### When the agent changed the code

Steps at which a new `optimizer.py` took over, from the saved copy of every version in `runs/<run>/history/`:

| setup | run 1 | run 2 | run 3 | run 4 | run 5 |
|---|---|---|---|---|---|
| before | 0 | 0 | 0 | 0 | 0 |
| free | 0 | 0 | 0 | 0 | 0 |
| look | 0 | 0 | 0 | 0 | 0 |
| live | 0, 53, 95 | 0 | 0, 134 | 0 | 0, 211 |
| forced | 0, 100, 200, 300 | 0, 100, 200, 300 | 0, 100, 200, 300 | 0, 100, 200, 300 | 0, 100, 200, 300 |

**The plan at step 0 was the same in almost every run:** raise the Muon learning rate from 0.02 to 0.03–0.04, raise the Adam learning rate from 0.003 to 0.006–0.008, add a short warmup, hold, then decay linearly to 0 (or near 0) over the last 140–240 steps. Two runs also warmed momentum up from 0.85 to 0.95.

**Changes during training:**
- live 1: at steps 40, 80 and 120 it scaled the next update by 0.5, 1, 2 and 4, measured the loss on two spare training batches, and undid it (`runs/live_1/probe.jsonl`). Doubling the Adam update helped, so it raised the Adam learning rate to 0.012 at step 53 and 0.016 at step 95. Final 1.8907.
- live 3: at step 134 raised both learning rates again (Muon 0.04, Adam 0.01) because the loss fell smoothly. Final 1.8831.
- live 5: at step 211 started the decay earlier (step 215 instead of 260) because the validation loss was improving slowly. Final 1.8824.
- forced runs: at the three stops they raised the learning rates again, moved the start of the decay earlier, changed the decay shape (for example to 1 − √progress), or lowered momentum for the last 100 steps.

### What we learned

1. **Supervision does not help on this run.** Writing the optimizer once (1.887) is as good as deciding when to step in (1.887), being made to look (1.885) or watching live (1.886).
2. **Forcing changes hurts a little.** Forced averages 1.896, about 0.01 worse than the rest. Most forced changes were made because the script required them, not because the loss showed a problem.
3. **The agent rarely steps in by itself.** 20 of 25 runs never changed the code after step 0. In the look setup it was shown the loss three times per run and changed nothing in all 5 runs.
4. **The agent still misses the embedding learning rate.** Most runs raised it 2–3×, against the 10× the sweep found. The one run that measured it (live 1) moved it to 5× and still ended at 1.891.

---

## Part 3: cost

Prices:
- **Laptop:** a $2,000 MacBook used for 3 years, so one 9-second training run costs $0.00018.
- **Opus 5.5 at API list price:** $4 per million input tokens, $5 per million cache-write tokens, $0.20 per million cache-read tokens, $20 per million output tokens. Token counts are in `agent_tokens.jsonl`.

| what you buy | cost | final validation loss |
|---|---|---|
| 1 training run, default settings | $0.0002 | 1.953 |
| 7 training runs, settings picked at random from the sweep | $0.0013 | 1.886 (average) |
| 19 training runs (the whole sweep) | $0.0034 | 1.878 |
| 1 training run, Opus makes 1 decision (at step 200) | $0.42 | 1.948 |
| 1 training run, Opus makes 3 decisions | $0.33 | 1.894 |
| 1 training run, Opus makes 7 decisions (3 runs) | $0.45 (average) | 1.888 (average) |

One Opus run reaches the same loss as about 7 random training runs, so it saves 6 runs. It is worth it when 6 runs cost more than Opus, which is when one run costs more than **$0.075**: about 1 hour on the laptop or **39 seconds on an H100** at $6.89 per hour. Half or more of the Opus cost is writing the agent's starting instructions into the cache, which happens once per run. The 25 runs of Part 2 used 73,000–81,000 tokens each, about the same as Part 1. We ran everything on the $20 per month Claude plan.

`python3 cost.py` prints these numbers and draws `cost.png`.

---

## Limits

- One model, 400 steps, about 9 seconds per run.
- 5 runs per setup can show differences of about 0.01, not 0.002.
- The losses fell smoothly in every run, with no spikes. On long runs where something goes wrong mid-training, supervising may matter more (a guess, not tested).
- The sweep did not include late-decay schedules. With them, it would have found 1.864 itself.
- OPAL's agent makes about 100 decisions on a 162M model; ours made at most 7.
