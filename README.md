# An AI agent changing the optimizer while a tiny LLM trains

Can an AI agent (Claude Sonnet) that watches the loss and changes the optimizer settings every 50 steps match a human-tuned learning-rate sweep, in a single run?

This is a small version of [Reactive Optimizers (OPAL)](https://xingjianbai.com/reactive-optimizers/), the MIT paper where an agent rewrites the optimizer during training.

| run | runs needed | final validation loss |
|---|---|---|
| default settings, fixed | 1 | 1.953 |
| Sonnet changes the settings every 50 steps | 1 | 1.900 |
| best of 19 fixed settings (learning-rate sweep, linear decay) | 19 | 1.878 |
| sweep's learning rates + Sonnet's late-decay schedule | 1 | **1.864** |

![curves](curves.png)

- In one run, Sonnet closed 70% of the gap between the default settings and the 19-run sweep.
- Sonnet found a better schedule shape than the sweep: full learning rate until step 250, then a fast drop. With the same learning rates, that shape beats a linear decay by 0.04 (1.907 vs 1.947).
- Sonnet missed that the embedding learning rate was 10x too low (0.003 vs the best 0.03); it only moved it to 0.004. The sweep found that.
- Putting the two together gives the best result: 1.864.

Short paper: [paper.md](paper.md). Sonnet's own reasons for each change: [decisions.md](decisions.md).

## Run it

```bash
python3 train.py static --lr 0.02 --adam 0.03 --decay --decay_from 250   # one fixed run, ~9 s on one CPU thread
python3 train.py reset                                                    # start an agent-controlled run
python3 train.py segment                                                  # train 50 steps with config.json; the agent edits config.json between segments
python3 plot.py
```

Model: 3-layer byte-level transformer, width 64, RoPE, QK-norm, RMSNorm, SwiGLU, no biases, on TinyShakespeare. Muon on hidden matrices, Adam on embeddings and norm gains. 400 steps, batch 16, context 64.
