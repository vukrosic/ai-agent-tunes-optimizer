# An AI agent changing the optimizer while a tiny LLM trains

Can an AI agent (Claude) that watches the loss and changes the optimizer settings every 50 steps match a human-tuned learning-rate sweep, in a single run? And does it do the same thing every time?

This is a small version of [Reactive Optimizers (OPAL)](https://xingjianbai.com/reactive-optimizers/), the MIT paper where an agent rewrites the optimizer during training.

| run | runs needed | final validation loss |
|---|---|---|
| default settings, fixed (same loss on 3 repeats: training is deterministic) | 1 | 1.953 |
| Sonnet changes the settings every 50 steps | 1 | 1.900 |
| Opus changes the settings every 50 steps, 3 separate runs | 1 each | 1.890, 1.888, 1.886 |
| best of 19 fixed settings (learning-rate sweep, linear decay) | 19 | 1.878 |
| sweep's learning rates + Sonnet's late-decay schedule | 1 | **1.864** |

![curves](curves.png)

- Training repeats exactly: the same fixed settings give 1.9526 three times. So different agent runs differ only by the agent's choices.
- Opus is consistent: 3 runs ended within 0.004 of each other (1.886 to 1.890) and closed 87% of the gap between the default settings and the 19-run sweep. Sonnet closed 70%.
- All 4 agent runs did the same thing: raise both learning rates together, then lower them toward the end. None touched momentum or weight decay.
- Sonnet found a better schedule shape than the sweep: full learning rate until step 250, then a fast drop. With the same learning rates, that shape beats a linear decay by 0.04 (1.907 vs 1.947).
- No agent found the biggest gain: the embedding (Adam) learning rate should be 1.5x the Muon one, not 0.15x. Every agent moved both rates together and kept the default ratio, so Adam stayed 4-8x too low. The sweep found that.
- Putting the two together gives the best result: 1.864.

Short paper: [paper.md](paper.md). Each agent's reasons for its changes: `runs/<name>/decisions.md`.

## Run it

```bash
python3 train.py static --lr 0.02 --adam 0.03 --decay --decay_from 250   # one fixed run, ~9 s on one CPU thread
python3 train.py reset --run NAME                                         # start an agent-controlled run in runs/NAME/
python3 train.py segment --run NAME                                       # train 50 steps with runs/NAME/config.json; the agent edits it between segments
python3 plot.py
```

Model: 3-layer byte-level transformer, width 64, RoPE, QK-norm, RMSNorm, SwiGLU, no biases, on TinyShakespeare. Muon on hidden matrices, Adam on embeddings and norm gains. 400 steps, batch 16, context 64.
