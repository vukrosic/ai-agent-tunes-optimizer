# The exact prompts

Each run got one of these prompts, with `<folder>` replaced by its own run folder. The folder was outside this repo, so the agent could not see other runs. The agent was Claude Opus 5.5 in Claude Code; it had all of its usual tools and saw everything inside its folder.

The opening and the rules are the same in every setup. Only the middle paragraph changes.

## Shared opening

> A small language model is about to train for 400 steps. Your folder is `<folder>`. It contains everything: the model and data (fixed.py, shakespeare.txt), the optimizer (optimizer.py) and the training script (train.py). The score is the validation loss after step 400 (lower is better).

## Middle paragraph, per setup

**before** (writes the optimizer once, never sees training):

> Change optimizer.py to improve the training. Then `cd` into your folder and run `python3 train.py` once: it trains all 400 steps in one go and prints the final validation loss. Nothing can be changed once training has started.

**free** (may change it any time, decides when):

> `cd` into your folder and run `python3 train.py --steps N` to train N more steps; it continues from the checkpoint and prints the validation loss. You may change optimizer.py at any time (it is reloaded on every call), change train.py, and measure anything inside the run.

**look** (training pauses at steps 100, 200 and 300, may change it):

> `cd` into your folder and run `python3 train.py --steps N` to train N more steps; it continues from the checkpoint and prints the validation loss. Training pauses at steps 100, 200 and 300. At each pause, look at how training is going and decide whether to change optimizer.py; the next call continues either way. You may also change optimizer.py at step 0 or any other time, change train.py, and measure anything inside the run.

**live** (watches training live, may edit it mid-run):

> Training runs from step 0 to step 400 without pausing and takes about 2 minutes. Start it with `cd <folder> && nohup python3 train.py > train.out 2>&1 &`. While it runs, it writes the training loss of every step (and the validation loss every 25 steps) to log.jsonl. Before every step it checks optimizer.py: if you edit the file, your new code takes over from the next step, keeping the optimizer state. To watch, run `python3 wait.py --until S`, which waits until training reaches step S (at most 100 s) and prints the new log rows, or read log.jsonl directly. You may change optimizer.py before starting and at any time while training runs, and measure anything inside the run.

**forced** (training pauses at steps 100, 200 and 300, must change it):

> `cd` into your folder and run `python3 train.py --steps N` to train N more steps; it continues from the checkpoint and prints the validation loss. Training stops at steps 100, 200 and 300. At each of those points, change optimizer.py to improve the rest of training; train.py continues only after optimizer.py has changed. You may also change optimizer.py at step 0 or any other time, change train.py, and measure anything inside the run.

## Shared rules and ending

> Rules:
> - Work only inside your folder. Do not read or open anything outside it.
> - Training runs once from step 0 to step 400, with no restarting or undoing (do not delete or edit ckpt.pt).
> - This is the only training: do not train any other copies of the model, no side experiments or sweeps, and run only one process at a time. The computer is in use by someone else.
> - Do not change fixed.py (model, data, batch order, validation set, 400 steps).
>
> Improve the training.
>
> When step 400 is done, reply with the final validation loss and one line per change you made.

Small wording differences: in **before**, the second rule reads "train.py runs only once; do not delete or edit ckpt.pt" and the last rule also forbids changing train.py. In **live**, the second rule starts "Start train.py only once", the third says "run no other heavy process", and the reply asks for the step of each change.
