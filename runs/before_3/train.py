"""python3 train.py   train all 400 steps in one go with optimizer.py, then print the final validation loss.
Training runs only once: a second call does nothing. It saves the optimizer code it used to history/."""
import json, os, shutil, sys, time
import torch, torch.nn.functional as F

RUN = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RUN)
import fixed, optimizer

ck = os.path.join(RUN, "ckpt.pt")
if os.path.exists(ck): print("training finished"); raise SystemExit
m = fixed.fresh_model(); o = optimizer.Optimizer(m)
os.makedirs(os.path.join(RUN, "history"), exist_ok=True); shutil.copy(os.path.join(RUN, "optimizer.py"), os.path.join(RUN, "history", "optimizer_from_step_0.py"))
t0 = time.time(); losses = []
for i in range(fixed.STEPS):
    x, y = fixed.batch(i); loss = F.cross_entropy(m(x).flatten(0, 1), y.flatten())
    m.zero_grad(set_to_none=False); loss.backward(); o.step(i); losses.append(loss.item())
v = fixed.val(m)
torch.save(dict(model=m.state_dict(), opt=o.state_dict(), step=fixed.STEPS), ck)
row = dict(step=fixed.STEPS, val=round(v, 4), sec=round(time.time() - t0, 2))
open(os.path.join(RUN, "log.jsonl"), "a").write(json.dumps(row) + "\n")
print(f"training finished: final validation loss {v:.4f}")
