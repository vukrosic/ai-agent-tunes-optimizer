"""python3 train.py --steps N   train N more steps, continuing from ckpt.pt (or from step 0 if there is none), then print the validation loss.
Training goes once from step 0 to 400. It stops at steps 100, 200 and 300, and continues only after optimizer.py has been changed. Each call appends a row to log.jsonl and saves the optimizer code it used to history/."""
import argparse, importlib, json, os, shutil, sys, time
import torch, torch.nn.functional as F

RUN = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RUN)
import fixed, optimizer

ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, required=True); a = ap.parse_args()
ck = os.path.join(RUN, "ckpt.pt")
m = fixed.fresh_model(); o = optimizer.Optimizer(m); step = 0
if os.path.exists(ck):
    s = torch.load(ck, weights_only=False); m.load_state_dict(s["model"]); o.load_state_dict(s["opt"]); step = s["step"]
CHECKS = (100, 200, 300)
end = min([fixed.STEPS, step + a.steps] + [c for c in CHECKS if c > step])
hist = os.path.join(RUN, "history"); prev = sorted(os.listdir(hist), key=lambda f: int(f.split("_")[-1][:-3]))[-1] if os.path.isdir(hist) and os.listdir(hist) else None
if step in CHECKS and prev and open(os.path.join(hist, prev)).read() == open(os.path.join(RUN, "optimizer.py")).read():
    print(f"step {step}: change optimizer.py to improve the rest of training before continuing"); raise SystemExit
if step >= fixed.STEPS: print("training finished"); raise SystemExit
os.makedirs(os.path.join(RUN, "history"), exist_ok=True); shutil.copy(os.path.join(RUN, "optimizer.py"), os.path.join(RUN, "history", f"optimizer_from_step_{step}.py"))
t0 = time.time(); losses = []
for i in range(step, end):
    x, y = fixed.batch(i); loss = F.cross_entropy(m(x).flatten(0, 1), y.flatten())
    m.zero_grad(set_to_none=False); loss.backward(); o.step(i); losses.append(loss.item())
v = fixed.val(m)
torch.save(dict(model=m.state_dict(), opt=o.state_dict(), step=end), ck)
row = dict(step=end, val=round(v, 4), train_loss_first=round(losses[0], 4), train_loss_last=round(losses[-1], 4),
           train_loss_max=round(max(losses), 4), sec=round(time.time() - t0, 2))
open(os.path.join(RUN, "log.jsonl"), "a").write(json.dumps(row) + "\n"); print(json.dumps(row))
if end in CHECKS: print(f"stopped at step {end}: change optimizer.py to improve the rest of training, then continue")
if end >= fixed.STEPS: print(f"training finished: final validation loss {v:.4f}")
