"""python3 train.py   train all 400 steps without pausing, slowed to about 0.25 s per step so you can watch.
Before every step it checks optimizer.py; if the file changed, the new code takes over from that step on (keeping the optimizer state).
Every step appends a row to log.jsonl (training loss; validation loss every 25 steps). Each optimizer version is saved to history/."""
import importlib, json, os, shutil, sys, time, traceback
import torch, torch.nn.functional as F

RUN = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RUN)
import fixed, optimizer

ck = os.path.join(RUN, "ckpt.pt"); log = os.path.join(RUN, "log.jsonl"); src = os.path.join(RUN, "optimizer.py")
if os.path.exists(ck): print("training finished"); raise SystemExit
def write(row): open(log, "a").write(json.dumps(row) + "\n")
m = fixed.fresh_model(); o = optimizer.Optimizer(m); code = open(src).read()
os.makedirs(os.path.join(RUN, "history"), exist_ok=True); shutil.copy(src, os.path.join(RUN, "history", "optimizer_from_step_0.py"))
for i in range(fixed.STEPS):
    new = open(src).read()
    if new != code:
        code = new
        try:
            importlib.reload(optimizer); n = optimizer.Optimizer(m); n.load_state_dict(o.state_dict()); o = n
            shutil.copy(src, os.path.join(RUN, "history", f"optimizer_from_step_{i}.py")); write(dict(step=i, event="new optimizer.py loaded"))
        except Exception:
            write(dict(step=i, event="new optimizer.py failed to load, kept the previous one", error=traceback.format_exc(limit=1)))
    x, y = fixed.batch(i); loss = F.cross_entropy(m(x).flatten(0, 1), y.flatten())
    m.zero_grad(set_to_none=False); loss.backward(); o.step(i)
    row = dict(step=i + 1, train_loss=round(loss.item(), 4))
    if (i + 1) % 25 == 0: row["val"] = round(fixed.val(m), 4)
    write(row); time.sleep(0.25)
v = fixed.val(m)
torch.save(dict(model=m.state_dict(), opt=o.state_dict(), step=fixed.STEPS), ck)
write(dict(step=fixed.STEPS, final_val=round(v, 4))); print(f"training finished: final validation loss {v:.4f}")
