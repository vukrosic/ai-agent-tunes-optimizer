"""python3 setups/check.py NAME...   for each run in runs/NAME: recompute the final validation loss from ckpt.pt with the untouched fixed.py,
and count the optimizer versions it used (and from which steps)."""
import os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import fixed
S = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")
for name in sys.argv[1:]:
    d = os.path.join(S, name); s = torch.load(os.path.join(d, "ckpt.pt"), weights_only=False)
    m = fixed.fresh_model(); m.load_state_dict(s["model"])
    h = sorted(os.listdir(os.path.join(d, "history")), key=lambda f: int(f.split("_")[-1][:-3])); last, starts = None, []
    for f in h:
        c = open(os.path.join(d, "history", f), "rb").read()
        if c != last: starts.append(int(f.split("_")[-1][:-3])); last = c
    print(f"{name}: step {s['step']}, val {fixed.val(m):.4f}, optimizer versions from steps {starts}")
