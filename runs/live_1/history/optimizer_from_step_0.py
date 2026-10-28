"""The optimizer. train.py re-imports this file on every call, so changes apply from the next step on.
Muon (Newton-Schulz, Nesterov momentum) on the hidden weight matrices; Adam on the embedding table and the norm gains.
Changes: higher base lrs, linear warmdown to ~0 at the end, and an lr probe on held-out training batches at a few steps."""
import json, os
import torch, torch.nn.functional as F
import fixed

HERE = os.path.dirname(os.path.abspath(__file__))
MUON_LR, ADAM_LR, MOMENTUM = 0.03, 0.006, 0.95
DECAY_START, FINAL_FRAC = 160, 0.0
PROBE_STEPS = {40, 80, 120}
PROBE_SCALES = (0.5, 1.0, 2.0)


def ortho(g):
    """Newton-Schulz approximation of the nearest orthogonal matrix."""
    x = g / (g.norm() + 1e-7); tr = x.shape[0] > x.shape[1]
    if tr: x = x.T
    for _ in range(5):
        a = x @ x.T; x = 3.4445 * x + (-4.7750 * a + 2.0315 * a @ a) @ x
    return x.T if tr else x


def sched(step):
    if step < DECAY_START: return 1.0
    f = (step - DECAY_START) / (fixed.STEPS - DECAY_START)
    return 1.0 - (1.0 - FINAL_FRAC) * f


class Optimizer:
    def __init__(s, model):
        s.model = model
        s.mats = [p for n, p in model.named_parameters() if p.ndim == 2 and n != "emb.weight"]
        s.rest = [p for n, p in model.named_parameters() if p.ndim != 2 or n == "emb.weight"]
        s.state = dict(buf=[torch.zeros_like(p) for p in s.mats], m1=[torch.zeros_like(p) for p in s.rest],
                       m2=[torch.zeros_like(p) for p in s.rest], t=0)

    @torch.no_grad()
    def probe_loss(s):
        tot = 0.0
        for k in range(2):
            x, y = fixed.batch(100000 + k)
            tot += F.cross_entropy(s.model(x).flatten(0, 1), y.flatten()).item()
        return tot / 2

    @torch.no_grad()
    def step(s, step):
        """step = the global training step (0 to 399)."""
        lm = sched(step)
        deltas = []
        for p, b in zip(s.mats, s.state["buf"]):
            b.mul_(MOMENTUM).add_(p.grad)
            deltas.append(ortho(p.grad + MOMENTUM * b) * max(1, p.shape[0] / p.shape[1]) ** 0.5 * (-MUON_LR * lm))
        s.state["t"] += 1; t = s.state["t"]
        for p, m1, m2 in zip(s.rest, s.state["m1"], s.state["m2"]):
            m1.lerp_(p.grad, 0.1); m2.lerp_(p.grad.square(), 0.05)
            deltas.append(m1 / (1 - 0.9 ** t) / ((m2 / (1 - 0.95 ** t)).sqrt() + 1e-8) * (-ADAM_LR * lm))
        params = s.mats + s.rest
        if step in PROBE_STEPS:
            row = dict(step=step, base=s.probe_loss())
            nm = len(s.mats)
            for name, idx in (("all", range(len(params))), ("muon", range(nm)), ("adam", range(nm, len(params)))):
                for c in PROBE_SCALES:
                    for i in idx: params[i].add_(deltas[i], alpha=c)
                    row[f"{name}_x{c}"] = round(s.probe_loss(), 5)
                    for i in idx: params[i].sub_(deltas[i], alpha=c)
            open(os.path.join(HERE, "probe.jsonl"), "a").write(json.dumps(row) + "\n")
        for p, d in zip(params, deltas): p.add_(d)

    def state_dict(s): return s.state

    def load_state_dict(s, state): s.state.update(state)
