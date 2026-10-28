"""The optimizer. train.py re-imports this file on every call, so changes apply from the next step on.
Muon (Newton-Schulz, Nesterov momentum) on the hidden weight matrices; Adam on the embedding table and the norm gains."""
import torch


def ortho(g):
    """Newton-Schulz approximation of the nearest orthogonal matrix."""
    x = g / (g.norm() + 1e-7); tr = x.shape[0] > x.shape[1]
    if tr: x = x.T
    for _ in range(5):
        a = x @ x.T; x = 3.4445 * x + (-4.7750 * a + 2.0315 * a @ a) @ x
    return x.T if tr else x


class Optimizer:
    def __init__(s, model):
        s.mats = [p for n, p in model.named_parameters() if p.ndim == 2 and n != "emb.weight"]
        s.rest = [p for n, p in model.named_parameters() if p.ndim != 2 or n == "emb.weight"]
        s.state = dict(buf=[torch.zeros_like(p) for p in s.mats], m1=[torch.zeros_like(p) for p in s.rest],
                       m2=[torch.zeros_like(p) for p in s.rest], t=0)

    @torch.no_grad()
    def step(s, step):
        """step = the global training step (0 to 399)."""
        muon_lr, momentum, adam_lr = 0.03, 0.95, 0.008
        warm, decay_from, total = 20, 200, 400
        f = min(1.0, (step + 1) / warm) * (1.0 if step < decay_from else max(0.0, (total - step) / (total - decay_from)))
        muon_lr, adam_lr = muon_lr * f, adam_lr * f
        for p, b in zip(s.mats, s.state["buf"]):
            b.mul_(momentum).add_(p.grad)
            p.add_(ortho(p.grad + momentum * b) * max(1, p.shape[0] / p.shape[1]) ** 0.5, alpha=-muon_lr)
        s.state["t"] += 1; t = s.state["t"]
        for p, m1, m2 in zip(s.rest, s.state["m1"], s.state["m2"]):
            m1.lerp_(p.grad, 0.1); m2.lerp_(p.grad.square(), 0.05)
            p.add_(m1 / (1 - 0.9 ** t) / ((m2 / (1 - 0.95 ** t)).sqrt() + 1e-8), alpha=-adam_lr)

    def state_dict(s): return s.state

    def load_state_dict(s, state): s.state.update(state)
