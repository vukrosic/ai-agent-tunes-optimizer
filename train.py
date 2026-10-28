"""Tiny byte-level LLM on TinyShakespeare whose optimizer settings can change during training.

python3 train.py static --lr 0.02 [--adam 0.003] [--decay [--decay_from N]]   one full run with fixed settings (lr optionally decays linearly to 0 from step N)
python3 train.py reset                        start an agent run: fresh model, config.json = default settings
python3 train.py segment                      train the next 50 steps with the settings in config.json, print the log
Model: 3 layers, width 64, 4 heads, RoPE, QK-norm, RMSNorm, SwiGLU, no biases, tied embeddings, 64 tokens of context.
Hidden matrices train with Muon; embeddings and norm gains train with Adam. 400 steps, batch 16, ~8 s on one CPU thread.
"""
import argparse, json, math, os, time
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
T, B, D, L, H, STEPS, SEG = 64, 16, 64, 3, 4, 400, 50
DEFAULT = dict(muon_lr=0.02, momentum=0.95, adam_lr=0.003, weight_decay=0.0)
data = torch.frombuffer(bytearray(open(os.path.join(HERE, "shakespeare.txt"), "rb").read()), dtype=torch.uint8).long()
cut = int(0.9 * len(data)); TR, VA = data[:cut], data[cut:]
gv = torch.Generator().manual_seed(1)
VAL = [torch.stack([VA[i:i + T + 1] for i in torch.randint(len(VA) - T - 1, (B,), generator=gv)]) for _ in range(8)]


def batch(step):
    """Batch of step i is the same in every run, so runs differ only by their optimizer."""
    idx = torch.randint(len(TR) - T - 1, (B,), generator=torch.Generator().manual_seed(1000 + step))
    x = torch.stack([TR[i:i + T + 1] for i in idx]); return x[:, :-1], x[:, 1:]


ANG = torch.outer(torch.arange(T).float(), 1e4 ** (-torch.arange(0, D // H, 2).float() / (D // H)))
COS, SIN = ANG.cos(), ANG.sin()
MASK = torch.ones(T, T, dtype=torch.bool).triu(1)


def rope(x):
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack([a * COS - b * SIN, b * COS + a * SIN], -1).flatten(-2)


class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.q, s.k, s.v, s.o = (nn.Linear(D, D, bias=False) for _ in range(4))
        s.gate, s.up, s.down = nn.Linear(D, 2 * D, bias=False), nn.Linear(D, 2 * D, bias=False), nn.Linear(2 * D, D, bias=False)
        s.n1, s.n2, s.qn, s.kn = nn.RMSNorm(D), nn.RMSNorm(D), nn.RMSNorm(D // H), nn.RMSNorm(D // H)

    def forward(s, x):
        b, t, _ = x.shape; h = s.n1(x)
        q, k, v = (m(h).view(b, t, H, D // H).transpose(1, 2) for m in (s.q, s.k, s.v))
        a = (rope(s.qn(q)) @ rope(s.kn(k)).transpose(-1, -2) / math.sqrt(D // H)).masked_fill(MASK[:t, :t], float("-inf"))
        o = a.softmax(-1) @ v
        x = x + s.o(o.transpose(1, 2).reshape(b, t, D)); h = s.n2(x)
        return x + s.down(F.silu(s.gate(h)) * s.up(h))


class LM(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb = nn.Embedding(256, D); s.blocks = nn.ModuleList(Block() for _ in range(L)); s.norm = nn.RMSNorm(D)
        nn.init.normal_(s.emb.weight, std=0.02)

    def forward(s, x):
        x = s.emb(x)
        for b in s.blocks: x = b(x)
        return s.norm(x) @ s.emb.weight.T


def ortho(g):
    """Newton-Schulz approximation of the nearest orthogonal matrix (Muon)."""
    x = g / (g.norm() + 1e-7); tr = x.shape[0] > x.shape[1]
    if tr: x = x.T
    for _ in range(5):
        a = x @ x.T; x = 3.4445 * x + (-4.7750 * a + 2.0315 * a @ a) @ x
    return x.T if tr else x


class Opt:
    def __init__(s, m):
        s.mats = [p for n, p in m.named_parameters() if p.ndim == 2 and n != "emb.weight"]
        s.rest = [p for n, p in m.named_parameters() if p.ndim != 2 or n == "emb.weight"]
        s.buf = [torch.zeros_like(p) for p in s.mats]
        s.m1, s.m2, s.t = [torch.zeros_like(p) for p in s.rest], [torch.zeros_like(p) for p in s.rest], 0

    @torch.no_grad()
    def step(s, c, mult=1.0):
        for p, b in zip(s.mats, s.buf):
            b.mul_(c["momentum"]).add_(p.grad)
            u = ortho(p.grad + c["momentum"] * b) * max(1, p.shape[0] / p.shape[1]) ** 0.5
            p.mul_(1 - c["muon_lr"] * mult * c["weight_decay"]).add_(u, alpha=-c["muon_lr"] * mult)
        s.t += 1
        for p, m1, m2 in zip(s.rest, s.m1, s.m2):
            m1.lerp_(p.grad, 0.1); m2.lerp_(p.grad.square(), 0.05)
            p.add_(m1 / (1 - 0.9 ** s.t) / ((m2 / (1 - 0.95 ** s.t)).sqrt() + 1e-8), alpha=-c["adam_lr"] * mult)


@torch.no_grad()
def val(m): return sum(F.cross_entropy(m(v[:, :-1]).flatten(0, 1), v[:, 1:].flatten()).item() for v in VAL) / len(VAL)


def train_steps(m, o, c, a, b, mult_fn=lambda i: 1.0):
    losses = []
    for i in range(a, b):
        x, y = batch(i); loss = F.cross_entropy(m(x).flatten(0, 1), y.flatten())
        m.zero_grad(set_to_none=False); loss.backward(); o.step(c, mult_fn(i)); losses.append(loss.item())
    return losses


def fresh():
    torch.manual_seed(0); m = LM(); return m, Opt(m)


def log(row):
    with open(os.path.join(HERE, "results.jsonl"), "a") as f: f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--lr", type=float, default=0.02); ap.add_argument("--decay", action="store_true"); ap.add_argument("--adam", type=float, default=0.003); ap.add_argument("--decay_from", type=int, default=0)
    a = ap.parse_args(); ck, cf, rl = (os.path.join(HERE, f) for f in ("run.pt", "config.json", "run_log.jsonl"))
    if a.cmd == "static":
        t0 = time.time(); m, o = fresh(); c = dict(DEFAULT, muon_lr=a.lr, adam_lr=a.adam); curve = []
        for s in range(0, STEPS, SEG):
            train_steps(m, o, c, s, s + SEG, (lambda i: min(1, (STEPS - i) / (STEPS - a.decay_from))) if a.decay else (lambda i: 1.0)); curve.append([s + SEG, round(val(m), 4)])
        r = dict(run=f"static lr {a.lr} adam {a.adam}" + (f" decay from {a.decay_from}" if a.decay else ""), val=curve[-1][1], curve=curve, sec=round(time.time() - t0, 1)); print(json.dumps(r)); log(r)
    elif a.cmd == "reset":
        m, o = fresh(); torch.save(dict(m=m.state_dict(), o=o.__dict__ | dict(mats=None, rest=None), step=0), ck)
        json.dump(DEFAULT, open(cf, "w"), indent=1); open(rl, "w").close(); print("fresh model, config.json =", json.dumps(DEFAULT))
    elif a.cmd == "segment":
        s = torch.load(ck, weights_only=False); m = LM(); m.load_state_dict(s["m"]); o = Opt(m)
        for k in ("buf", "m1", "m2", "t"): setattr(o, k, s["o"][k])
        step = s["step"]
        if step >= STEPS: print("training finished"); raise SystemExit
        c = dict(DEFAULT, **json.load(open(cf)))
        t0 = time.time(); losses = train_steps(m, o, c, step, step + SEG); v = val(m); step += SEG
        torch.save(dict(m=m.state_dict(), o=o.__dict__ | dict(mats=None, rest=None), step=step), ck)
        row = dict(step=step, config=c, train_loss_first=round(losses[0], 4), train_loss_last10=round(sum(losses[-10:]) / 10, 4),
                   train_loss_max=round(max(losses), 4), val=round(v, 4), sec=round(time.time() - t0, 2))
        open(rl, "a").write(json.dumps(row) + "\n"); print(json.dumps(row))
        if step >= STEPS: print(f"training finished: final val loss {v:.4f}")
