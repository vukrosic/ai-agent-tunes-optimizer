"""Fixed part of the open setup: model, data, batch order, validation set, 400 steps. Agents may not change this file."""
import fcntl, json, math, os, subprocess, time
# League of Legends runs on this Mac: lowest priority, background QoS, and only one training at a time across all runs
os.nice(19); subprocess.run(["taskpolicy", "-b", "-p", str(os.getpid())], capture_output=True)
_LOCK = open("/tmp/agent-optimizer.lock", "w"); fcntl.flock(_LOCK, fcntl.LOCK_EX)
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
T, B, D, L, H, STEPS = 64, 16, 64, 3, 4, 400
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


@torch.no_grad()
def val(m): return sum(F.cross_entropy(m(v[:, :-1]).flatten(0, 1), v[:, 1:].flatten()).item() for v in VAL) / len(VAL)


def fresh_model():
    torch.manual_seed(0); return LM()
