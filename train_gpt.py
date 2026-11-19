"""C2G train_gpt.py -- a self-contained, wallclock-budgeted GPT trainer.

Design goals (matching the C2G constraint set as far as this hardware allows):
  * hard wallclock budget (default 600 s = 10 min), enforced by a timer, not by
    a fixed step count -- the challenge's real constraint is *time*, and on a
    slower machine the same step budget would silently break the rule;
  * tiny model + tiny tokenizer stay inside a 16 MB artifact envelope;
  * every run writes a JSONL log with per-step loss, lr, tokens/s and elapsed,
    so the ablation deltas in the report can be re-derived from the logs;
  * final BPB is measured on a held-out byte slice never seen in training.

Honest scope note: this box has a single RTX 4060 Laptop (8 GB) and no CUDA
torch is installable at the available bandwidth, so the runs reported in this
repo are CPU runs with a reduced model. They are a *scaled replication* of the
C2G procedure, not H100 leaderboard numbers.

Usage:
  python train_gpt.py --data-dir DIR --vocab 8192 --budget-sec 600 --seed 1 \
      --n-layer 6 --n-head 6 --n-embd 384 --block 256 --batch 16 --out runs/x
"""

import argparse
import json
import math
import os
import time
import zlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(int(os.environ.get("C2G_THREADS", "20")))


# ---------------------------------------------------------------- model pieces
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        return self.ln(x)


def build_norm(kind, dim):
    return RMSNorm(dim) if kind == "rmsnorm" else LayerNorm(dim)


def rope_cache(block, head_dim, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(block).float()
    freqs = torch.outer(t, inv)
    # half-split convention (GPT-NeoX style): apply_rope chunks x into two halves,
    # so cos/sin carry head_dim//2 frequencies, not head_dim.
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, T, D)
    T = x.size(2)
    cos = cos[:T].to(x.dtype).unsqueeze(0).unsqueeze(0)
    sin = sin[:T].to(x.dtype).unsqueeze(0).unsqueeze(0)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attn(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.h = cfg.n_head
        self.d = cfg.n_embd
        self.qkv = nn.Linear(self.d, 3 * self.d, bias=False)
        self.proj = nn.Linear(self.d, self.d, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.d, dim=2)
        q = q.view(B, T, self.h, C // self.h).transpose(1, 2)
        k = k.view(B, T, self.h, C // self.h).transpose(1, 2)
        v = v.view(B, T, self.h, C // self.h).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = int(cfg.mlp_mult * cfg.n_embd)
        self.swiglu = cfg.act == "swiglu"
        if self.swiglu:
            self.w1 = nn.Linear(cfg.n_embd, hidden, bias=False)
            self.w3 = nn.Linear(cfg.n_embd, hidden, bias=False)
            self.w2 = nn.Linear(hidden, cfg.n_embd, bias=False)
        else:
            self.w1 = nn.Linear(cfg.n_embd, hidden, bias=False)
            self.w2 = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        if self.swiglu:
            h = F.silu(self.w1(x)) * self.w3(x)
        else:
            h = F.gelu(self.w1(x))
        return self.drop(self.w2(h))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1 = build_norm(cfg.norm, cfg.n_embd)
        self.attn = Attn(cfg)
        self.n2 = build_norm(cfg.norm, cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        x = x + self.mlp(self.n2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.nf = build_norm(cfg.norm, cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok.weight
        self.block = cfg.block
        cos, sin = rope_cache(cfg.block, cfg.n_embd // cfg.n_head)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        x = self.tok(idx)
        for b in self.blocks:
            x = b(x, self.cos, self.sin)
        logits = self.head(self.nf(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.reshape(-1), reduction="sum"
        )
        return logits, loss


# ------------------------------------------------------------------ optimizers
@torch.no_grad()
def zeropower_via_newtonschulz5(g, steps=5, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = g.float()
    if X.size(0) > X.size(1):
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if g.size(0) > g.size(1):
        X = X.T
    return X.to(g.dtype)


class Muon(torch.optim.Optimizer):
    """Muon for 2-D hidden matrices (Keller Jordan), AdamW handles the rest."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "mom" not in st:
                    st["mom"] = torch.zeros_like(g)
                st["mom"].mul_(group["momentum"]).add_(g)
                u = g.add(st["mom"], alpha=group["momentum"]) if group["nesterov"] else st["mom"]
                u = zeropower_via_newtonschulz5(u.reshape(u.size(0), -1), steps=5)
                scale = max(1.0, p.size(0) / max(1.0, p.size(1))) ** 0.5
                p.add_(u.reshape(p.shape), alpha=-group["lr"] * scale)


def split_params(model):
    """muon gets >=2-D hidden matrices; everything else goes to AdamW."""
    muon, adam = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and "tok" not in name and "head" not in name:
            muon.append(p)
        else:
            adam.append(p)
    return muon, adam


# ----------------------------------------------------------------- data access
class BinData:
    def __init__(self, path):
        self.ids = np.load(path, mmap_mode="r")
        self.n = int(self.ids.shape[0])

    def batch(self, batch, block, rng):
        starts = rng.integers(0, self.n - block - 1, size=batch)
        x = np.stack([np.asarray(self.ids[s:s + block], dtype=np.int64) for s in starts])
        y = np.stack([np.asarray(self.ids[s + 1:s + 1 + block], dtype=np.int64) for s in starts])
        return torch.from_numpy(x), torch.from_numpy(y)


@torch.no_grad()
def eval_bpb(model, ids, block, bpt, max_tokens=65536):
    """Held-out bits-per-byte.

    BPB must normalise by the *bytes actually covered* by the evaluated tokens,
    not by the whole eval slice. We convert tokens->bytes with the eval set's own
    bytes-per-token ratio (bpt = eval_bytes / eval_tokens), which is how BPE-based
    BPB is normally reported.

    Windows are spread evenly over the eval stream (stride >= block) so the
    estimate is not dominated by the first few thousand bytes.
    """
    model.eval()
    n = int(ids.shape[0])
    n_win = max(1, min(max_tokens // block, max(1, (n - block - 1) // block)))
    stride = max(block, (n - block - 1) // n_win)
    total_nats, total_tok, wins = 0.0, 0, 0
    for s in range(0, n - block - 1, stride):
        x = torch.from_numpy(np.asarray(ids[s:s + block], dtype=np.int64))[None]
        y = torch.from_numpy(np.asarray(ids[s + 1:s + 1 + block], dtype=np.int64))[None]
        _, loss = model(x, y)
        total_nats += float(loss.detach())
        total_tok += int(y.numel())
        wins += 1
    model.train()
    mean_nats = total_nats / max(total_tok, 1)
    bpb = mean_nats / (bpt * math.log(2.0))
    return bpb, total_tok, wins


# ------------------------------------------------------- artifact size probing
def artifact_estimate(model, tok_bytes):
    """Real int8 + zlib size, plus an int6 bit-budget estimate."""
    buf = bytearray()
    n_params = 0
    for name, p in model.state_dict().items():
        if p.dtype.is_floating_point and p.ndim >= 2:
            w = p.detach().float()
            scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
            q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
            buf += q.numpy().tobytes()
            buf += scale.to(torch.float16).numpy().tobytes()
            n_params += int(w.numel())
        else:
            buf += p.detach().to(torch.float16).numpy().tobytes()
            n_params += int(p.numel())
    q8 = len(zlib.compress(bytes(buf), 9)) + tok_bytes
    q6 = int(math.ceil(n_params * 6 / 8.0)) + tok_bytes + 4096
    return dict(raw_params=n_params, int8_zlib_bytes=q8, int6_est_bytes=q6)


def save_quant(model, out_dir, tok_path):
    """Persist the *real* submission artifact: int8 weights + fp16 scales,
    zlib-compressed, together with the tokenizer. Returns on-disk byte counts
    so the 16,000,000-byte budget can be audited instead of trusted."""
    os.makedirs(out_dir, exist_ok=True)
    buf = bytearray()
    for name, p in model.state_dict().items():
        if p.dtype.is_floating_point and p.ndim >= 2:
            w = p.detach().float()
            scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
            q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
            buf += name.encode("utf-8") + b"\x00"
            buf += q.numpy().tobytes()
            buf += scale.to(torch.float16).numpy().tobytes()
        else:
            buf += name.encode("utf-8") + b"\x00"
            buf += p.detach().to(torch.float16).numpy().tobytes()
    packed = zlib.compress(bytes(buf), 9)
    art_path = os.path.join(out_dir, "weights_int8_zlib.bin")
    with open(art_path, "wb") as f:
        f.write(packed)
    tok_bytes = os.path.getsize(tok_path)
    with open(tok_path, "rb") as f:
        tok_packed = zlib.compress(f.read(), 9)
    tok_out = os.path.join(out_dir, "bytebpe.json.zlib")
    with open(tok_out, "wb") as f:
        f.write(tok_packed)
    code_bytes = os.path.getsize(os.path.abspath(__file__))
    total = len(packed) + len(tok_packed) + code_bytes
    return dict(weights_zlib=len(packed), tok_raw=tok_bytes,
                tok_zlib=len(tok_packed), code=code_bytes,
                total=total, budget=16000000,
                fits=bool(total <= 16000000))


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--n-embd", type=int, default=384)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--mlp-mult", type=float, default=4.0)
    ap.add_argument("--act", default="gelu", choices=["gelu", "swiglu"])
    ap.add_argument("--norm", default="rmsnorm", choices=["rmsnorm", "layernorm"])
    ap.add_argument("--tie-weights", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "muon"])
    ap.add_argument("--muon-lr", type=float, default=0.02)
    ap.add_argument("--budget-sec", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--save-quant", default="",
                    help="if set, persist the real int8+zlib artifact here")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    tok_dir = os.path.join(args.data_dir, "tok%d" % args.vocab)
    train_path = os.path.join(tok_dir, "train_ids.npy")
    eval_path = os.path.join(tok_dir, "eval_ids.npy")
    with open(os.path.join(tok_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)

    train = BinData(train_path)
    eval_ids = np.load(eval_path)

    model = GPT(args)
    n_params = sum(p.numel() for p in model.parameters())
    muon, adam = split_params(model)
    opt_adam = torch.optim.AdamW(
        adam, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )
    opt_muon = Muon(muon, lr=args.muon_lr) if args.optimizer == "muon" and muon else None

    log_path = os.path.join(args.out, "train.jsonl")
    log = open(log_path, "w", encoding="utf-8")
    log.write(json.dumps(dict(event="start", tag=args.tag, seed=args.seed,
                              config=vars(args), n_params=n_params,
                              n_params_muon=sum(p.numel() for p in muon),
                              data=meta), ensure_ascii=True) + "\n")

    # lr schedule is expressed in *time*, because the budget is in time
    t0 = time.time()
    step, tok_seen, run_nats = 0, 0, 0.0
    warm_frac = args.warmup_frac
    est_total_steps = 1
    while True:
        elapsed = time.time() - t0
        if elapsed >= args.budget_sec:
            break
        frac = min(1.0, elapsed / args.budget_sec)
        if frac < warm_frac:
            lr_mul = frac / max(warm_frac, 1e-9)
        else:
            p = (frac - warm_frac) / max(1e-9, 1.0 - warm_frac)
            lr_mul = args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p))
        for g in opt_adam.param_groups:
            g["lr"] = args.lr * lr_mul
        if opt_muon is not None:
            for g in opt_muon.param_groups:
                g["lr"] = args.muon_lr * lr_mul

        x, y = train.batch(args.batch, args.block, rng)
        _, loss = model(x, y)
        loss = loss / y.numel()
        opt_adam.zero_grad(set_to_none=True)
        if opt_muon is not None:
            opt_muon.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt_adam.step()
        if opt_muon is not None:
            opt_muon.step()

        step += 1
        tok_seen += int(y.numel())
        run_nats += float(loss.detach()) * int(y.numel())
        est_total_steps = max(1, step)

        if step % 20 == 0:
            el = time.time() - t0
            rec = dict(event="step", step=step, elapsed=round(el, 2),
                       loss=round(float(loss.detach()), 5),
                       avg_loss_so_far=round(run_nats / tok_seen, 5),
                       lr=round(args.lr * lr_mul, 6), tok_seen=tok_seen,
                       tok_per_sec=round(tok_seen / max(el, 1e-9)))
            log.write(json.dumps(rec) + "\n")
            log.flush()

    train_secs = time.time() - t0
    bpt = meta["eval_bytes"] / float(meta["eval_tokens"])
    bpb, ev_tok, wins = eval_bpb(model, eval_ids, args.block, bpt)
    art = artifact_estimate(model, os.path.getsize(os.path.join(tok_dir, "bytebpe.json")))
    if args.save_quant:
        art["on_disk"] = save_quant(
            model, args.save_quant, os.path.join(tok_dir, "bytebpe.json"))
    summary = dict(event="final", tag=args.tag, seed=args.seed, bpb=round(bpb, 5),
                   train_secs=round(train_secs, 1), steps=step, tokens_seen=tok_seen,
                   tokens_per_sec=round(tok_seen / max(train_secs, 1e-9)),
                   eval_tokens=ev_tok, eval_windows=wins,
                   eval_bytes=meta["eval_bytes"], n_params=n_params,
                   train_loss=round(run_nats / tok_seen, 5), artifact=art)
    log.write(json.dumps(summary) + "\n")
    log.close()
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    torch.save(dict(model=model.state_dict(), config=vars(args)),
               os.path.join(args.out, "ckpt.pt"))
    print("FINAL tag=%s seed=%d bpb=%.5f steps=%d tok=%d tokps=%d int8_bytes=%d int6_est=%d secs=%.1f"
          % (args.tag, args.seed, bpb, step, tok_seen,
             int(tok_seen / max(train_secs, 1e-9)), art["int8_zlib_bytes"],
             art["int6_est_bytes"], train_secs))


if __name__ == "__main__":
    main()
