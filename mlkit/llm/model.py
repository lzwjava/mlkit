"""Decoder-only Transformer language model (GPT-style) on the numpy engine.

Supports both pre-training objectives from the reference table:

* ``clm`` — causal / autoregressive next-token prediction (GPT)
* ``mlm`` — masked language modelling with a bidirectional mask (BERT-style)

Inference supports incremental decoding with a **KV cache**: each layer
stores past keys/values so only the newest token is forwarded.
"""

from __future__ import annotations

import numpy as np

from ..dl import nn
from ..dl.tensor import Tensor, cat, cross_entropy, gather_rows
from ..dl.optim import get_optimizer, clip_grad_norm_
from ..dl.train import iterate_minibatches
from ..utils import curve


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, n_head: int = 4,
                 n_layer: int = 2, block_size: int = 64, mlp_ratio: int = 4,
                 dropout: float = 0.0, tie_weights: bool = True, seed: int = 0):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.block_size = block_size
        self.n_layer = n_layer
        self.tie_weights = tie_weights

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = [nn.TransformerBlock(d_model, n_head, block_size,
                                           mlp_ratio=mlp_ratio,
                                           dropout=dropout, causal=True)
                       for _ in range(n_layer)]
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False, init="normal",
                              std=0.02)
        if tie_weights:
            self.head.weight = self.tok_emb.weight     # shared tensor

    # ------------------------------------------------------------------
    def set_causal(self, causal: bool):
        for blk in self.blocks:
            blk.attn.causal = causal
    def forward(self, idx, targets=None, past_kv=None, use_cache: bool = False,
                causal: bool = True, return_logits: bool = True):
        """idx: int array (B, T). Returns (logits, loss, present_kv)."""
        idx = np.asarray(idx, dtype=np.int64)
        if idx.ndim == 1:
            idx = idx[None, :]
        B, T = idx.shape
        past_len = 0 if not past_kv or past_kv[0] is None else past_kv[0][0].shape[2]
        if past_len + T > self.block_size:
            raise ValueError(f"sequence length {past_len + T} exceeds "
                             f"context window {self.block_size}")

        pos_ids = np.arange(past_len, past_len + T)
        h = self.tok_emb(idx) + self.pos_emb(pos_ids)
        h = self.drop(h)

        was_causal = self.blocks[0].attn.causal if self.blocks else True
        if not causal:
            self.set_causal(False)

        present = []
        for i, blk in enumerate(self.blocks):
            p = None if past_kv is None else past_kv[i]
            h, kv = blk(h, past_kv=p, use_cache=use_cache)
            if use_cache:
                present.append(kv)

        if not causal:
            self.set_causal(was_causal)

        h = self.ln_f(h)
        logits = h @ self.head.weight.transpose()      # (B, T, V)
        if self.head.bias is not None:
            logits = logits + self.head.bias

        loss = None
        if targets is not None:
            targets = np.asarray(targets, dtype=np.int64)
            if targets.ndim == 1:                      # already flattened
                loss = cross_entropy(
                    logits.reshape(B * T, self.vocab_size), targets)
            elif targets.shape == idx.shape:            # CLM / MLM per position
                loss = cross_entropy(logits.reshape(B * T, self.vocab_size),
                                     targets.reshape(-1))
        out = (logits, loss, present) if return_logits else (None, loss, present)
        return out


# --------------------------------------------------------------------------
# dataset builders for the two objectives
# --------------------------------------------------------------------------

def build_clm_batch(tokens: np.ndarray, batch: int, block: int, rng):
    """Next-token prediction: x = t[i:i+T], y = t[i+1:i+T+1]."""
    n = len(tokens) - block - 1
    ix = rng.integers(0, n, size=batch)
    x = np.stack([tokens[i:i + block] for i in ix]).astype(np.int64)
    y = np.stack([tokens[i + 1:i + block + 1] for i in ix]).astype(np.int64)
    return x, y


def build_mlm_batch(tokens: np.ndarray, batch: int, block: int, rng,
                    mask_id: int, mask_prob: float = 0.15,
                    vocab_size: int | None = None):
    """Masked LM: mask_prob of positions replaced by <mask>; loss on them."""
    n = len(tokens) - block
    ix = rng.integers(0, n, size=batch)
    x = np.stack([tokens[i:i + block] for i in ix]).astype(np.int64)
    y = x.copy()
    sel = rng.random(x.shape) < mask_prob
    if not sel.any():
        sel[rng.integers(0, block), 0] = True
    x[sel] = mask_id
    y = np.where(sel, y, -100)          # -100 ignored via masked targets
    return x, y, sel


def mlm_loss(logits: Tensor, targets_masked: np.ndarray) -> Tensor:
    """Cross-entropy on masked positions only."""
    B, T, V = logits.shape
    mask = targets_masked >= 0
    rows = np.where(mask.reshape(-1))[0]
    flat = logits.reshape(B * T, V)
    sub = flat[rows]
    return cross_entropy(sub, targets_masked.reshape(-1)[rows])


# --------------------------------------------------------------------------
# training loop
# --------------------------------------------------------------------------

def train_lm(model: TinyGPT, tokens: np.ndarray, *, objective: str = "clm",
             mask_id: int | None = None, steps: int = 500, batch_size: int = 32,
             block_size: int | None = None, lr: float = 3e-3,
             optimizer: str = "adamw", weight_decay: float = 0.01,
             clip: float = 1.0, val_frac: float = 0.1, seed: int = 0,
             log_every: int = 50, verbose: bool = True, lora_params=None) -> dict:
    """Pre-train (or fine-tune) a TinyGPT with CLM or MLM loss.

    Returns a history dict with train/val loss and validation perplexity.
    """
    rng = np.random.default_rng(seed)
    block = block_size or model.block_size
    tokens = np.asarray(tokens, dtype=np.int64)

    n_val = max(int(len(tokens) * val_frac), block + 2)
    train_tok, val_tok = tokens[:-n_val], tokens[-n_val:]

    params = lora_params if lora_params is not None else model.parameters()
    opt = get_optimizer(optimizer, params, lr=lr, weight_decay=weight_decay)
    model.train(True)

    if verbose:
        info = opt.info()
        unique = model.num_params(trainable_only=False)
        raw = sum(p.data.size for _, p in model.named_parameters())
        obj_name = "next-token (CLM)" if objective == "clm" else "masked (MLM)"
        print(f"    model: {unique:,} params "
              f"({raw - unique:,} tied)  objective: {obj_name}")
        print(f"    opt={info['optimizer']} lr={info['lr']:g} "
              f"wd={weight_decay:g}  batch={batch_size}  ctx={block}")
        print(f"    corpus: {len(train_tok):,} train / {len(val_tok):,} val tokens")

    hist = {"train_loss": [], "val_loss": [], "val_ppl": []}

    def step_batch() -> float:
        if objective == "clm":
            x, y = build_clm_batch(train_tok, batch_size, block, rng)
            logits, loss, _ = model(x, targets=y)
        else:
            if mask_id is None:
                raise ValueError("MLM objective needs mask_id")
            x, y, sel = build_mlm_batch(train_tok, batch_size, block, rng,
                                        mask_id, vocab_size=model.vocab_size)
            logits, _, _ = model(x, causal=False)
            loss = mlm_loss(logits, y)
        opt.zero_grad()
        loss.backward()
        if clip:
            clip_grad_norm_(params, clip)
        opt.step()
        return float(loss.item())

    def evaluate() -> tuple[float, float]:
        """Sliding-window val loss (CLM) / masked val loss (MLM)."""
        model.eval()
        losses = []
        stride = block
        starts = range(0, max(len(val_tok) - block - 1, 1), stride)
        for s in starts:
            chunk = val_tok[s:s + block + 1]
            if len(chunk) < 8:
                continue
            if objective == "clm":
                x = chunk[:-1][None, :]
                y = chunk[1:][None, :]
                _, loss, _ = model(x, targets=y)
            else:
                r = np.random.default_rng(s)
                x = chunk[None, :min(block, len(chunk))].copy()
                y = x.copy()
                sel = r.random(x.shape) < 0.15
                if not sel.any():
                    sel[0, 0] = True
                x[sel] = mask_id
                y = np.where(sel, y, -100)
                logits, _, _ = model(x, causal=False)
                loss = mlm_loss(logits, y)
            losses.append(float(loss.item()))
        model.train(True)
        vl = float(np.mean(losses)) if losses else float("nan")
        return vl, float(np.exp(min(vl, 20.0)))

    for s in range(steps):
        tl = step_batch()
        if s % log_every == 0 or s == steps - 1:
            vl, ppl = evaluate()
            hist["train_loss"].append(tl)
            hist["val_loss"].append(vl)
            hist["val_ppl"].append(ppl)
            if verbose:
                print(f"    step {s:5d}  train {tl:.4f}  val {vl:.4f}  "
                      f"ppl {ppl:.2f}   {curve(hist['train_loss'])}")

    final_vl, final_ppl = evaluate()
    hist["final_val_loss"] = final_vl
    hist["final_val_ppl"] = final_ppl
    if verbose:
        print(f"    final: val loss {final_vl:.4f}  perplexity {final_ppl:.2f}")
    return hist