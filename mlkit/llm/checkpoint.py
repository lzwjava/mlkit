"""Save/load TinyGPT checkpoints (weights + config + tokenizer)."""

from __future__ import annotations

import json
import os

import numpy as np

from .model import TinyGPT
from .tokenizer import tokenizer_from_state


def save_checkpoint(model: TinyGPT, tokenizer, path: str,
                    meta: dict | None = None) -> str:
    os.makedirs(path, exist_ok=True)
    config = {"vocab_size": model.vocab_size, "d_model": model.d_model,
              "n_head": model.blocks[0].attn.n_head if model.blocks else 4,
              "n_layer": model.n_layer, "block_size": model.block_size,
              "mlp_ratio": model.blocks[0].mlp.layers[0].out_features
              // model.d_model if model.blocks else 4,
              "tie_weights": model.tie_weights}
    with open(os.path.join(path, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)
    tokenizer.save(os.path.join(path, "tokenizer.json"))
    state = {}
    seen = set()
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        state[name] = p.data
    np.savez(os.path.join(path, "weights.npz"), **state)
    if meta:
        with open(os.path.join(path, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
    return path


def load_checkpoint(path: str) -> tuple[TinyGPT, object, dict]:
    with open(os.path.join(path, "config.json")) as fh:
        cfg = json.load(fh)
    tokenizer = tokenizer_from_state(
        json.load(open(os.path.join(path, "tokenizer.json"))))
    model = TinyGPT(vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
                    n_head=cfg["n_head"], n_layer=cfg["n_layer"],
                    block_size=cfg["block_size"],
                    mlp_ratio=cfg.get("mlp_ratio", 4),
                    tie_weights=cfg.get("tie_weights", True))
    data = np.load(os.path.join(path, "weights.npz"))
    own = dict(model.named_parameters())
    loaded = 0
    for k in data.files:
        if k in own:
            own[k].data[...] = data[k]
            loaded += 1
    missing = [k for k in own if k not in data.files]
    meta = {}
    meta_path = os.path.join(path, "meta.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
    info = {"loaded": loaded, "missing": missing, "config": cfg}
    return model, tokenizer, {**meta, **info}


def train_default_model(objective: str = "clm", tokenizer_kind: str = "char",
                        corpus_chars: int = 8000, steps: int = 400,
                        d_model: int = 64, n_head: int = 4, n_layer: int = 2,
                        block_size: int = 48, lr: float = 5e-3,
                        batch_size: int = 32, seed: int = 0,
                        verbose: bool = True):
    """Build + train a small LM on the built-in synthetic corpus."""
    from ..data import tiny_corpus
    from .tokenizer import get_tokenizer
    from .model import train_lm

    text = tiny_corpus(corpus_chars, seed=seed)
    tokenizer = get_tokenizer(tokenizer_kind, text,
                              vocab_size=max(150, corpus_chars // 40))
    if objective == "mlm":
        tokenizer.ensure_mask()          # reserve <mask> *before* sizing the model
    ids = np.asarray(tokenizer.encode(text), dtype=np.int64)
    model = TinyGPT(vocab_size=tokenizer.vocab_size, d_model=d_model,
                    n_head=n_head, n_layer=n_layer, block_size=block_size,
                    seed=seed)
    mask_id = tokenizer.mask_id if objective == "mlm" else None
    hist = train_lm(model, ids, objective=objective, mask_id=mask_id,
                    steps=steps, batch_size=batch_size, lr=lr,
                    log_every=max(steps // 4, 1), verbose=verbose, seed=seed)
    return model, tokenizer, ids, hist


def ensure_model(args) -> tuple[TinyGPT, object, np.ndarray, str]:
    """Return (model, tokenizer, corpus_ids, source) from --ckpt or fresh
    training according to CLI args."""
    if getattr(args, "ckpt", None):
        model, tok, meta = load_checkpoint(args.ckpt)
        text = _corpus_from_meta(meta, args)
        ids = np.asarray(tok.encode(text), dtype=np.int64)
        return model, tok, ids, f"checkpoint:{args.ckpt}"
    model, tok, ids, _ = train_default_model(
        objective=getattr(args, "objective", "clm"),
        tokenizer_kind=getattr(args, "tokenizer", "char"),
        corpus_chars=getattr(args, "corpus", 8000),
        steps=getattr(args, "train_steps", 300),
        d_model=getattr(args, "d_model", 64),
        n_head=getattr(args, "n_head", 4),
        n_layer=getattr(args, "n_layer", 2),
        block_size=getattr(args, "block_size", 48),
        lr=getattr(args, "lr", 5e-3),
        batch_size=getattr(args, "batch", 32),
        seed=getattr(args, "seed", 0),
        verbose=not (getattr(args, "quiet", False)
                     or getattr(args, "json", False)))
    return model, tok, ids, "fresh"


def _corpus_from_meta(meta: dict, args) -> str:
    from ..data import tiny_corpus
    return tiny_corpus(meta.get("corpus_chars", 8000),
                       seed=meta.get("corpus_seed", 0))