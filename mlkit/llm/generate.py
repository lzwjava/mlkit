"""Text generation / decoding strategies for autoregressive LMs.

* greedy decoding (argmax)
* temperature scaling
* top-k sampling
* nucleus / top-p sampling
* beam search
* speculative decoding helpers live in ``speculative.py``

All samplers support incremental decoding with a KV cache (``use_cache``).
"""

from __future__ import annotations

import numpy as np

from .model import TinyGPT


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        return logits
    return logits / max(temperature, 1e-6)


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    kth = np.partition(logits, -k, axis=-1)[..., -k]
    return np.where(logits < kth[..., None], -np.inf, logits)


def top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    """Nucleus sampling: keep the smallest set with cumulative prob >= p."""
    if p >= 1.0:
        return logits
    sorted_idx = np.argsort(logits, axis=-1)[..., ::-1]
    sorted_logits = np.take_along_axis(logits, sorted_idx, axis=-1)
    probs = _softmax(sorted_logits)
    cum = np.cumsum(probs, axis=-1)
    # keep first token above threshold + everything before it
    cutoff = cum > p
    cutoff[..., 1:] = cutoff[..., :-1].copy()
    cutoff[..., 0] = False
    remove = np.zeros_like(cutoff)
    np.put_along_axis(remove, sorted_idx, cutoff, axis=-1)
    return np.where(remove, -np.inf, logits)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def sample_from_logits(logits: np.ndarray, rng, temperature: float = 1.0,
                       top_k: int | None = None, top_p: float | None = None
                       ) -> np.ndarray:
    """(B, V) logits -> (B,) sampled ids."""
    if temperature <= 0:                       # greedy
        return logits.argmax(axis=-1)
    l = apply_temperature(logits, temperature)
    if top_k:
        l = top_k_filter(l, top_k)
    if top_p is not None and top_p < 1.0:
        l = top_p_filter(l, top_p)
    probs = _softmax(l)
    # guard against all -inf rows
    probs = np.where(np.isfinite(probs).all(axis=-1, keepdims=True), probs,
                     _softmax(logits))
    out = np.empty(probs.shape[0], dtype=np.int64)
    for i in range(probs.shape[0]):
        out[i] = rng.choice(probs.shape[1], p=probs[i])
    return out


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------

def generate(model: TinyGPT, prompt_ids, max_new_tokens: int = 40,
             temperature: float = 1.0, top_k: int | None = None,
             top_p: float | None = None, beam_size: int = 1,
             seed: int | None = 0, eos_id: int | None = None,
             use_cache: bool = True) -> list[int]:
    """Generate tokens from a prompt with the chosen decoding strategy."""
    ids = [int(i) for i in np.atleast_1d(np.asarray(prompt_ids)).ravel()]
    if beam_size and beam_size > 1:
        return beam_search(model, ids, max_new_tokens, beam_size,
                           temperature=temperature, top_k=top_k, top_p=top_p,
                           eos_id=eos_id)
    rng = np.random.default_rng(seed)
    model.eval()

    if not use_cache:
        # simple re-forwarding path (slow, useful as a KV-cache baseline)
        for _ in range(max_new_tokens):
            ctx = _truncate(ids, model.block_size)
            logits, _, _ = model(np.array([ctx], dtype=np.int64))
            nxt = sample_from_logits(logits.data[0, -1:], rng, temperature,
                                     top_k, top_p)[0]
            ids.append(int(nxt))
            if eos_id is not None and nxt == eos_id:
                break
        return ids

    # KV-cache incremental decoding
    ctx = _truncate(ids, model.block_size)
    logits, _, kv = model(np.array([ctx], dtype=np.int64), use_cache=True)
    for _ in range(max_new_tokens):
        nxt = sample_from_logits(logits.data[:, -1, :], rng, temperature,
                                 top_k, top_p)[0]
        ids.append(int(nxt))
        if eos_id is not None and nxt == eos_id:
            break
        if len(ids) >= model.block_size:
            # context window full: restart from the tail (drops cache)
            ctx = ids[-model.block_size:]
            logits, _, kv = model(np.array([ctx], dtype=np.int64),
                                  use_cache=True)
        else:
            logits, _, kv = model(np.array([[int(nxt)]]), past_kv=kv,
                                  use_cache=True)
    return ids


def _truncate(ids, block: int) -> list[int]:
    return ids[-block:] if len(ids) > block else ids


# --------------------------------------------------------------------------
# beam search
# --------------------------------------------------------------------------

def beam_search(model: TinyGPT, prompt_ids, max_new_tokens: int,
                beam_size: int = 4, temperature: float = 1.0,
                top_k: int | None = None, top_p: float | None = None,
                eos_id: int | None = None) -> list[int]:
    """Standard beam search over log-probabilities (length-normalised)."""
    model.eval()
    beams = [[int(i) for i in prompt_ids]]
    scores = np.zeros(1)
    finished: list[tuple[float, list[int]]] = []

    for _ in range(max_new_tokens):
        cand_seqs, cand_scores = [], []
        for seq, sc in zip(beams, scores):
            ctx = _truncate(seq, model.block_size)
            logits, _, _ = model(np.array([ctx], dtype=np.int64))
            l = logits.data[0, -1]
            if temperature > 0 and temperature != 1.0:
                l = l / temperature
            if top_k:
                l = top_k_filter(l[None], top_k)[0]
            if top_p is not None and top_p < 1.0:
                l = top_p_filter(l[None], top_p)[0]
            logp = l - _logsumexp(l)
            finite = np.where(np.isfinite(logp), logp, -1e30)
            k = min(beam_size, finite.size)
            top_idx = np.argpartition(finite, -k)[-k:]
            for j in top_idx:
                new_seq = seq + [int(j)]
                new_score = sc + float(finite[j])
                if eos_id is not None and j == eos_id:
                    finished.append((new_score / max(len(new_seq), 1), new_seq))
                else:
                    cand_seqs.append(new_seq)
                    cand_scores.append(new_score)
        if not cand_seqs:
            break
        order = np.argsort(cand_scores)[::-1][:beam_size]
        beams = [cand_seqs[i] for i in order]
        scores = np.array([cand_scores[i] for i in order])

    if finished:
        best = max(finished, key=lambda t: t[0])
        return best[1]
    # length-normalised best beam
    norms = np.array([len(b) for b in beams], dtype=float)
    best = beams[int(np.argmax(scores / norms))]
    return best


def _logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    if not np.isfinite(m):
        return -np.inf
    return m + float(np.log(np.exp(x - m).sum()))


# --------------------------------------------------------------------------
# KV-cache benchmark
# --------------------------------------------------------------------------

def benchmark_cache(model: TinyGPT, prompt_ids, n_tokens: int = 32,
                    seed: int = 0) -> dict:
    """Compare cached vs non-cached decoding speed (same greedy path)."""
    import time
    out = {}
    for use_cache, name in ((False, "no_cache"), (True, "kv_cache")):
        t0 = time.perf_counter()
        ids = generate(model, prompt_ids, max_new_tokens=n_tokens,
                       temperature=0.0, use_cache=use_cache, seed=seed)
        out[name] = time.perf_counter() - t0
        out[f"{name}_ids"] = ids
    out["speedup"] = out["no_cache"] / max(out["kv_cache"], 1e-9)
    out["identical"] = out["no_cache_ids"] == out["kv_cache_ids"]
    return out