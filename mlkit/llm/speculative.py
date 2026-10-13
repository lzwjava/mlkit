"""Speculative decoding (Leviathan et al. 2023 / Chen et al. 2023).

A small, cheap **draft** model proposes ``k`` tokens autoregressively; the
large **target** model verifies them in a *single* parallel forward pass and
each draft token is accepted with probability ``min(1, p_target/p_draft)``.
On rejection we resample from ``normalise(max(0, p_target - p_draft))``.

The resulting token stream is provably distributed *exactly* like sampling
from the target model alone, but uses far fewer target-model forwards.
"""

from __future__ import annotations

import time

import numpy as np

from .model import TinyGPT
from .generate import _logsumexp


def _step_probs(model: TinyGPT, ctx: np.ndarray, temperature: float) -> np.ndarray:
    """Softmax over the last position's logits: (B, V)."""
    logits, _, _ = model(ctx)
    l = logits.data[:, -1]
    if temperature != 1.0:
        l = l / max(temperature, 1e-6)
    l = l - l.max(axis=-1, keepdims=True)
    e = np.exp(l)
    return e / e.sum(axis=-1, keepdims=True)


def speculative_greedy_tokens(target: TinyGPT, draft: TinyGPT, prompt_ids,
                              n_tokens: int, k: int = 4,
                              temperature: float = 1.0, seed: int = 0,
                              cache_target: bool = True) -> dict:
    """Generate `n_tokens` tokens (sampled at given temperature) with
    speculative decoding; returns ids + statistics."""
    rng = np.random.default_rng(seed)
    target.eval()
    draft.eval()
    ids = [int(i) for i in np.atleast_1d(np.asarray(prompt_ids)).ravel()]
    accepted_total = 0
    proposed_total = 0
    target_forwards = 0
    draft_forwards = 0

    def sample(p: np.ndarray) -> int:
        return int(rng.choice(p.shape[-1], p=p[0]))

    while len(ids) - len(np.atleast_1d(np.asarray(prompt_ids)).ravel()) < n_tokens:
        base_len = len(ids)
        # 1) draft proposes up to k tokens (its own tiny forward passes)
        ctx_d = list(ids)
        proposed = []
        for _ in range(k):
            if len(ctx_d) >= target.block_size:
                break
            p_d = _step_probs(draft, np.array([ctx_d[-target.block_size:]],
                                              dtype=np.int64), temperature)
            draft_forwards += 1
            tok = sample(p_d)
            proposed.append((tok, p_d))
            ctx_d.append(tok)
        if not proposed:
            break
        proposed_total += len(proposed)

        # 2) target verifies all proposals in ONE forward pass
        verify_ctx = ids + [t for t, _ in proposed]
        verify_ctx = verify_ctx[-target.block_size:]
        offset = len(verify_ctx) - len(proposed)
        logits, _, _ = target(np.array([verify_ctx], dtype=np.int64))
        target_forwards += 1
        raw = logits.data[0]
        if temperature != 1.0:
            raw = raw / max(temperature, 1e-6)
        # softmax per position
        raw = raw - raw.max(axis=-1, keepdims=True)
        probs_t = np.exp(raw)
        probs_t /= probs_t.sum(axis=-1, keepdims=True)

        # 3) rejection sampling
        # logits[j] predicts verify_ctx[j+1]; proposed[i] lives at index
        # offset+i, so its logits are at offset+i-1
        rejected = False
        for i, (tok, p_d) in enumerate(proposed):
            pos = offset + i - 1
            p_t = probs_t[pos][None]
            ratio = float(p_t[0, tok] / max(p_d[0, tok], 1e-12))
            if rng.random() < min(1.0, ratio):
                ids.append(tok)
                accepted_total += 1
            else:
                # residual distribution: normalise(max(0, p_t - p_d))
                resid = np.maximum(p_t - p_d, 0.0)
                s = resid.sum()
                if s <= 1e-12:
                    resid = p_t
                    s = resid.sum()
                resid = (resid / s)[0]
                ids.append(int(rng.choice(len(resid), p=resid)))
                rejected = True
                break
        if not rejected and len(proposed) == k:
            # bonus token: sample from target at the position after the last
            # proposal (logits index = offset + k - 1)
            pos = offset + len(proposed) - 1
            if 0 <= pos < probs_t.shape[0]:
                ids.append(int(rng.choice(probs_t.shape[-1], p=probs_t[pos])))
                accepted_total += 1
                proposed_total += 1

        _ = base_len  # (kept for readability)

    limit = len(np.atleast_1d(np.asarray(prompt_ids)).ravel()) + n_tokens
    if len(ids) > limit:                     # drop bonus-token overshoot
        del ids[limit:]
    generated = len(ids) - len(np.atleast_1d(np.asarray(prompt_ids)).ravel())
    return {"ids": ids, "generated": generated,
            "acceptance_rate": accepted_total / max(proposed_total, 1),
            "target_forwards": target_forwards,
            "draft_forwards": draft_forwards,
            "proposed": proposed_total, "accepted": accepted_total}


def baseline_sample(target: TinyGPT, prompt_ids, n_tokens: int,
                    temperature: float = 1.0, seed: int = 0,
                    cache_target: bool = True) -> dict:
    """Plain target-model sampling (one forward per token, with KV cache)."""
    rng = np.random.default_rng(seed)
    target.eval()
    ids = [int(i) for i in np.atleast_1d(np.asarray(prompt_ids)).ravel()]
    n_prompt = len(ids)
    ctx = ids[-target.block_size:]
    logits, _, kv = (target(np.array([ctx], dtype=np.int64), use_cache=True)
                     if cache_target else (None, None, None))
    forwards = 0
    while len(ids) - n_prompt < n_tokens:
        if cache_target and kv is not None:
            l = logits.data[0, -1]
        else:
            ctx = ids[-target.block_size:]
            logits, _, kv = target(np.array([ctx], dtype=np.int64),
                                   use_cache=cache_target)
            l = logits.data[0, -1]
            if cache_target:
                pass
        forwards += 1
        if temperature != 1.0:
            l = l / max(temperature, 1e-6)
        l = l - l.max()
        p = np.exp(l)
        p /= p.sum()
        tok = int(rng.choice(len(p), p=p))
        ids.append(tok)
        if cache_target and kv is not None and len(ids) <= target.block_size:
            logits, _, kv = target(np.array([[tok]]), past_kv=kv,
                                   use_cache=True)
    return {"ids": ids, "generated": len(ids) - n_prompt,
            "target_forwards": forwards}


def benchmark(target: TinyGPT, draft: TinyGPT, prompt_ids, n_tokens: int = 32,
              k: int = 4, temperature: float = 1.0, seed: int = 0) -> dict:
    t0 = time.perf_counter()
    spec = speculative_greedy_tokens(target, draft, prompt_ids, n_tokens,
                                     k=k, temperature=temperature, seed=seed)
    t_spec = time.perf_counter() - t0
    t0 = time.perf_counter()
    base = baseline_sample(target, prompt_ids, n_tokens,
                           temperature=temperature, seed=seed)
    t_base = time.perf_counter() - t0
    return {
        "tokens": n_tokens, "k": k,
        "baseline_s": t_base, "speculative_s": t_spec,
        "speedup": t_base / max(t_spec, 1e-9),
        "acceptance_rate": spec["acceptance_rate"],
        "target_forwards_baseline": base["target_forwards"],
        "target_forwards_speculative": spec["target_forwards"],
        "draft_forwards": spec["draft_forwards"],
        "spec_ids": spec["ids"], "base_ids": base["ids"],
    }