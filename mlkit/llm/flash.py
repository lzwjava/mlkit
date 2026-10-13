"""FlashAttention-style attention in NumPy.

The "flash" kernel computes attention in tiles while maintaining a running
row-max ``m`` and running row-sum ``l`` (the online softmax of the Flash
Attention / Milakov & Gimelshein paper).  This avoids materialising the full
S×S×S intermediate products and is numerically identical (up to float
rounding) to the naive implementation, which we use to verify it.

``benchmark()`` compares both kernels for speed and accuracy — the CLI calls
it with `mlkit llm flash`.
"""

from __future__ import annotations

import time

import numpy as np


def naive_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                    causal: bool = False) -> np.ndarray:
    """q,k,v: (B, H, T, d) float32 -> (B, H, T, d)."""
    scale = np.float32(1.0 / np.sqrt(q.shape[-1]))
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
    if causal:
        T = scores.shape[-1]
        mask = np.triu(np.ones((T, T), dtype=bool), 1)
        scores = np.where(mask, np.float32(-1e30), scores)
    m = scores.max(axis=-1, keepdims=True)
    p = np.exp(scores - m)
    p = p / p.sum(axis=-1, keepdims=True)
    return np.matmul(p, v)


def flash_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                    causal: bool = False, block_q: int = 16,
                    block_k: int = 16) -> np.ndarray:
    """Tiled attention with online softmax (FlashAttention algorithm 1)."""
    B, H, T, d = q.shape
    scale = np.float32(1.0 / np.sqrt(d))
    out = np.zeros((B, H, T, d), dtype=np.float32)
    NEG = np.float32(-1e30)

    for qb in range(0, T, block_q):
        qe = min(qb + block_q, T)
        Q = q[:, :, qb:qe]                                  # (B,H,bq,d)
        acc = np.zeros((B, H, qe - qb, d), dtype=np.float32)  # numerator
        l = np.zeros((B, H, qe - qb, 1), dtype=np.float32)    # denominator
        m = np.full((B, H, qe - qb, 1), NEG, dtype=np.float32)

        for kb in range(0, T, block_k):
            ke = min(kb + block_k, T)
            if causal and kb > qe - 1:
                break
            S = np.matmul(Q, np.swapaxes(k[:, :, kb:ke], -1, -2)) * scale
            if causal:
                q_idx = np.arange(qb, qe)[:, None]
                k_idx = np.arange(kb, ke)[None, :]
                S = np.where(k_idx > q_idx, NEG, S)
            # online softmax update (rescale previous accumulator)
            m_new = np.maximum(m, S.max(axis=-1, keepdims=True))
            P = np.exp(S - m_new)
            scale_old = np.exp(m - m_new)
            acc = acc * scale_old + np.matmul(P, v[:, :, kb:ke])
            l = l * scale_old + P.sum(axis=-1, keepdims=True)
            m = m_new
        out[:, :, qb:qe] = acc / np.maximum(l, np.float32(1e-12))
    return out


# --------------------------------------------------------------------------

def benchmark(B: int = 1, H: int = 8, T: int = 256, d: int = 32,
              causal: bool = True, reps: int = 20, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    shape = (B, H, T, d)
    q = rng.normal(size=shape).astype(np.float32)
    k = rng.normal(size=shape).astype(np.float32)
    v = rng.normal(size=shape).astype(np.float32)

    # correctness
    ref = naive_attention(q, k, v, causal=causal)
    ours = flash_attention(q, k, v, causal=causal)
    max_err = float(np.abs(ref - ours).max())

    def timeit(fn, reps):
        fn()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t0) / reps

    t_naive = timeit(lambda: naive_attention(q, k, v, causal), reps)
    t_flash = timeit(lambda: flash_attention(q, k, v, causal,
                                             block_q=32, block_k=32), reps)
    return {
        "shape": f"B={B} H={H} T={T} d={d}",
        "causal": causal,
        "naive_ms": t_naive * 1e3,
        "flash_ms": t_flash * 1e3,
        "max_abs_err": max_err,
        "speedup": t_naive / max(t_flash, 1e-9),
    }


if __name__ == "__main__":
    r = benchmark()
    for kk, vv in r.items():
        print(f"{kk:12s} {vv}")