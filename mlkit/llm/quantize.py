"""Weight quantization: symmetric INT8 (per-channel) and INT4 (group-wise).

Quantising every Linear weight from fp32 to int8/int4 shrinks memory 4-8x;
we evaluate the exact perplexity delta of a quantise→dequantise round-trip
(the same "fake-quant" simulation used by GPTQ/AWQ style tooling).
"""

from __future__ import annotations

import numpy as np

from ..dl import nn
from .model import TinyGPT


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------

def quantize_int8_per_channel(W: np.ndarray, axis: int = 1):
    """Symmetric per-output-channel INT8: q in [-127, 127], scale per row."""
    W = np.asarray(W, dtype=np.float32)
    if axis == 1:
        scale = np.abs(W).max(axis=1, keepdims=True) / 127.0
    else:
        scale = np.abs(W).max(axis=0, keepdims=True) / 127.0
    scale = np.maximum(scale, 1e-12)
    q = np.clip(np.round(W / scale), -127, 127).astype(np.int8)
    return q, scale.astype(np.float32)


def quantize_int4_groupwise(W: np.ndarray, group: int = 64):
    """Symmetric group-wise INT4: q in [-7, 7] along the last dimension."""
    W = np.asarray(W, dtype=np.float32)
    orig_shape = W.shape
    d = orig_shape[-1]
    pad = (-d) % group
    Wf = np.concatenate([W.reshape(-1, d),
                         np.zeros((W.shape[0], pad), np.float32)], axis=1) \
        if pad else W.reshape(-1, d)
    n_groups = Wf.shape[1] // group
    blocks = Wf.reshape(Wf.shape[0], n_groups, group)
    scale = np.abs(blocks).max(axis=-1, keepdims=True) / 7.0
    scale = np.maximum(scale, 1e-12)
    q = np.clip(np.round(blocks / scale), -8, 7).astype(np.int8)
    deq = (q * scale).reshape(Wf.shape[0], -1)[:, :d].reshape(orig_shape)
    return q, scale.astype(np.float32), deq


def dequantize_int8(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (q.astype(np.float32) * scale)


# --------------------------------------------------------------------------
# model-level quantisation
# --------------------------------------------------------------------------

def iter_linears(model) -> list[tuple[str, nn.Linear]]:
    out = []
    for name, module in _walk(model):
        if isinstance(module, nn.Linear):
            out.append((name, module))
    return out


def _walk(module, prefix: str = ""):
    for key, val in module.__dict__.items():
        if isinstance(val, nn.Module):
            name = f"{prefix}{key}"
            yield name, val
            yield from _walk(val, prefix=f"{name}.")
        elif isinstance(val, (list, tuple)):
            for i, item in enumerate(val):
                if isinstance(item, nn.Module):
                    name = f"{prefix}{key}.{i}"
                    yield name, item
                    yield from _walk(item, prefix=f"{name}.")


def evaluate_ppl(model: TinyGPT, tokens, block_size: int | None = None,
                 stride: int | None = None) -> float:
    """Sliding-window causal-LM perplexity on a token array."""
    from ..dl.tensor import cross_entropy
    tokens = np.asarray(tokens, dtype=np.int64)
    block = block_size or model.block_size
    stride = stride or block
    model.eval()
    losses = []
    for s in range(0, max(len(tokens) - block - 1, 1), stride):
        chunk = tokens[s:s + block + 1]
        if len(chunk) < 8:
            continue
        logits, loss, _ = model(chunk[:-1][None, :], targets=chunk[1:][None, :])
        losses.append(float(loss.item()))
    if not losses:
        return float("nan")
    return float(np.exp(min(float(np.mean(losses)), 20.0)))


def quantize_model(model: TinyGPT, bits: str = "int8", group: int = 64,
                   apply: bool = True) -> dict:
    """Quantise all Linear weights; returns a report and (optionally) applies
    the dequantised weights in-place so eval runs on quantised precision.

    Every Linear also gets ``q``/``scale`` stored, so the quantised tensors
    are what actually occupies memory (``quantized_bytes``).
    """
    bits = bits.lower()
    report = {"bits": bits, "layers": [], "modules": 0}
    fp_bytes = 0
    q_bytes = 0

    for name, lin in iter_linears(model):
        W = lin.weight.data
        fp_bytes += W.nbytes
        if bits in ("int8", "8"):
            q, scale = quantize_int8_per_channel(W, axis=1)
            deq = dequantize_int8(q, scale)
            nbytes = q.nbytes + scale.nbytes
        elif bits in ("int4", "4"):
            q, scale, deq = quantize_int4_groupwise(W, group=group)
            nbytes = (q.size * 4) // 8 + scale.nbytes     # 4-bit packed
        else:
            raise ValueError("bits must be 'int8' or 'int4'")
        err = float(np.abs(deq - W).max())
        report["layers"].append({"name": name, "shape": list(W.shape),
                                 "max_abs_err": err,
                                 "mean_abs_err": float(np.abs(deq - W).mean())})
        report["modules"] += 1
        q_bytes += nbytes
        lin.quant_state = {"q": q, "scale": scale, "bits": bits}
        if apply:
            lin.weight.data[...] = deq.astype(np.float32)

    report["fp32_bytes"] = fp_bytes
    report["quantized_bytes"] = q_bytes
    report["compression"] = fp_bytes / max(q_bytes, 1)
    report["max_abs_err"] = max((l["max_abs_err"] for l in report["layers"]),
                                default=0.0)
    return report


def dequantize_model(model: TinyGPT, originals: dict):
    """Restore original fp32 weights (kept by `snapshot_weights`)."""
    for name, lin in iter_linears(model):
        if name in originals:
            lin.weight.data[...] = originals[name]


def snapshot_weights(model: TinyGPT) -> dict:
    return {name: lin.weight.data.copy() for name, lin in iter_linears(model)}