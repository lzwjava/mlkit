"""Gradient checks: the autograd engine vs central finite differences."""

import numpy as np
import pytest

from mlkit.dl.tensor import (Tensor, cat, stack, gather_rows, cross_entropy,
                             conv2d, maxpool2d, avgpool2d, dropout,
                             gradcheck_report)
from mlkit.dl.nn import (attention_gradcheck, layernorm_gradcheck,
                         MultiHeadAttention, TransformerBlock, LayerNorm)
from mlkit.dl.train import VAE, gaussian_recon, kl_standard_normal


@pytest.mark.parametrize("name", ["linear-chain"])
def test_elementwise_chain(name):
    rng = np.random.default_rng(0)
    x = Tensor(rng.normal(size=(4, 5)), requires_grad=True, name="x")
    w = Tensor(rng.normal(size=(5, 3)), requires_grad=True, name="w")
    b = Tensor(rng.normal(size=3), requires_grad=True, name="b")
    rep = gradcheck_report(
        lambda: (((x @ w + b) * (x @ w)) * (x @ w + b).tanh()).mean(),
        [x, w, b])
    assert all(ok for _, _, _, ok in rep), rep


def test_cross_entropy_and_pow():
    rng = np.random.default_rng(0)
    logits = Tensor(rng.normal(size=(6, 4)), requires_grad=True, name="logits")
    targets = rng.integers(0, 4, size=6)
    rep = gradcheck_report(
        lambda: cross_entropy(logits, targets) + (logits ** 2).sum(axis=-1).mean(),
        [logits])
    assert all(ok for _, _, _, ok in rep), rep


def test_softmax_logsoftmax():
    rng = np.random.default_rng(0)
    z = Tensor(rng.normal(size=(5, 6)), requires_grad=True, name="z")
    rep = gradcheck_report(
        lambda: (z.softmax(-1) * z).sum() + z.log_softmax(-1).sum() * 2, [z])
    assert all(ok for _, _, _, ok in rep), rep


def test_reductions_with_axes():
    rng = np.random.default_rng(0)
    z = Tensor(rng.normal(size=(4, 5, 3)), requires_grad=True, name="z")
    rep = gradcheck_report(
        lambda: (z.mean(axis=1) ** 2).sum() + (z.sum(axis=(0, 2))
                                               * z.mean(axis=2)).sum(), [z])
    assert all(ok for _, _, _, ok in rep), rep
    rep = gradcheck_report(lambda: (z * z).sum(axis=-1).mean(), [z])
    assert all(ok for _, _, _, ok in rep), rep


def test_conv_and_pool():
    rng = np.random.default_rng(0)
    img = Tensor(rng.normal(size=(2, 3, 8, 8)), requires_grad=True, name="img")
    kw = Tensor(rng.normal(size=(4, 3, 3, 3)) * 0.3, requires_grad=True, name="kw")
    kb = Tensor(rng.normal(size=4), requires_grad=True, name="kb")

    def loss():
        h = conv2d(img, kw, kb, padding=1).relu()
        return (maxpool2d(h, 2) ** 2).mean() + \
            avgpool2d(conv2d(img, kw, kb, padding=1), 2).sum() * 0.01
    rep = gradcheck_report(loss, [img, kw, kb])
    assert all(ok for _, _, _, ok in rep), rep


def test_conv_forward_reference():
    """Hand-computed 4x4 conv with a 3x3 kernel."""
    x = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    w = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3)
    out = conv2d(Tensor(x), Tensor(w), None, stride=1, padding=0)
    ref = np.zeros((2, 2), dtype=np.float32)
    for i in range(2):
        for j in range(2):
            ref[i, j] = (x[0, 0, i:i + 3, j:j + 3] * w[0, 0]).sum()
    assert np.allclose(out.data[0, 0], ref)


def test_gather_concat_stack_index():
    rng = np.random.default_rng(0)
    emb = Tensor(rng.normal(size=(10, 4)), requires_grad=True, name="emb")
    rep = gradcheck_report(
        lambda: (gather_rows(emb, np.array([0, 3, 3, 7])) ** 2).sum(), [emb])
    assert all(ok for _, _, _, ok in rep), rep

    a = Tensor(rng.normal(size=(3, 4)), requires_grad=True, name="a")
    c = Tensor(rng.normal(size=(3, 2)), requires_grad=True, name="c")
    rep = gradcheck_report(
        lambda: (cat([a, c], axis=1) ** 3).sum(axis=1).mean(), [a, c])
    assert all(ok for _, _, _, ok in rep), rep


def test_attention_and_layernorm():
    rep = attention_gradcheck()
    assert all(ok for _, _, _, ok in rep), rep
    rep = layernorm_gradcheck()
    assert all(ok for _, _, _, ok in rep), rep


def test_multihead_attention_module():
    rng = np.random.default_rng(1)
    mha = MultiHeadAttention(8, 2, block_size=16, causal=True, bias=True)
    x = Tensor(rng.normal(size=(3, 5, 8)), requires_grad=True, name="x")
    rep = gradcheck_report(lambda: (mha(x)[0] ** 2).mean(),
                           [x] + mha.parameters())
    assert all(ok for _, _, _, ok in rep), rep


def test_transformer_block_gradcheck():
    rng = np.random.default_rng(2)
    blk = TransformerBlock(8, 2, 16, dropout=0.0, causal=True)
    x = Tensor(rng.normal(size=(2, 6, 8)), requires_grad=True, name="x")
    rep = gradcheck_report(lambda: (blk(x)[0] ** 2).mean(),
                           [x] + blk.parameters())
    assert all(ok for _, _, _, ok in rep), rep


def test_kv_cache_matches_full_forward():
    rng = np.random.default_rng(3)
    blk = TransformerBlock(8, 2, 16, dropout=0.0, causal=True).eval()
    xs = Tensor(rng.normal(size=(1, 6, 8)))
    full, _ = blk(xs, use_cache=True)
    kv = None
    for t in range(6):
        out, kv = blk(xs[:, t:t + 1, :], past_kv=kv, use_cache=True)
    assert np.allclose(full.data[0, -1], out.data[0, 0], atol=1e-5)


def test_vae_elbo_gradcheck():
    np.random.seed(0)
    rng = np.random.default_rng(0)
    vae = VAE(4, 2, 8)
    xv = Tensor(rng.normal(size=(6, 4)), requires_grad=True, name="x")
    target = Tensor(xv.data.copy())            # frozen reconstruction target

    def loss():
        np.random.seed(1)
        r2 = np.random.default_rng(1)
        recon, mu, logvar, z = vae(xv, r2)
        return gaussian_recon(recon, target) + kl_standard_normal(mu, logvar)

    rep = gradcheck_report(loss, list(vae.parameters()) + [xv])
    assert all(ok for _, _, _, ok in rep), rep