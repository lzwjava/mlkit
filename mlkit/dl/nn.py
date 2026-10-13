"""Neural-network layers built on the ``mlkit.dl.tensor`` autograd engine.

Includes everything needed by the DL/LLM parts of the CLI:

* core layers: Linear, Embedding, LayerNorm, Dropout, Conv2d, pooling, Flatten
* RNN cells/layers: vanilla RNN, GRU, LSTM
* attention: scaled dot-product + Multi-Head Attention (causal mask, KV cache)
* Transformer blocks (pre-LayerNorm, GELU MLP)
"""

from __future__ import annotations

import numpy as np

from .tensor import (Tensor, cat, conv2d, dropout as dropout_op, gather_rows,
                     maxpool2d, avgpool2d)

_MODULE_TYPES = ()


# --------------------------------------------------------------------------
# module base
# --------------------------------------------------------------------------

class Module:
    training: bool = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def named_modules(self, prefix: str = ""):
        for key, val in self.__dict__.items():
            if isinstance(val, Module):
                name = f"{prefix}{key}"
                yield name, val
                yield from val.named_modules(prefix=f"{name}.")

    def named_parameters(self, prefix: str = ""):
        """Yield (name, tensor) for every parameter, frozen ones included."""
        for key, val in self.__dict__.items():
            if isinstance(val, Tensor):
                val.name = val.name or f"{prefix}{key}"
                yield f"{prefix}{key}", val
            elif isinstance(val, Module):
                yield from val.named_parameters(prefix=f"{prefix}{key}.")
            elif isinstance(val, (list, tuple)):
                for i, item in enumerate(val):
                    if isinstance(item, Module):
                        yield from item.named_parameters(
                            prefix=f"{prefix}{key}.{i}.")
                    elif isinstance(item, Tensor):
                        yield f"{prefix}{key}.{i}", item
            elif isinstance(val, dict):
                for k2, item in val.items():
                    if isinstance(item, Module):
                        yield from item.named_parameters(
                            prefix=f"{prefix}{key}.{k2}.")
                    elif isinstance(item, Tensor):
                        yield f"{prefix}{key}.{k2}", item

    def parameters(self) -> list[Tensor]:
        """Trainable parameters only (requires_grad=True)."""
        return [p for _, p in self.named_parameters() if p.requires_grad]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def num_params(self, trainable_only: bool = True) -> int:
        seen, total = set(), 0
        for _, p in self.named_parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            if trainable_only and not p.requires_grad:
                continue
            total += p.data.size
        return total

    def train(self, mode: bool = True):
        self.training = mode
        for _, m in self.named_modules():
            m.training = mode
        return self

    def eval(self):
        return self.train(False)

    def state_dict(self) -> dict:
        return {k: p.data.copy() for k, p in self.named_parameters()}

    def load_state_dict(self, state: dict, strict: bool = True):
        own = dict(self.named_parameters())
        for k, v in state.items():
            if k in own:
                if own[k].data.shape != np.asarray(v).shape:
                    raise ValueError(f"shape mismatch for {k}: "
                                     f"{own[k].data.shape} vs {np.asarray(v).shape}")
                own[k].data[...] = v
            elif strict:
                raise KeyError(f"unexpected key in state dict: {k}")
        return self

    def apply(self, fn):
        fn(self)
        for _, m in self.named_modules():
            fn(m)
        return self


def init_normal(t: Tensor, std: float):
    t.data[...] = np.random.normal(0, std, size=t.data.shape).astype(t.data.dtype)
    return t


def kaiming_uniform(t: Tensor, fan_in: int):
    bound = np.sqrt(6.0 / fan_in)                # relu-safe uniform
    t.data[...] = np.random.uniform(-bound, bound, size=t.data.shape)
    return t


# --------------------------------------------------------------------------
# core layers
# --------------------------------------------------------------------------

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 init: str = "kaiming", std: float = 0.02):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor(np.zeros((in_features, out_features)),
                             requires_grad=True)
        if init == "kaiming":
            kaiming_uniform(self.weight, in_features)
        elif init == "xavier":
            std = np.sqrt(2.0 / (in_features + out_features))
            init_normal(self.weight, std)
        else:
            init_normal(self.weight, std)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True) \
            if bias else None

    def forward(self, x):
        y = x @ self.weight
        if self.bias is not None:
            y = y + self.bias
        return y

    def __repr__(self):
        return f"Linear({self.in_features}, {self.out_features})"


class Embedding(Module):
    def __init__(self, num_embeddings: int, dim: int, std: float = 0.02):
        self.num_embeddings = num_embeddings
        self.dim = dim
        self.weight = Tensor(np.zeros((num_embeddings, dim)),
                             requires_grad=True)
        init_normal(self.weight, std)

    def forward(self, idx):
        return gather_rows(self.weight, idx)


class LayerNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        self.dim = dim
        self.eps = eps
        self.weight = Tensor(np.ones(dim), requires_grad=True)
        self.bias = Tensor(np.zeros(dim), requires_grad=True)

    def forward(self, x):
        mu = x.mean(axis=-1, keepdims=True)
        centered = x - mu
        var = (centered * centered).mean(axis=-1, keepdims=True)
        inv = (var + self.eps) ** -0.5
        return centered * inv * self.weight + self.bias


class Dropout(Module):
    def __init__(self, p: float = 0.0):
        self.p = p

    def forward(self, x):
        return dropout_op(x, self.p, self.training)


class Flatten(Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class Conv2d(Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, padding: int | None = None,
                 bias: bool = True, std: float | None = None):
        self.in_ch, self.out_ch = in_ch, out_ch
        self.kernel, self.stride = kernel, stride
        self.padding = kernel // 2 if padding is None else padding
        fan_in = in_ch * kernel * kernel
        std = std if std is not None else np.sqrt(2.0 / fan_in)
        self.weight = Tensor(np.zeros((out_ch, in_ch, kernel, kernel)),
                             requires_grad=True)
        init_normal(self.weight, std)
        self.bias = Tensor(np.zeros(out_ch), requires_grad=True) if bias else None

    def forward(self, x):
        return conv2d(x, self.weight, self.bias, stride=self.stride,
                      padding=self.padding)


class MaxPool2d(Module):
    def __init__(self, pool: int = 2, stride: int | None = None):
        self.pool = pool
        self.stride = stride

    def forward(self, x):
        return maxpool2d(x, self.pool, self.stride)


class AvgPool2d(Module):
    def __init__(self, pool: int = 2, stride: int | None = None):
        self.pool = pool
        self.stride = stride

    def forward(self, x):
        return avgpool2d(x, self.pool, self.stride)


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class GELU(Module):
    def forward(self, x):
        return x.gelu()


class ReLU(Module):
    def forward(self, x):
        return x.relu()


class Tanh(Module):
    def forward(self, x):
        return x.tanh()


class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()


# --------------------------------------------------------------------------
# recurrent layers (BPTT through the autograd graph)
# --------------------------------------------------------------------------

class RNN(Module):
    """Elman cell: h = phi(x Wxh + h Whh + b)."""

    def __init__(self, input_size: int, hidden_size: int,
                 nonlinearity: str = "tanh", bias: bool = True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        std_i, std_h = np.sqrt(1.0 / input_size), np.sqrt(1.0 / hidden_size)
        self.w_ih = Tensor(np.zeros((input_size, hidden_size)), requires_grad=True)
        self.w_hh = Tensor(np.zeros((hidden_size, hidden_size)), requires_grad=True)
        init_normal(self.w_ih, std_i)
        init_normal(self.w_hh, std_h)
        self.bias = Tensor(np.zeros(hidden_size), requires_grad=True) if bias else None

    def step(self, x_t, h):
        pre = x_t @ self.w_ih + h @ self.w_hh
        if self.bias is not None:
            pre = pre + self.bias
        return pre.tanh() if self.nonlinearity == "tanh" else pre.relu()

    def forward(self, x, h0=None):
        """x: (B, T, D) -> (Y: (B, T, H), h_last)."""
        B, T, D = x.shape
        h = h0 if h0 is not None else Tensor(np.zeros((B, self.hidden_size)))
        outs = []
        for t in range(T):
            h = self.step(x[:, t, :], h)
            outs.append(h)
        Y = outs[0].reshape(B, 1, -1) if T == 1 else cat(outs, axis=1).reshape(B, T, -1)
        return Y, h


class GRU(Module):
    """Gated recurrent unit (update z, reset r, candidate n)."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.w_ih = Tensor(np.zeros((input_size, 3 * hidden_size)),
                           requires_grad=True)
        self.w_hh = Tensor(np.zeros((hidden_size, 3 * hidden_size)),
                           requires_grad=True)
        init_normal(self.w_ih, np.sqrt(1.0 / input_size))
        init_normal(self.w_hh, np.sqrt(1.0 / hidden_size))
        self.bias_ih = Tensor(np.zeros(3 * hidden_size), requires_grad=True) \
            if bias else None
        self.bias_hh = Tensor(np.zeros(3 * hidden_size), requires_grad=True) \
            if bias else None

    def step(self, x_t, h):
        gi = x_t @ self.w_ih
        gh = h @ self.w_hh
        if self.bias_ih is not None:
            gi = gi + self.bias_ih
            gh = gh + self.bias_hh
        H = self.hidden_size
        ir, iz, inn = gi[:, :H], gi[:, H:2 * H], gi[:, 2 * H:]
        hr, hz, hn = gh[:, :H], gh[:, H:2 * H], gh[:, 2 * H:]
        r = (ir + hr).sigmoid()
        z = (iz + hz).sigmoid()
        n = (inn + r * hn).tanh()
        return (1 - z) * n + z * h

    def forward(self, x, h0=None):
        B, T, D = x.shape
        h = h0 if h0 is not None else Tensor(np.zeros((B, self.hidden_size)))
        outs = []
        for t in range(T):
            h = self.step(x[:, t, :], h)
            outs.append(h)
        Y = outs[0].reshape(B, 1, -1) if T == 1 else cat(outs, axis=1).reshape(B, T, -1)
        return Y, h


class LSTM(Module):
    """Long Short-Term Memory (input i, forget f, output o, candidate g)."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.w_ih = Tensor(np.zeros((input_size, 4 * hidden_size)),
                           requires_grad=True)
        self.w_hh = Tensor(np.zeros((hidden_size, 4 * hidden_size)),
                           requires_grad=True)
        init_normal(self.w_ih, np.sqrt(1.0 / input_size))
        init_normal(self.w_hh, np.sqrt(1.0 / hidden_size))
        self.bias_ih = Tensor(np.zeros(4 * hidden_size), requires_grad=True) \
            if bias else None
        self.bias_hh = Tensor(np.zeros(4 * hidden_size), requires_grad=True) \
            if bias else None

    def step(self, x_t, state):
        h, c = state
        gi = x_t @ self.w_ih
        gh = h @ self.w_hh
        if self.bias_ih is not None:
            gi = gi + self.bias_ih
            gh = gh + self.bias_hh
        gates = gi + gh
        H = self.hidden_size
        i = gates[:, 0:H].sigmoid()
        f = gates[:, H:2 * H].sigmoid()
        o = gates[:, 2 * H:3 * H].sigmoid()
        g = gates[:, 3 * H:].tanh()
        c_new = f * c + i * g
        h_new = o * c_new.tanh()
        return h_new, c_new

    def forward(self, x, state=None):
        """x: (B, T, D) -> (Y: (B, T, H), (h, c))."""
        B, T, D = x.shape
        if state is None:
            z = Tensor(np.zeros((B, self.hidden_size)))
            state = (z, z)
        hs, cs = [], []
        h, c = state
        for t in range(T):
            h, c = self.step(x[:, t, :], (h, c))
            hs.append(h)
            cs.append(c)
        Y = hs[0].reshape(B, 1, -1) if T == 1 else cat(hs, axis=1).reshape(B, T, -1)
        return Y, (h, c)


# --------------------------------------------------------------------------
# attention & transformer
# --------------------------------------------------------------------------

NEG = -1e4          # additive mask value (softmax -> ~0)


def scaled_dot_product_attention(q, k, v, mask: np.ndarray | None = None,
                                 training: bool = False, p_drop: float = 0.0,
                                 rng=None):
    """q,k,v: (B, H, T, Dh). mask: broadcastable additive mask (T, Tk)."""
    scale = np.sqrt(q.shape[-1]).astype(q.data.dtype)
    scores = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / scale)
    if mask is not None:
        scores = scores + Tensor(mask)
    att = scores.softmax(axis=-1)
    att = dropout_op(att, p_drop, training, rng)
    return att @ v, att


class MultiHeadAttention(Module):
    def __init__(self, d_model: int, n_head: int, block_size: int,
                 dropout: float = 0.0, causal: bool = True, bias: bool = False,
                 kv_dim: int | None = None):
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.block_size = block_size
        self.dropout = dropout
        self.causal = causal
        kv_dim = kv_dim or d_model
        self.c_attn = Linear(kv_dim, 3 * d_model, bias=bias)
        self.c_proj = Linear(d_model, d_model, bias=bias)
        self.resid_dropout = Dropout(dropout)

    def _mask(self, Tq: int, Tk: int) -> np.ndarray | None:
        if not self.causal:
            return None
        # query i corresponds to absolute position Tk - Tq + i
        q_pos = np.arange(Tk - Tq, Tk)[:, None]
        k_pos = np.arange(Tk)[None, :]
        return np.where(k_pos > q_pos, NEG, 0.0).astype(np.float32)

    def forward(self, x, past_kv=None, use_cache: bool = False):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q = qkv[:, :, :C]
        k = qkv[:, :, C:2 * C]
        v = qkv[:, :, 2 * C:]
        nh, hd = self.n_head, self.head_dim

        def split(t):
            return t.reshape(B, T, nh, hd).transpose(0, 2, 1, 3)

        q, k, v = split(q), split(k), split(v)
        if past_kv is not None:
            pk, pv = past_kv
            k = cat([pk, k], axis=2)
            v = cat([pv, v], axis=2)
        Tk = k.shape[2]
        mask = self._mask(T, Tk)
        y, att = scaled_dot_product_attention(
            q, k, v, mask, self.training, self.dropout)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        out = self.resid_dropout(self.c_proj(y))
        return (out, (k, v)) if use_cache else (out, None)


class TransformerBlock(Module):
    """Pre-LayerNorm Transformer block: x + MHA(LN(x)); x + MLP(LN(x))."""

    def __init__(self, d_model: int, n_head: int, block_size: int,
                 mlp_ratio: int = 4, dropout: float = 0.0, causal: bool = True):
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_head, block_size,
                                       dropout=dropout, causal=causal)
        self.ln2 = LayerNorm(d_model)
        hidden = d_model * mlp_ratio
        self.mlp = Sequential(Linear(d_model, hidden, bias=False), GELU(),
                              Linear(hidden, d_model, bias=False),
                              Dropout(dropout))

    def forward(self, x, past_kv=None, use_cache: bool = False):
        a, kv = self.attn(self.ln1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, kv


class MLP(Module):
    """Simple feed-forward network for tabular/vector inputs."""

    def __init__(self, in_dim: int, hidden: list[int], out_dim: int,
                 activation: str = "relu", dropout: float = 0.0,
                 batch_norm: bool = False):
        acts = {"relu": ReLU, "tanh": Tanh, "gelu": GELU,
                "sigmoid": Sigmoid}[activation]
        layers: list[Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(Linear(prev, h))
            if batch_norm:
                layers.append(BatchNorm1d(h))
            layers.append(acts())
            if dropout > 0:
                layers.append(Dropout(dropout))
            prev = h
        layers.append(Linear(prev, out_dim))
        self.net = Sequential(*layers)
        self.activation_name = activation

    def forward(self, x):
        return self.net(x)


class BatchNorm1d(Module):
    """Batch normalization (running stats for eval mode)."""

    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1):
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Tensor(np.ones(dim), requires_grad=True)
        self.bias = Tensor(np.zeros(dim), requires_grad=True)
        self.running_mean = np.zeros(dim)
        self.running_var = np.ones(dim)

    def forward(self, x):
        if self.training:
            mu = x.mean(axis=0, keepdims=True)
            var = ((x - mu) * (x - mu)).mean(axis=0, keepdims=True)
            self.running_mean = (1 - self.momentum) * self.running_mean \
                + self.momentum * mu.data.reshape(-1)
            self.running_var = (1 - self.momentum) * self.running_var \
                + self.momentum * var.data.reshape(-1)
            inv = (var + self.eps) ** -0.5
            return (x - mu) * inv * self.weight + self.bias
        mu = Tensor(self.running_mean.reshape(1, -1))
        var = Tensor(self.running_var.reshape(1, -1))
        inv = (var + self.eps) ** -0.5
        return (x - mu) * inv * self.weight + self.bias


# --------------------------------------------------------------------------
# attention gradcheck helper (also used by `mlkit dl gradcheck`)
# --------------------------------------------------------------------------

def attention_gradcheck(seed: int = 0, tol: float | None = None) -> list[tuple]:
    from .tensor import gradcheck_report
    rng = np.random.default_rng(seed)
    B, H, T, Dh = 2, 2, 4, 3
    q = Tensor(rng.normal(size=(B, H, T, Dh)).astype(np.float32),
               requires_grad=True, name="q")
    k = Tensor(rng.normal(size=(B, H, T, Dh)).astype(np.float32),
               requires_grad=True, name="k")
    v = Tensor(rng.normal(size=(B, H, T, Dh)).astype(np.float32),
               requires_grad=True, name="v")
    mask = np.where(np.triu(np.ones((T, T)), 1) > 0, NEG, 0.0).astype(np.float32)

    def loss_fn():
        y, _ = scaled_dot_product_attention(q, k, v, mask)
        return (y * y).mean()

    return gradcheck_report(loss_fn, [q, k, v], tol=tol)


def layernorm_gradcheck(seed: int = 0, tol: float | None = None) -> list[tuple]:
    from .tensor import gradcheck_report
    rng = np.random.default_rng(seed)
    ln = LayerNorm(6)
    x = Tensor(rng.normal(size=(5, 6)).astype(np.float32) * 3,
               requires_grad=True, name="x")

    def loss_fn():
        return (ln(x) ** 2).sum(axis=-1).mean()

    return gradcheck_report(loss_fn, list(ln.parameters()) + [x], tol=tol)