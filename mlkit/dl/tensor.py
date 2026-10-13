"""A tiny reverse-mode automatic differentiation engine over NumPy arrays.

This is the foundation for every deep-learning model in ``mlkit``: MLPs,
CNNs, RNNs/LSTMs/GRUs, attention/Transformers, GANs, VAEs and diffusion
models are all built from the ops defined here.

Design notes
------------
* ``Tensor.data`` is a float32 ``np.ndarray``.
* Each op records its inputs (``_prev``) and a closure (``_backward``) that
  scatters gradients into ``.grad`` — like PyTorch, but fully vectorised.
* The output node is created *first* and captured by the closure, so
  backward closures always read ``node.grad`` (the propagated gradient).
* Graph construction is skipped automatically when no input requires
  gradients, so inference is allocation-light.
"""

from __future__ import annotations

import numpy as np

DTYPE = np.float32


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum `grad` back down to `shape` (inverse of numpy broadcasting)."""
    if grad.shape == shape:
        return grad
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


def _noop():
    pass


class Tensor:
    __slots__ = ("data", "grad", "requires_grad", "_prev", "_backward", "name")

    def __init__(self, data, requires_grad: bool = False, name: str = ""):
        arr = np.asarray(data)
        if arr.dtype != DTYPE:
            arr = arr.astype(DTYPE)
        self.data: np.ndarray = arr
        self.grad: np.ndarray | None = None
        self.requires_grad = requires_grad
        self._prev: tuple = ()
        self._backward = _noop
        self.name = name

    # ---- basics --------------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def item(self) -> float:
        return float(self.data.reshape(-1)[0])

    def numpy(self) -> np.ndarray:
        return self.data

    def detach(self) -> "Tensor":
        return Tensor(self.data.copy())

    def zero_grad(self):
        self.grad = None

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        name = f" name='{self.name}'" if self.name else ""
        return (f"Tensor(shape={self.shape}, requires_grad={self.requires_grad}"
                f"{name})")

    # ---- graph helpers -------------------------------------------------
    def _acc(self, g: np.ndarray):
        if g is None:
            return
        g = np.asarray(g, dtype=DTYPE)
        if self.grad is None:
            self.grad = g.copy()
        else:
            self.grad += g

    @staticmethod
    def _node(data, parents: tuple) -> "Tensor":
        """Create an output tensor (graph wired only when needed)."""
        arr = np.asarray(data)
        if arr.dtype != DTYPE:
            arr = arr.astype(DTYPE)
        out = Tensor.__new__(Tensor)
        out.data = arr
        out.grad = None
        out.name = ""
        out.requires_grad = any(p.requires_grad for p in parents)
        out._prev = parents if out.requires_grad else ()
        out._backward = _noop
        return out

    @staticmethod
    def _binary(a, b, data, bw_fwd):
        """Shared binary-op scaffolding: bw_fwd(self_data, other_data, grad)."""
        a, b = _wrap(a), _wrap(b)
        node = Tensor._node(data, (a, b))
        if node.requires_grad:
            da, db = a.data, b.data

            def bw():
                ga, gb = bw_fwd(da, db, node.grad)
                if a.requires_grad:
                    a._acc(_unbroadcast(ga, da.shape))
                if b.requires_grad:
                    b._acc(_unbroadcast(gb, db.shape))
            node._backward = bw
        return node

    @staticmethod
    def _unary(a, data, bw_fwd):
        a = _wrap(a)
        node = Tensor._node(data, (a,))
        if node.requires_grad:
            src = a.data

            def bw():
                if a.requires_grad:
                    a._acc(bw_fwd(src, node.grad, node.data))
            node._backward = bw
        return node

    def backward(self):
        """Reverse-mode backprop from this (usually scalar) tensor."""
        topo = _topological_order(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            if v.grad is not None:
                v._backward()

    # ---- elementwise arithmetic ---------------------------------------
    def __add__(self, other):
        return Tensor._binary(self, other, self.data + _wrap(other).data,
                              lambda a, b, g: (g, g))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return Tensor._binary(self, other, self.data - _wrap(other).data,
                              lambda a, b, g: (g, -g))

    def __rsub__(self, other):
        return _wrap(other).__sub__(self)

    def __mul__(self, other):
        return Tensor._binary(self, other, self.data * _wrap(other).data,
                              lambda a, b, g: (g * b, g * a))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return Tensor._binary(self, other, self.data / _wrap(other).data,
                              lambda a, b, g: (g / b, -g * a / b ** 2))

    def __rtruediv__(self, other):
        return _wrap(other).__truediv__(self)

    def __pow__(self, p: float):
        return Tensor._unary(self, np.power(self.data, p),
                             lambda a, g, o: g * p * np.power(a, p - 1))

    def __neg__(self):
        return Tensor._unary(self, -self.data, lambda a, g, o: -g)

    # ---- shape ops -----------------------------------------------------
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        shape = tuple(int(s) for s in shape)
        node = Tensor._node(self.data.reshape(shape), (self,))
        if node.requires_grad:
            orig = self.data.shape

            def bw():
                self._acc(node.grad.reshape(orig))
            node._backward = bw
        return node

    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        elif not axes:
            axes = tuple(reversed(range(self.data.ndim)))
        axes = tuple(axes)
        node = Tensor._node(self.data.transpose(axes), (self,))
        if node.requires_grad:
            inverse = np.argsort(axes)

            def bw():
                self._acc(node.grad.transpose(inverse))
            node._backward = bw
        return node

    def _reduce(self, data, axis, keepdims, fn):
        """fn(grad_broadcast) -> gradient for the input tensor."""
        node = Tensor._node(data, (self,))
        if node.requires_grad:
            shape = self.data.shape
            ndim = len(shape)
            if axis is None:
                resh = shape if keepdims else ()
            else:
                ax = tuple(a if a >= 0 else a + ndim
                           for a in (axis if isinstance(axis, (tuple, list))
                                     else (axis,)))
                resh = tuple(1 if i in ax else s for i, s in enumerate(shape))

            def bw():
                g = node.grad
                if resh != g.shape:
                    g = g.reshape(resh)
                self._acc(fn(np.broadcast_to(g, shape).astype(DTYPE)))
            node._backward = bw
        return node

    def sum(self, axis=None, keepdims: bool = False):
        return self._reduce(self.data.sum(axis=axis, keepdims=keepdims),
                            axis, keepdims, lambda g: g)

    def mean(self, axis=None, keepdims: bool = False):
        node = Tensor._node(self.data.mean(axis=axis, keepdims=keepdims),
                            (self,))
        if node.requires_grad:
            shape = self.data.shape
            ndim = len(shape)
            n = self.data.size // max(node.data.size, 1)
            if axis is None:
                resh = shape if keepdims else ()
            else:
                ax = tuple(a if a >= 0 else a + ndim
                           for a in (axis if isinstance(axis, (tuple, list))
                                     else (axis,)))
                resh = tuple(1 if i in ax else s for i, s in enumerate(shape))

            def bw():
                g = node.grad
                if resh != g.shape:
                    g = g.reshape(resh)
                self._acc(np.broadcast_to(g, shape).astype(DTYPE) / n)
            node._backward = bw
        return node

    def max(self, axis=None, keepdims: bool = False):
        """Hard max (non-differentiable path: gradient to argmax only)."""
        node = Tensor._node(self.data.max(axis=axis, keepdims=keepdims),
                            (self,))
        if node.requires_grad:
            src = self.data

            def bw():
                g = node.grad
                if not keepdims and axis is not None:
                    g = g.reshape(node.data.shape)
                mx = src.max(axis=axis, keepdims=True)
                mask = (src == mx).astype(DTYPE)
                mask /= np.maximum(mask.sum(axis=axis, keepdims=True), 1.0)
                self._acc(np.broadcast_to(mask * g, src.shape).astype(DTYPE))
            node._backward = bw
        return node

    def __getitem__(self, idx):
        node = Tensor._node(self.data[idx], (self,))
        if node.requires_grad:
            shape = self.data.shape

            def bw():
                g = np.zeros(shape, dtype=DTYPE)
                np.add.at(g, idx, node.grad)
                self._acc(g)
            node._backward = bw
        return node

    # ---- nonlinearities ------------------------------------------------
    def exp(self):
        data = np.exp(self.data)
        node = Tensor._node(data, (self,))
        if node.requires_grad:
            def bw():
                self._acc(node.grad * data)
            node._backward = bw
        return node

    def log(self):
        data = np.log(np.maximum(self.data, 1e-30))
        node = Tensor._node(data, (self,))
        if node.requires_grad:
            src = self.data

            def bw():
                self._acc(node.grad / np.maximum(src, 1e-30))
            node._backward = bw
        return node

    def sqrt(self):
        data = np.sqrt(np.maximum(self.data, 0.0))
        node = Tensor._node(data, (self,))
        if node.requires_grad:
            def bw():
                self._acc(node.grad * 0.5 / np.maximum(data, 1e-12))
            node._backward = bw
        return node

    def tanh(self):
        data = np.tanh(self.data)
        node = Tensor._node(data, (self,))
        if node.requires_grad:
            def bw():
                self._acc(node.grad * (1.0 - data * data))
            node._backward = bw
        return node

    def sigmoid(self):
        data = 1.0 / (1.0 + np.exp(-np.clip(self.data, -60, 60)))
        node = Tensor._node(data, (self,))
        if node.requires_grad:
            def bw():
                self._acc(node.grad * data * (1.0 - data))
            node._backward = bw
        return node

    def relu(self):
        data = np.maximum(self.data, 0.0)
        node = Tensor._node(data, (self,))
        if node.requires_grad:
            src = self.data

            def bw():
                self._acc(node.grad * (src > 0))
            node._backward = bw
        return node

    def gelu(self):
        """tanh approximation of GELU (Hendrycks & Gimpel)."""
        k = np.sqrt(2.0 / np.pi)
        inner = k * (self.data + 0.044715 * self.data ** 3)
        t = np.tanh(inner)
        data = 0.5 * self.data * (1.0 + t)
        node = Tensor._node(data, (self,))
        if node.requires_grad:
            src = self.data

            def bw():
                dt = 1.0 - t * t
                din = k * (1.0 + 3.0 * 0.044715 * src ** 2)
                d = 0.5 * (1.0 + t) + 0.5 * src * dt * din
                self._acc(node.grad * d)
            node._backward = bw
        return node

    def silu(self):
        s = 1.0 / (1.0 + np.exp(-np.clip(self.data, -60, 60)))
        data = self.data * s
        node = Tensor._node(data, (self,))
        if node.requires_grad:
            src = self.data

            def bw():
                self._acc(node.grad * (s + src * s * (1 - s)))
            node._backward = bw
        return node

    def softmax(self, axis: int = -1):
        m = self.data.max(axis=axis, keepdims=True)
        e = np.exp(self.data - m)
        data = e / e.sum(axis=axis, keepdims=True)
        node = Tensor._node(data, (self,))
        if node.requires_grad:
            def bw():
                dot = (node.grad * data).sum(axis=axis, keepdims=True)
                self._acc(data * (node.grad - dot))
            node._backward = bw
        return node

    def log_softmax(self, axis: int = -1):
        m = self.data.max(axis=axis, keepdims=True)
        shifted = self.data - m
        data = shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
        node = Tensor._node(data, (self,))
        if node.requires_grad:
            def bw():
                self._acc(node.grad - np.exp(data) * node.grad.sum(
                    axis=axis, keepdims=True))
            node._backward = bw
        return node

    def cat(self, *others, axis: int = 0):
        return cat([self, *others], axis=axis)

    def __matmul__(self, other):
        return matmul(self, other)


# --------------------------------------------------------------------------
# module-level ops
# --------------------------------------------------------------------------


def _wrap(x) -> Tensor:
    return x if isinstance(x, Tensor) else Tensor(x)


def matmul(a, b) -> Tensor:
    """Batched matrix product (>=2D operands, broadcasting on batch dims)."""
    a, b = _wrap(a), _wrap(b)
    node = Tensor._node(np.matmul(a.data, b.data), (a, b))
    if node.requires_grad:
        ad, bd = a.data, b.data

        def bw():
            g = node.grad
            if a.requires_grad:
                a._acc(_unbroadcast(np.matmul(g, bd.swapaxes(-1, -2)),
                                    ad.shape))
            if b.requires_grad:
                b._acc(_unbroadcast(np.matmul(ad.swapaxes(-1, -2), g),
                                    bd.shape))
        node._backward = bw
    return node


def cat(tensors, axis: int = 0) -> Tensor:
    ts = tuple(_wrap(t) for t in tensors)
    node = Tensor._node(np.concatenate([t.data for t in ts], axis=axis), ts)
    if node.requires_grad:
        sizes = [t.data.shape[axis] for t in ts[:-1]]

        def bw():
            parts = np.split(node.grad, np.cumsum(sizes), axis=axis)
            for t, p in zip(ts, parts):
                if t.requires_grad:
                    t._acc(p)
        node._backward = bw
    return node


def stack(tensors, axis: int = 0) -> Tensor:
    ts = tuple(_wrap(t) for t in tensors)
    node = Tensor._node(np.stack([t.data for t in ts], axis=axis), ts)
    if node.requires_grad:
        def bw():
            for i, t in enumerate(ts):
                if t.requires_grad:
                    t._acc(np.take(node.grad, i, axis=axis))
        node._backward = bw
    return node


def gather_rows(weight: Tensor, index: np.ndarray) -> Tensor:
    """Rows of ``weight`` selected by an integer array (embedding lookup)."""
    index = np.asarray(index, dtype=np.int64)
    node = Tensor._node(weight.data[index], (weight,))
    if node.requires_grad:
        def bw():
            g = np.zeros_like(weight.data)
            np.add.at(g, index, node.grad)
            weight._acc(g)
        node._backward = bw
    return node


def dropout(x: Tensor, p: float, training: bool, rng=None) -> Tensor:
    if not training or p <= 0.0:
        return x
    rng = rng or np.random.default_rng()
    mask = (rng.random(x.data.shape) >= p).astype(DTYPE) / (1.0 - p)
    node = Tensor._node(x.data * mask, (x,))
    if node.requires_grad:
        def bw():
            x._acc(node.grad * mask)
        node._backward = bw
    return node


def cross_entropy(logits: Tensor, targets, class_weight=None) -> Tensor:
    """Mean softmax cross-entropy: stable log-softmax + NLL fused in one op."""
    targets = np.asarray(targets, dtype=np.int64).ravel()
    x = logits.data
    n = x.shape[0]
    m = x.max(1, keepdims=True)
    lse = np.log(np.exp(x - m).sum(1, keepdims=True)) + m
    logp = x - lse
    nll = -logp[np.arange(n), targets]
    cw = None if class_weight is None else \
        np.asarray(class_weight, dtype=DTYPE)[targets]
    if cw is not None:
        nll = nll * cw
    node = Tensor._node(np.asarray(nll.mean(), dtype=DTYPE), (logits,))
    if node.requires_grad:
        def bw():
            probs = np.exp(logp)
            probs[np.arange(n), targets] -= 1.0
            probs /= n
            if cw is not None:
                probs *= cw[:, None]
            logits._acc(node.grad * probs)
        node._backward = bw
    return node


# --------------------------------------------------------------------------
# convolution / pooling (im2col based)
# --------------------------------------------------------------------------


def _im2col(x: np.ndarray, kH: int, kW: int, stride: int, pad: int):
    N, C, H, W = x.shape
    Ho = (H + 2 * pad - kH) // stride + 1
    Wo = (W + 2 * pad - kW) // stride + 1
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad))) if pad else x
    cols = np.empty((N, C, kH, kW, Ho, Wo), dtype=x.dtype)
    for i in range(kH):
        for j in range(kW):
            cols[:, :, i, j] = xp[:, :, i:i + stride * Ho:stride,
                                  j:j + stride * Wo:stride]
    return cols.reshape(N, C * kH * kW, Ho * Wo), Ho, Wo


def _col2im(cols: np.ndarray, x_shape, kH: int, kW: int, stride: int, pad: int,
            Ho: int, Wo: int) -> np.ndarray:
    N, C, H, W = x_shape
    cols = cols.reshape(N, C, kH, kW, Ho, Wo)
    xp = np.zeros((N, C, H + 2 * pad, W + 2 * pad), dtype=cols.dtype)
    for i in range(kH):
        for j in range(kW):
            xp[:, :, i:i + stride * Ho:stride, j:j + stride * Wo:stride] \
                += cols[:, :, i, j]
    if pad:
        return xp[:, :, pad:pad + H, pad:pad + W]
    return xp


def conv2d(x: Tensor, weight: Tensor, bias: Tensor | None = None,
           stride: int = 1, padding: int = 0) -> Tensor:
    """2D cross-correlation. weight shape (out_ch, in_ch, kH, kW)."""
    kH, kW = weight.data.shape[2], weight.data.shape[3]
    cols, Ho, Wo = _im2col(x.data, kH, kW, stride, padding)
    W2 = weight.data.reshape(weight.data.shape[0], -1)          # (OC, Ck2)
    raw = np.matmul(W2, cols)                                   # (N, OC, L)
    data = raw.reshape(x.data.shape[0], weight.data.shape[0], Ho, Wo)
    parents = (x, weight) if bias is None else (x, weight, bias)
    if bias is not None:
        data = data + bias.data.reshape(1, -1, 1, 1)
    node = Tensor._node(data, parents)
    if node.requires_grad:
        def bw():
            g = node.grad
            if bias is not None and bias.requires_grad:
                bias._acc(g.sum(axis=(0, 2, 3)))
            g2 = g.reshape(g.shape[0], g.shape[1], -1)           # (N, OC, L)
            if weight.requires_grad:
                # dW[oc, c] = sum_{n,l} g2[n, oc, l] * cols[n, c, l]
                dW = np.einsum("nol,ncl->oc", g2, cols)
                weight._acc(dW.reshape(weight.data.shape))
            if x.requires_grad:
                # dcols[n, c, l] = sum_oc W2[oc, c] * g2[n, oc, l]
                dcols = np.einsum("oc,nol->ncl", W2, g2)
                x._acc(_col2im(dcols, x.data.shape, kH, kW, stride,
                               padding, Ho, Wo))
        node._backward = bw
    return node


def maxpool2d(x: Tensor, pool: int = 2, stride: int | None = None) -> Tensor:
    stride = stride or pool
    cols, Ho, Wo = _im2col(x.data, pool, pool, stride, 0)
    N, C = x.data.shape[0], x.data.shape[1]
    win = pool * pool
    win_cols = cols.reshape(N, C, win, -1)          # per-channel windows
    idx = win_cols.argmax(axis=2)                    # (N, C, L)
    data = win_cols.max(axis=2).reshape(N, C, Ho, Wo)
    node = Tensor._node(data, (x,))
    if node.requires_grad:
        def bw():
            g = node.grad.reshape(N, C, -1)          # (N, C, L)
            gcols = np.zeros_like(win_cols)
            np.put_along_axis(gcols, idx[:, :, None, :], g[:, :, None, :],
                              axis=2)
            x._acc(_col2im(gcols.reshape(N, C * win, -1), x.data.shape,
                           pool, pool, stride, 0, Ho, Wo))
        node._backward = bw
    return node


def avgpool2d(x: Tensor, pool: int = 2, stride: int | None = None) -> Tensor:
    stride = stride or pool
    cols, Ho, Wo = _im2col(x.data, pool, pool, stride, 0)
    N, C = x.data.shape[0], x.data.shape[1]
    win = pool * pool
    data = cols.reshape(N, C, win, -1).mean(axis=2).reshape(N, C, Ho, Wo)
    node = Tensor._node(data, (x,))
    if node.requires_grad:
        scale = 1.0 / win

        def bw():
            g = node.grad.reshape(N, C, -1)
            gcols = np.repeat(g[:, :, None, :], win, axis=2)
            x._acc(_col2im(gcols.reshape(N, C * win, -1), x.data.shape,
                           pool, pool, stride, 0, Ho, Wo) * scale)
        node._backward = bw
    return node


# --------------------------------------------------------------------------
# graph traversal
# --------------------------------------------------------------------------


def _topological_order(root: Tensor) -> list[Tensor]:
    """Iterative post-order traversal (no recursion → no depth limits).

    Parents (``_prev``) are emitted strictly before their consumers so the
    reversed execution propagates gradients correctly.
    """
    out: list[Tensor] = []
    done: set[int] = set()
    stack = [root]
    while stack:
        v = stack.pop()
        vid = id(v)
        if vid in done:
            continue
        pending = [p for p in v._prev if id(p) not in done]
        if pending:
            stack.append(v)
            stack.extend(pending)
        else:
            done.add(vid)
            out.append(v)
    return out


# --------------------------------------------------------------------------
# gradient checking utility
# --------------------------------------------------------------------------


def numerical_grad(f, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        fp = f()
        x[idx] = orig - eps
        fm = f()
        x[idx] = orig
        g[idx] = (fp - fm) / (2 * eps)
        it.iternext()
    return g


def gradcheck_report(build_loss, params: list[Tensor], eps: float | None = None,
                     tol: float | None = None) -> list[tuple]:
    """Backprop-vs-numerical gradient check for a list of parameter tensors.

    Central differences on float32 losses have quantization noise of roughly
    ``spacing(loss) / (2*eps)``.  We pick ``eps`` from the loss scale and add
    that noise floor to the per-element tolerance, so the check is meaningful
    for both tiny and large losses.
    """
    atol, rtol = 1e-2, 5e-2
    kink_tol = 0.05          # ReLU kinks are non-differentiable: central
    # differences legitimately disagree near the kink when analytic == 0
    for p in params:
        p.grad = None
    loss = build_loss()
    L = abs(float(loss.item()))
    if eps is None:
        eps = 1e-3 * max(1.0, float(np.sqrt(max(L, 1.0))))
    noise = float(np.spacing(np.float32(max(L, 1e-12)))) / (2.0 * eps)
    loss.backward()
    report = []
    for p in params:
        analytic = p.grad.copy() if p.grad is not None else np.zeros_like(p.data)
        numeric = numerical_grad(lambda: float(build_loss().item()), p.data, eps)
        diff = np.abs(analytic - numeric)
        scale = np.abs(analytic) + np.abs(numeric)
        kink = (analytic == 0) & (np.abs(numeric) < kink_tol)
        sig_cut = max(1e-3, 10.0 * noise)
        significant = (scale > sig_cut) & ~kink
        max_rel = float((diff[significant] / scale[significant]).max()) \
            if significant.any() else 0.0
        ok = bool(np.all((diff <= atol + rtol * np.abs(numeric)
                          + 3.0 * noise) | kink))
        if tol is not None:
            ok = ok and max_rel < tol
        report.append((p.name or "param", p.shape, max_rel, ok))
    return report