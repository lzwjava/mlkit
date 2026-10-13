"""First-order optimizers implemented from scratch:

* SGD        (momentum / Nesterov, weight decay)
* Adam       (Kingma & Ba; coupled L2 weight decay)
* AdamW      (Loshchilov & Hutter; decoupled weight decay)
* RMSProp    (Hinton; optional momentum, centered variance)

Plus gradient utilities: ``clip_grad_norm_`` / ``clip_grad_value_``.
"""

from __future__ import annotations

import numpy as np

from .tensor import Tensor


class Optimizer:
    name = "optimizer"

    def __init__(self, params, lr: float = 1e-3, weight_decay: float = 0.0):
        self.params, seen = [], set()
        for p in params:
            if p.requires_grad and id(p) not in seen:      # dedup tied weights
                seen.add(id(p))
                self.params.append(p)
        if not self.params:
            raise ValueError("optimizer got an empty parameter list")
        self.lr = lr
        self.weight_decay = weight_decay
        self.state: dict[int, dict] = {}
        self.step_count = 0

    def _state(self, p: Tensor) -> dict:
        return self.state.setdefault(id(p), {})

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        raise NotImplementedError

    def grad_norm(self) -> float:
        total = 0.0
        for p in self.params:
            if p.grad is not None:
                total += float((p.grad ** 2).sum())
        return float(np.sqrt(total))

    def param_norm(self) -> float:
        return float(np.sqrt(sum(float((p.data ** 2).sum())
                                 for p in self.params)))

    def info(self) -> dict:
        return {"optimizer": self.name, "lr": self.lr,
                "params": sum(p.data.size for p in self.params),
                "weight_decay": self.weight_decay}


class SGD(Optimizer):
    """Stochastic gradient descent with (optional) momentum & Nesterov."""

    name = "sgd"

    def __init__(self, params, lr: float = 0.01, momentum: float = 0.0,
                 nesterov: bool = False, weight_decay: float = 0.0):
        super().__init__(params, lr, weight_decay)
        self.momentum = momentum
        self.nesterov = nesterov

    def step(self):
        self.step_count += 1
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay:
                g = g + self.weight_decay * p.data
            if self.momentum:
                st = self._state(p)
                buf = st.get("buf")
                buf = g if buf is None else self.momentum * buf + g
                st["buf"] = buf
                if self.nesterov:
                    g = g + self.momentum * buf
                else:
                    g = buf
            p.data -= self.lr * g


class Adam(Optimizer):
    """Adam (bias-corrected first/second moments, coupled weight decay)."""

    name = "adam"

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(params, lr, weight_decay)
        self.beta1, self.beta2 = betas
        self.eps = eps

    def step(self):
        self.step_count += 1
        b1, b2 = self.beta1, self.beta2
        t = self.step_count
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay:                 # coupled L2
                g = g + self.weight_decay * p.data
            st = self._state(p)
            m = st.get("m")
            v = st.get("v")
            m = g if m is None else b1 * m + (1 - b1) * g
            v = g * g if v is None else b2 * v + (1 - b2) * g * g
            st["m"], st["v"] = m, v
            m_hat = m / bc1
            v_hat = v / bc2
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Optimizer):
    """AdamW: decoupled weight decay applied directly to the weights."""

    name = "adamw"

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.01):
        super().__init__(params, lr, weight_decay)
        self.beta1, self.beta2 = betas
        self.eps = eps

    def step(self):
        self.step_count += 1
        b1, b2 = self.beta1, self.beta2
        t = self.step_count
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad
            st = self._state(p)
            m = st.get("m")
            v = st.get("v")
            m = g if m is None else b1 * m + (1 - b1) * g
            v = g * g if v is None else b2 * v + (1 - b2) * g * g
            st["m"], st["v"] = m, v
            m_hat = m / bc1
            v_hat = v / bc2
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            if self.weight_decay:                 # decoupled
                p.data -= self.lr * self.weight_decay * p.data


class RMSProp(Optimizer):
    """RMSProp (optionally momentum / centered)."""

    name = "rmsprop"

    def __init__(self, params, lr: float = 1e-3, rho: float = 0.9,
                 eps: float = 1e-8, momentum: float = 0.0,
                 centered: bool = False, weight_decay: float = 0.0):
        super().__init__(params, lr, weight_decay)
        self.rho = rho
        self.eps = eps
        self.momentum = momentum
        self.centered = centered

    def step(self):
        self.step_count += 1
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay:
                g = g + self.weight_decay * p.data
            st = self._state(p)
            eg = st.get("eg")
            eg = g * g if eg is None else self.rho * eg + (1 - self.rho) * g * g
            st["eg"] = eg
            if self.centered:
                g_avg = st.get("g_avg")
                g_avg = g if g_avg is None else self.rho * g_avg + \
                    (1 - self.rho) * g
                st["g_avg"] = g_avg
                avg = np.sqrt(np.maximum(eg - g_avg ** 2, 1e-10))
            else:
                avg = np.sqrt(eg)
            denom = avg + self.eps
            update = g / denom
            if self.momentum:
                buf = st.get("buf")
                buf = update if buf is None else self.momentum * buf + update
                st["buf"] = buf
                update = buf
            p.data -= self.lr * update


OPTIMIZERS = {
    "sgd": SGD,
    "adam": Adam,
    "adamw": AdamW,
    "rmsprop": RMSProp,
}


def get_optimizer(name: str, params, lr: float | None = None,
                  weight_decay: float = 0.0, **kwargs) -> Optimizer:
    """Instantiate an optimizer by name with sane defaults."""
    name = name.lower()
    if name not in OPTIMIZERS:
        raise KeyError(f"unknown optimizer '{name}', choose from "
                       f"{sorted(OPTIMIZERS)}")
    defaults = {"sgd": dict(lr=0.1, momentum=0.9),
                "adam": dict(lr=1e-3),
                "adamw": dict(lr=1e-3, weight_decay=0.01),
                "rmsprop": dict(lr=1e-3, rho=0.9)}[name]
    if lr is not None:
        defaults["lr"] = lr
    if weight_decay:
        defaults["weight_decay"] = weight_decay
    defaults.update(kwargs)
    return OPTIMIZERS[name](params, **defaults)


# --------------------------------------------------------------------------
# gradient utilities
# --------------------------------------------------------------------------


def clip_grad_norm_(params, max_norm: float) -> float:
    """Rescale gradients so their global L2 norm is at most max_norm."""
    params = [p for p in params if p.grad is not None]
    total = sum(float((p.grad ** 2).sum()) for p in params)
    norm = float(np.sqrt(total))
    if norm > max_norm and norm > 0:
        scale = max_norm / norm
        for p in params:
            p.grad *= scale
    return norm


def clip_grad_value_(params, clip_value: float):
    for p in params:
        if p.grad is not None:
            np.clip(p.grad, -clip_value, clip_value, out=p.grad)


# --------------------------------------------------------------------------
# convergence comparison (the `mlkit dl optim` demo)
# --------------------------------------------------------------------------

def optimize_function(fgrad, x0, opt_cls, opt_kwargs, steps=300, clip=None):
    """Minimise f starting at x0 with an optimizer; return trajectory."""
    x = Tensor(np.array(x0, dtype=np.float32, copy=True),   # never alias x0
               requires_grad=True)
    opt = opt_cls([x], **opt_kwargs)
    traj, losses = [], []
    for _ in range(steps):
        opt.zero_grad()
        loss = fgrad(x)
        loss.backward()
        if clip:
            clip_grad_norm_([x], clip)
        opt.step()
        flat = x.data.ravel()
        traj.append((float(flat[0]), float(flat[1])))
        losses.append(float(loss.item()))
    return np.array(traj), losses


def rosenbrock(x: Tensor, a: float = 1.0, b: float = 100.0):
    """Classic banana-shaped test function."""
    x1 = x[:, 0:1]
    x2 = x[:, 1:2]
    return ((a - x1) ** 2 + b * (x2 - x1 ** 2) ** 2).mean()


def beale(x: Tensor):
    x1 = x[:, 0:1]
    x2 = x[:, 1:2]
    return ((1.5 - x1 + x1 * x2) ** 2 + (2.25 - x1 + x1 * x2 ** 2) ** 2
            + (2.625 - x1 + x1 * x2 ** 3) ** 2).mean()