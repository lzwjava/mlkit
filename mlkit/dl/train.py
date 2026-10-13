"""Model definitions + training loops for the deep-learning CLI commands.

Models: MLP, CNN, RNN/GRU/LSTM sequence models, Vision Transformer (ViT),
GAN, VAE and a DDPM diffusion model — all on the numpy autograd engine.
"""

from __future__ import annotations

import numpy as np

from ..utils import curve
from . import nn
from .nn import (Conv2d, Embedding, Flatten, Linear, LayerNorm, MaxPool2d,
                 MLP, Sequential, Dropout, ReLU, LSTM, GRU, RNN,
                 TransformerBlock)
from .optim import clip_grad_norm_, get_optimizer
from .tensor import Tensor, cat, cross_entropy


# --------------------------------------------------------------------------
# generic supervised trainer (classification / regression)
# --------------------------------------------------------------------------

def iterate_minibatches(n, batch, rng, shuffle=True):
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, n, batch):
        yield idx[i:i + batch]


def fit_model(model: nn.Module, X, y, *, task: str = "classification",
              optimizer: str = "adam", lr: float | None = None,
              epochs: int = 50, batch_size: int = 64, l2: float = 0.0,
              clip: float = 0.0, verbose: bool = True, seed: int = 0,
              log_every: int = 10) -> dict:
    """Train any Module with a (B, D) -> logits/output interface."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    opt = get_optimizer(optimizer, model.parameters(), lr=lr, weight_decay=l2)
    if verbose:
        info = opt.info()
        print(f"    params={info['params']:,}  opt={info['optimizer']} "
              f"lr={info['lr']:g}  batch={batch_size}")
    history = {"loss": [], "metric": []}
    metric_name = "acc" if task == "classification" else "r2"

    for epoch in range(epochs):
        model.train(True)
        epoch_loss, seen = 0.0, 0
        for mb in iterate_minibatches(len(X), batch_size, rng):
            xb = Tensor(X[mb])
            if task == "classification":
                loss = cross_entropy(model(xb), y[mb].astype(np.int64))
            else:
                pred = model(xb)
                target = y[mb].astype(np.float32).reshape(pred.data.shape)
                loss = ((pred - Tensor(target)) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            if clip:
                clip_grad_norm_(model.parameters(), clip)
            opt.step()
            epoch_loss += float(loss.item()) * len(mb)
            seen += len(mb)
        epoch_loss /= max(seen, 1)

        model.eval()
        with np.errstate(invalid="ignore"):
            pred = model(Tensor(X))
        if task == "classification":
            yhat = pred.data.argmax(1)
            metric = float((yhat == np.asarray(y).ravel()).mean())
        else:
            yf = np.asarray(y, dtype=float).ravel()
            yhat = pred.data.ravel().astype(float)
            ss = float(((yf - yhat) ** 2).sum())
            st = float(((yf - yf.mean()) ** 2).sum())
            metric = 1.0 - ss / max(st, 1e-12)
        history["loss"].append(epoch_loss)
        history["metric"].append(metric)
        if verbose and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"    epoch {epoch:4d}  loss {epoch_loss:.4f}  "
                  f"{metric_name} {metric:.4f}   {curve(history['loss'])}")
    history["final_metric"] = history["metric"][-1] if history["metric"] else 0.0
    return history


# --------------------------------------------------------------------------
# MLP (feedforward classifier / regressor)
# --------------------------------------------------------------------------

def build_mlp(in_dim: int, out_dim: int, hidden=(64, 64),
              activation: str = "relu", dropout: float = 0.0,
              batch_norm: bool = False) -> MLP:
    return MLP(in_dim, list(hidden), out_dim, activation=activation,
               dropout=dropout, batch_norm=batch_norm)


# --------------------------------------------------------------------------
# CNN for (N, C, H, W) image inputs
# --------------------------------------------------------------------------

class CNN(nn.Module):
    def __init__(self, in_ch: int, n_classes: int, img_size: int = 16,
                 channels=(8, 16), kernel: int = 3, pool: int = 2,
                 dropout: float = 0.0):
        c1, c2 = channels
        self.features = Sequential(
            Conv2d(in_ch, c1, kernel), ReLU(), MaxPool2d(pool),
            Conv2d(c1, c2, kernel), ReLU(), MaxPool2d(pool),
            Flatten(),
        )
        probe = Tensor(np.zeros((1, in_ch, img_size, img_size), np.float32))
        flat = int(np.prod(self.features(probe).shape[1:]))
        self.head_dropout = Dropout(dropout)
        self.head = Linear(flat, n_classes)
        self.n_classes = n_classes

    def forward(self, x):
        return self.head(self.head_dropout(self.features(x)))


# --------------------------------------------------------------------------
# sequence model (RNN / GRU / LSTM) for (N, T, D) inputs
# --------------------------------------------------------------------------

class SequenceModel(nn.Module):
    """Many-to-one sequence classifier/regressor.

    Accepts either float features ``x: (B, T, D)`` or integer token ids
    ``x: (B, T)`` when ``vocab_size`` is given (tokens are embedded first).
    """

    def __init__(self, input_size: int, hidden_size: int, n_classes: int,
                 cell: str = "lstm", dropout: float = 0.0,
                 task: str = "classification", vocab_size: int | None = None):
        cell = cell.lower()
        rnn_cls = {"rnn": RNN, "gru": GRU, "lstm": LSTM}[cell]
        self.cell = cell
        self.vocab_size = vocab_size
        self.embed = Embedding(vocab_size, input_size) if vocab_size else None
        self.rnn = rnn_cls(input_size, hidden_size)
        self.dropout = Dropout(dropout)
        self.task = task
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.head = Linear(hidden_size, n_classes if task == "classification" else 1)

    def forward(self, x):
        if self.embed is not None and x.data.ndim == 2:
            x = self.embed(x.data.astype(np.int64))
        if self.cell == "lstm":
            Y, (h, _) = self.rnn(x)
        else:
            Y, h = self.rnn(x)
        last = Y[:, Y.shape[1] - 1, :]                 # many-to-one readout
        return self.head(self.dropout(last))


# --------------------------------------------------------------------------
# Vision Transformer (patch embedding + Transformer encoder)
# --------------------------------------------------------------------------

class ViT(nn.Module):
    def __init__(self, img_size: int = 16, in_ch: int = 1, patch: int = 4,
                 d_model: int = 32, n_head: int = 4, depth: int = 2,
                 n_classes: int = 4, mlp_ratio: int = 4, dropout: float = 0.0,
                 seed: int = 0):
        assert img_size % patch == 0, "img_size must be divisible by patch"
        self.patch = patch
        self.grid = img_size // patch
        self.n_patches = self.grid * self.grid
        self.patch_dim = in_ch * patch * patch
        self.d_model = d_model
        self.embed = Linear(self.patch_dim, d_model, bias=False)
        rng = np.random.default_rng(seed)
        self.pos = Tensor(
            rng.normal(0, 0.02, size=(self.n_patches + 1, d_model)),
            requires_grad=True)
        self.cls = Tensor(rng.normal(0, 0.02, size=(1, d_model)),
                          requires_grad=True)
        self.blocks = [TransformerBlock(d_model, n_head, self.n_patches + 1,
                                        mlp_ratio=mlp_ratio, dropout=dropout,
                                        causal=False)
                       for _ in range(depth)]
        self.ln = LayerNorm(d_model)
        self.head = Linear(d_model, n_classes)

    def _patchify(self, x: Tensor) -> Tensor:
        """(N, C, H, W) -> (N, n_patches, patch*patch*C)."""
        N, C, H, W = x.data.shape
        p, g = self.patch, self.grid
        a = x.reshape(N, C, g, p, g, p)        # N,C,gH,p,gW,p
        a = a.transpose(0, 2, 4, 3, 5, 1)       # N,gH,gW,p,p,C
        return a.reshape(N, self.n_patches, p * p * C)

    def forward(self, x):
        N = x.shape[0]
        patches = self._patchify(x) @ self.embed.weight       # (N, T, d)
        cls_b = self.cls.reshape(1, 1, -1) + Tensor(
            np.zeros((N, 1, self.d_model), np.float32))       # broadcast CLS
        z = cat([cls_b, patches], axis=1) + self.pos          # + positional
        for blk in self.blocks:
            z, _ = blk(z)
        return self.head(self.ln(z[:, 0, :]))                 # CLS readout


# --------------------------------------------------------------------------
# binary cross-entropy with logits (fused, exact gradient)
# --------------------------------------------------------------------------

def bce_logits(logits: Tensor, targets) -> Tensor:
    x = logits.data
    t = np.asarray(targets, dtype=np.float32).reshape(x.shape)
    sp = np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x)))      # softplus
    val = float((sp - t * x).mean())
    node = Tensor._node(np.asarray(val, np.float32), (logits,))
    if node.requires_grad:
        def bw():
            s = 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))
            logits._acc(np.asarray(node.grad * (s - t) / x.size, np.float32))
        node._backward = bw
    return node


# --------------------------------------------------------------------------
# GAN (2-D synthetic data)
# --------------------------------------------------------------------------

class LeakyReLU(nn.Module):
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def forward(self, x):
        a = self.alpha
        return Tensor._unary(
            x, np.maximum(x.data, a * x.data),
            lambda v, g, o: g * ((v > 0) + a * (v <= 0)))


class Generator(nn.Module):
    def __init__(self, z_dim: int, out_dim: int, hidden: int = 64):
        self.net = Sequential(Linear(z_dim, hidden), ReLU(),
                              Linear(hidden, hidden), ReLU(),
                              Linear(hidden, out_dim))

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        self.net = Sequential(Linear(in_dim, hidden), LeakyReLU(),
                              Linear(hidden, hidden), LeakyReLU(),
                              Linear(hidden, 1))

    def forward(self, x):
        return self.net(x)


def train_gan(real: np.ndarray, z_dim: int = 16, hidden: int = 64,
              epochs: int = 400, batch_size: int = 64, lr: float = 2e-3,
              seed: int = 0, verbose: bool = True) -> dict:
    """Minimax GAN training with BCE-with-logits losses (Adam, beta1=0.5)."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    real = real.astype(np.float32)
    G = Generator(z_dim, real.shape[1], hidden)
    D = Discriminator(real.shape[1], hidden)
    opt_g = get_optimizer("adam", G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = get_optimizer("adam", D.parameters(), lr=lr, betas=(0.5, 0.999))
    hist = {"d_loss": [], "g_loss": [], "d_acc": []}

    for epoch in range(epochs):
        d_tot = g_tot = d_acc_tot = 0.0
        n_mb = 0
        for mb in iterate_minibatches(len(real), batch_size, rng):
            xb = real[mb]
            n = len(mb)
            z = rng.normal(0, 1, size=(n, z_dim)).astype(np.float32)

            # --- discriminator step
            fake = G(Tensor(z)).data
            opt_d.zero_grad()
            loss_d = 0.5 * (bce_logits(D(Tensor(xb)), np.ones((n, 1), np.float32))
                            + bce_logits(D(Tensor(fake)),
                                         np.zeros((n, 1), np.float32)))
            loss_d.backward()
            opt_d.step()

            # --- generator step
            opt_g.zero_grad()
            z = rng.normal(0, 1, size=(n, z_dim)).astype(np.float32)
            gen = G(Tensor(z))
            loss_g = bce_logits(D(gen), np.ones((n, 1), np.float32))
            loss_g.backward()
            opt_g.step()

            d_tot += float(loss_d.item())
            g_tot += float(loss_g.item())
            with np.errstate(invalid="ignore"):
                acc = 0.5 * (float((D(Tensor(xb)).data > 0).mean())
                             + float((D(Tensor(G(Tensor(z)).data)).data <= 0).mean()))
            d_acc_tot += acc
            n_mb += 1
        hist["d_loss"].append(d_tot / max(n_mb, 1))
        hist["g_loss"].append(g_tot / max(n_mb, 1))
        hist["d_acc"].append(d_acc_tot / max(n_mb, 1))
        if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
            print(f"    epoch {epoch:4d}  D {hist['d_loss'][-1]:.4f}  "
                  f"G {hist['g_loss'][-1]:.4f}  D_acc {hist['d_acc'][-1]:.2f}"
                  f"   {curve(hist['g_loss'])}")
    z = rng.normal(0, 1, size=(len(real), z_dim)).astype(np.float32)
    hist["generator"], hist["discriminator"] = G, D
    hist["samples"] = G(Tensor(z)).data
    return hist


# --------------------------------------------------------------------------
# VAE (reparameterisation trick, ELBO = BCE + KL)
# --------------------------------------------------------------------------

class VAE(nn.Module):
    def __init__(self, input_dim: int, latent: int = 2, hidden: int = 64):
        self.enc = Sequential(Linear(input_dim, hidden), ReLU(),
                              Linear(hidden, hidden), ReLU())
        self.fc_mu = Linear(hidden, latent)
        self.fc_logvar = Linear(hidden, latent)
        self.dec = Sequential(Linear(latent, hidden), ReLU(),
                              Linear(hidden, hidden), ReLU(),
                              Linear(hidden, input_dim))
        self.latent = latent

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(self, mu, logvar, rng):
        eps = rng.normal(0, 1, size=mu.data.shape).astype(np.float32)
        return mu + (logvar * 0.5).exp() * Tensor(eps)

    def decode(self, z):
        return self.dec(z)

    def forward(self, x, rng):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar, rng)
        return self.decode(z), mu, logvar, z


def gaussian_recon(recon: Tensor, target: Tensor) -> Tensor:
    """0.5 * sum((r - x)^2) per sample, mean over batch (exact gradient)."""
    r, x = recon.data, target.data
    val = float(0.5 * (((r - x) ** 2).sum(axis=1)).mean())
    node = Tensor._node(np.asarray(val, np.float32), (recon,))
    if node.requires_grad:
        def bw():
            recon._acc(np.asarray(node.grad * (r - x) / r.shape[0], np.float32))
        node._backward = bw
    return node


def kl_standard_normal(mu: Tensor, logvar: Tensor) -> Tensor:
    """KL(q(z|x) || N(0,I)), mean over batch (exact gradients)."""
    m, lv = mu.data, logvar.data
    val = float((-0.5 * (1 + lv - m ** 2 - np.exp(lv))).sum(axis=1).mean())
    node = Tensor._node(np.asarray(val, np.float32), (mu, logvar))
    if node.requires_grad:
        def bw():
            scale = float(node.grad) / m.shape[0]
            # KL = 0.5 * sum(m^2 + e^lv - lv - 1)  =>  d/dm = m
            mu._acc(np.asarray(scale * m, np.float32))
            # d/dlogvar = 0.5(e^lv - 1)
            logvar._acc(np.asarray(scale * 0.5 * (np.exp(lv) - 1), np.float32))
        node._backward = bw
    return node


def train_vae(X: np.ndarray, latent: int = 2, hidden: int = 64,
              epochs: int = 300, batch_size: int = 64, lr: float = 1e-3,
              beta: float = 1.0, seed: int = 0, verbose: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    X = X.astype(np.float32)
    model = VAE(X.shape[1], latent, hidden)
    opt = get_optimizer("adam", model.parameters(), lr=lr)
    hist = {"loss": [], "recon": [], "kl": []}
    beta_t = Tensor(np.asarray(beta, np.float32))

    for epoch in range(epochs):
        model.train(True)
        tot_r = tot_k = 0.0
        n = 0
        for mb in iterate_minibatches(len(X), batch_size, rng):
            xb = Tensor(X[mb])
            recon, mu, logvar, z = model(xb, rng)
            loss = gaussian_recon(recon, xb) + beta_t * kl_standard_normal(mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_r += recon_value(recon.data, xb.data)
            tot_k += kl_value(mu.data, logvar.data)
            n += 1
        hist["loss"].append((tot_r + beta * tot_k) / max(n, 1))
        hist["recon"].append(tot_r / max(n, 1))
        hist["kl"].append(tot_k / max(n, 1))
        if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
            print(f"    epoch {epoch:4d}  elbo {hist['loss'][-1]:.4f}  "
                  f"recon {hist['recon'][-1]:.4f}  kl {hist['kl'][-1]:.4f}"
                  f"   {curve(hist['loss'])}")
    mu, _ = model.encode(Tensor(X))
    hist["model"] = model
    hist["z_mean"] = mu.data
    hist["samples"] = model.decode(
        Tensor(rng.normal(0, 1, size=(len(X), latent)).astype(np.float32))).data
    return hist


def recon_value(r: np.ndarray, x: np.ndarray) -> float:
    return float(0.5 * ((r - x) ** 2).sum(axis=1).mean())


def kl_value(m: np.ndarray, lv: np.ndarray) -> float:
    return float((-0.5 * (1 + lv - m ** 2 - np.exp(lv))).sum(axis=1).mean())


# --------------------------------------------------------------------------
# DDPM diffusion model on low-dimensional data (e.g. 2-D blobs)
# --------------------------------------------------------------------------

def make_beta_schedule(T: int, kind: str = "linear",
                       start: float = 1e-4, end: float = 0.02) -> np.ndarray:
    if kind == "cosine":
        s = 0.008
        t = np.linspace(0, T, T + 1, dtype=np.float64)
        f = np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
        abar = f / f[0]
        betas = 1 - abar[1:] / abar[:-1]
        return np.clip(betas, 0, 0.999).astype(np.float32)
    return np.linspace(start, end, T, dtype=np.float32)


def train_diffusion(X: np.ndarray, T: int = 200, hidden: int = 64,
                    epochs: int = 400, batch_size: int = 64, lr: float = 1e-3,
                    schedule: str = "linear", seed: int = 0,
                    verbose: bool = True) -> dict:
    """DDPM: the denoising network predicts eps_hat(x_t, t).

    Inputs are standardised to zero-mean/unit-std (samples are mapped back),
    which keeps the diffusion signal-to-noise range well conditioned.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float32)
    mu_x, sd_x = X.mean(0, keepdims=True), X.std(0, keepdims=True)
    sd_x = np.where(sd_x < 1e-6, 1.0, sd_x)
    X = ((X - mu_x) / sd_x).astype(np.float32)
    dim = X.shape[1]
    betas = make_beta_schedule(T, schedule)
    alphas = 1.0 - betas
    abar = np.cumprod(alphas).astype(np.float32)
    sqrt_abar, sqrt_1mab = np.sqrt(abar), np.sqrt(1.0 - abar)

    net = MLP(dim + 8, [hidden, hidden], dim, activation="relu")
    emb = Embedding(T, 8, std=0.05)                   # learned time embedding
    opt = get_optimizer("adam", net.parameters() + emb.parameters(), lr=lr)
    hist = {"loss": []}

    for epoch in range(epochs):
        tot, n = 0.0, 0
        for mb in iterate_minibatches(len(X), batch_size, rng):
            x0 = X[mb]
            m = len(mb)
            t = rng.integers(0, T, size=m)
            eps = rng.normal(0, 1, size=x0.shape).astype(np.float32)
            xt = sqrt_abar[t][:, None] * x0 + sqrt_1mab[t][:, None] * eps
            t_feat = emb(t)                                  # (m, 8)
            h = cat([Tensor(xt), t_feat], axis=1)
            pred = net(h)
            loss = ((pred - Tensor(eps)) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item()) * m
            n += m
        hist["loss"].append(tot / max(n, 1))
        if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
            print(f"    epoch {epoch:4d}  noise-MSE {hist['loss'][-1]:.4f}"
                  f"   {curve(hist['loss'])}")

    def sample(n_samples: int) -> np.ndarray:
        out = rng.normal(0, 1, size=(n_samples, dim)).astype(np.float32)
        for tt in reversed(range(T)):
            t_arr = np.full(n_samples, tt, dtype=np.int64)
            h = Tensor(np.concatenate(
                [out, emb(t_arr).data], axis=1))
            eps_hat = net(h).data
            a, ab = alphas[tt], abar[tt]
            mean = (out - (1 - a) / np.sqrt(1 - ab) * eps_hat) / np.sqrt(a)
            if tt > 0:
                out = mean + np.sqrt(betas[tt]) * rng.normal(
                    0, 1, size=out.shape).astype(np.float32)
            else:
                out = mean
        return out

    hist["net"], hist["sample_fn"] = net, sample
    hist["samples"] = sample(min(500, len(X))) * sd_x + mu_x     # de-standardise
    hist["mu_x"], hist["sd_x"] = mu_x, sd_x
    return hist


# --------------------------------------------------------------------------
# optimizer convergence demo (Rosenbrock / Beale)
# --------------------------------------------------------------------------

OPTIMIZER_BENCH = {
    "sgd": dict(momentum=0.9),
    "adam": dict(),
    "adamw": dict(),
    "rmsprop": dict(),
}


def run_optimizer_bench(fn, x0, steps: int = 400, lr: float = 1e-3,
                        names=("sgd", "adam", "adamw", "rmsprop"),
                        clip: float | None = None) -> dict:
    """Run several optimizers on the same function; return trajectories."""
    from .optim import (SGD, Adam, AdamW, RMSProp, optimize_function)
    classes = {"sgd": SGD, "adam": Adam, "adamw": AdamW, "rmsprop": RMSProp}
    out = {}
    for name in names:
        kwargs = dict(OPTIMIZER_BENCH.get(name, {}))
        kwargs["lr"] = lr
        traj, losses = optimize_function(fn, x0, classes[name], kwargs,
                                         steps=steps, clip=clip)
        out[name] = {"traj": traj, "losses": losses,
                     "final": losses[-1], "steps_to_1e-3": next(
                         (i for i, v in enumerate(losses) if v < 1e-3), None)}
    return out