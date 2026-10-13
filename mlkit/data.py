"""Datasets: synthetic generators, CSV loading, splitting / scaling, text corpus.

Every generator returns ``(X, y, task)`` where task is ``"classification"``
or ``"regression"`` (or ``None`` for unlabelled data).
"""

from __future__ import annotations

import csv
import os

import numpy as np

# --------------------------------------------------------------------------
# generators
# --------------------------------------------------------------------------


def make_blobs(n=600, centers=3, dim=2, std=0.9, seed=0):
    """Isotropic gaussian clusters (k clusters in `dim` dimensions)."""
    rng = np.random.default_rng(seed)
    c = rng.uniform(-6, 6, size=(centers, dim))
    y = rng.integers(0, centers, size=n)
    X = c[y] + rng.normal(0, std, size=(n, dim))
    return X.astype(float), y, "classification"


def make_moons(n=600, noise=0.18, seed=0):
    """Two interleaving half-circles (non-linearly separable)."""
    rng = np.random.default_rng(seed)
    n1, n2 = n // 2, n - n // 2
    t1 = np.linspace(0, np.pi, n1)
    t2 = np.linspace(0, np.pi, n2)
    X1 = np.stack([np.cos(t1), np.sin(t1)], 1)
    X2 = np.stack([1 - np.cos(t2), 0.5 - np.sin(t2)], 1)
    X = np.vstack([X1, X2]) + rng.normal(0, noise, size=(n, 2))
    y = np.array([0] * n1 + [1] * n2)
    idx = rng.permutation(n)
    return X[idx].astype(float), y[idx], "classification"


def make_circles(n=600, noise=0.12, factor=0.5, seed=0):
    """Two concentric rings (separable by radius only)."""
    rng = np.random.default_rng(seed)
    n1, n2 = n // 2, n - n // 2
    t1 = rng.uniform(0, 2 * np.pi, n1)
    t2 = rng.uniform(0, 2 * np.pi, n2)
    X1 = np.stack([np.cos(t1), np.sin(t1)], 1)
    X2 = factor * np.stack([np.cos(t2), np.sin(t2)], 1)
    X = np.vstack([X1, X2]) + rng.normal(0, noise, size=(n, 2))
    y = np.array([0] * n1 + [1] * n2)
    idx = rng.permutation(n)
    return X[idx].astype(float), y[idx], "classification"


def make_spirals(n=600, classes=3, noise=0.25, seed=0):
    """Interleaving spirals (hard non-linear classification)."""
    rng = np.random.default_rng(seed)
    per = n // classes
    Xs, ys = [], []
    for c in range(classes):
        t = np.linspace(0.4, 2.4 * np.pi, per)
        r = t
        x = r * np.cos(t + c * 2 * np.pi / classes)
        y = r * np.sin(t + c * 2 * np.pi / classes)
        Xs.append(np.stack([x, y], 1))
        ys.append(np.full(per, c))
    X = np.vstack(Xs) + rng.normal(0, noise, size=(per * classes, 2))
    y = np.concatenate(ys)
    idx = rng.permutation(len(y))
    return X[idx].astype(float), y[idx], "classification"


def make_regression(n=600, dim=8, noise=0.3, nonlinear=True, seed=0):
    """y = X w1 + 0.6 sin(X w2) (+ noise) — partly linear, partly non-linear."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, dim))
    w1 = rng.normal(0, 1, size=dim)
    w2 = rng.normal(0, 1, size=dim)
    y = X @ w1 + (0.6 * np.sin(X @ w2) if nonlinear else 0.0)
    y = y + rng.normal(0, noise, size=n)
    return X.astype(float), y, "regression"


def make_xor(n=600, noise=0.3, seed=0):
    """XOR-shaped labels (linear models fail)."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2))
    y = ((X[:, 0] * X[:, 1]) > 0).astype(int)
    X = X + rng.normal(0, noise, size=(n, 2))
    return X.astype(float), y, "classification"


def make_gmm_data(n=600, k=3, dim=2, seed=0):
    """Clustering data drawn from a gaussian mixture (labels kept for scoring)."""
    rng = np.random.default_rng(seed)
    means = rng.uniform(-8, 8, size=(k, dim))
    y = rng.integers(0, k, size=n)
    X = means[y] + rng.normal(0, 1.0, size=(n, dim))
    return X.astype(float), y, "clustering"


def make_shapes(n=400, size=16, noise=0.1, seed=0):
    """Images of circle / square / triangle / cross -> (n, 1, size, size)."""
    rng = np.random.default_rng(seed)
    ys = rng.integers(0, 4, size=n)
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    c = (size - 1) / 2.0
    X = np.zeros((n, 1, size, size))
    for i, label in enumerate(ys):
        img = np.zeros((size, size))
        if label == 0:  # circle
            img[(xx - c) ** 2 + (yy - c) ** 2 <= (c * 0.85) ** 2] = 1.0
        elif label == 1:  # square
            img[(np.abs(xx - c) <= c * 0.75) & (np.abs(yy - c) <= c * 0.75)] = 1.0
        elif label == 2:  # triangle (upward)
            img[(yy >= size * 0.15) & (yy <= size * 0.85) &
                (np.abs(xx - c) <= (yy - size * 0.15) * 0.6)] = 1.0
        else:  # cross
            img[(np.abs(xx - c) <= c * 0.22)] = 1.0
            img[(np.abs(yy - c) <= c * 0.22)] = 1.0
        X[i, 0] = img + rng.normal(0, noise, size=(size, size))
    return X, ys.astype(int), "classification"


def make_sine(n=500, seq_len=24, horizon=4, freq=0.35, seed=0):
    """Sequence -> next-value regression (for RNN/LSTM/GRU)."""
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for _ in range(n):
        phase = rng.uniform(0, 2 * np.pi)
        f = freq * rng.uniform(0.7, 1.3)
        t = np.arange(seq_len + horizon)
        sig = np.sin(f * t + phase) + 0.1 * rng.normal(size=seq_len + horizon)
        Xs.append(sig[:seq_len])
        ys.append(sig[seq_len:seq_len + horizon])
    X = np.stack(Xs)[:, :, None]                      # (n, seq_len, 1)
    y = np.stack(ys)                                  # (n, horizon)
    return X, y, "regression"


def make_copy_task(n=400, length=8, vocab=6, seed=0):
    """'Repeat the sequence' classification task: input tokens -> target class
    is the first token (tests memory, LSTM vs vanilla RNN)."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, vocab, size=(n, length))
    y = X[:, 0]
    return X, y, "classification"


def make_text_line(n=1, seed=0):
    return "hello world", None, None


# --------------------------------------------------------------------------
# tiny text corpus (template-generated English, deterministic)
# --------------------------------------------------------------------------

_CORPUS_TEMPLATES = [
    "the {adj} {noun} {verb} the {object} in the {place} .",
    "when the {noun} {verb} , the {adj} {place} {verb2} .",
    "a {adj} {noun} and a {adj} {noun} {verb} under the {place} .",
    "the {place} was {adj} , and the {noun} {verb} the {object} .",
    "nobody knew why the {noun} {verb} the {object} every {time} .",
    "in the {place} , a {adj} {noun} {verb} a {adj} {object} .",
    "the {adj} {noun} said : the {place} is {adj} tonight .",
    "if the {noun} {verb} the {object} , the {place} will be {adj} .",
    "they carried the {object} across the {place} until the {time} .",
    "a {adj} {noun} {verb2} beside the {adj} {place} all {time} .",
]
_CORPUS_WORDS = {
    "adj": ["old", "young", "quiet", "bright", "dark", "small", "great", "strange"],
    "noun": ["king", "child", "river", "engine", "garden", "shadow", "traveller", "village"],
    "verb": ["found", "carried", "watched", "followed", "opened", "left"],
    "verb2": ["shined", "moved", "waited", "sang", "glowed"],
    "object": ["map", "stone", "lantern", "book", "box", "feather", "key"],
    "place": ["forest", "harbour", "tower", "valley", "bridge", "market", "shore"],
    "time": ["morning", "night", "winter", "summer", "evening"],
}


def tiny_corpus(target_chars: int = 12000, seed: int = 0) -> str:
    """Deterministic, pattern-rich English corpus for training a tiny LM."""
    rng = np.random.default_rng(seed)
    parts, size = [], 0
    while size < target_chars:
        tpl = _CORPUS_TEMPLATES[rng.integers(0, len(_CORPUS_TEMPLATES))]
        filled = tpl
        for key, words in _CORPUS_WORDS.items():
            while "{" + key + "}" in filled:
                filled = filled.replace("{" + key + "}",
                                         str(words[rng.integers(0, len(words))]), 1)
        parts.append(filled[0].upper() + filled[1:])
        size += len(filled)
    return " ".join(parts)


# --------------------------------------------------------------------------
# registry + loading
# --------------------------------------------------------------------------

DATASETS = {
    "blobs": make_blobs,
    "moons": make_moons,
    "circles": make_circles,
    "spirals": make_spirals,
    "xor": make_xor,
    "gmm": make_gmm_data,
    "regression": make_regression,
    "shapes": make_shapes,
    "sine": make_sine,
    "copy": make_copy_task,
}


def get_dataset(name: str, n: int = 600, seed: int = 42, **kw):
    if name not in DATASETS:
        raise KeyError(f"unknown dataset '{name}', choose from {sorted(DATASETS)}")
    return DATASETS[name](n=n, seed=seed, **kw)


def load_csv(path: str, target: str | None = None, test: float = 0.2,
             standardize: bool = True, seed: int = 42):
    """Load a CSV file. The target column can be named or defaults to the last one."""
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [r for r in reader if r]
    header = [h.strip() for h in header]
    if target is None:
        tcol = len(header) - 1
    else:
        tcol = header.index(target)
    feat_idx = [i for i in range(len(header)) if i != tcol]

    X_cols = [[r[i] for i in feat_idx] for r in rows]
    y_raw = [rows_i[tcol] for rows_i in rows]

    # numeric or categorical features
    def _to_float(vals):
        out = np.empty((len(vals), len(vals[0])), dtype=float)
        for j in range(len(vals[0])):
            for i in range(len(vals)):
                out[i, j] = float(vals[i][j])
        return out

    try:
        X = _to_float(X_cols)
    except ValueError:
        X = _categorical_matrix(X_cols)  # one-hot encode

    classes = np.unique(y_raw)
    if len(classes) <= 20 and all(_is_number(c) is False for c in classes) or \
       len(classes) <= 20 and _all_int(y_raw):
        y = np.searchsorted(classes, y_raw)
        task = "classification"
    else:
        y = np.array([float(v) for v in y_raw])
        task = "regression" if y.dtype.kind == "f" else "classification"
        if y.dtype.kind == "f" and len(np.unique(y)) > 20:
            task = "regression"
        else:
            y = y.astype(int)
            task = "classification"

    Xtr, Xte, ytr, yte = train_test_split(X, y, test=test, seed=seed,
                                           stratify=task == "classification")
    if standardize:
        Xtr, mu, sd = standardize_fit(Xtr)
        Xte = standardize_apply(Xte, mu, sd)
    return (Xtr, Xte, ytr, yte), task, header[tcol]


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _all_int(vals) -> bool:
    try:
        return all(float(v).is_integer() for v in vals)
    except (ValueError, AttributeError):
        return False


def _categorical_matrix(cols) -> np.ndarray:
    out_cols = []
    for col in zip(*cols):
        uniq = sorted(set(col))
        idx = {v: i for i, v in enumerate(uniq)}
        m = np.zeros((len(col), len(uniq)))
        for i, v in enumerate(col):
            m[i, idx[v]] = 1.0
        out_cols.append(m)
    return np.hstack(out_cols)


# --------------------------------------------------------------------------
# splitting / scaling / encoding
# --------------------------------------------------------------------------


def train_test_split(X, y, test=0.2, seed=42, stratify=False):
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(X)
    rng = np.random.default_rng(seed)
    if stratify:
        idx_tr, idx_te = [], []
        for c in np.unique(y):
            ci = np.where(y == c)[0]
            rng.shuffle(ci)
            n_te = max(1, int(round(len(ci) * test)))
            idx_te.extend(ci[:n_te])
            idx_tr.extend(ci[n_te:])
        idx_tr, idx_te = np.array(idx_tr), np.array(idx_te)
    else:
        idx = rng.permutation(n)
        n_te = max(1, int(round(n * test)))
        idx_te, idx_tr = idx[:n_te], idx[n_te:]
    return X[idx_tr], X[idx_te], y[idx_tr], y[idx_te]


def standardize_fit(X, mean=None, std=None):
    X = np.asarray(X, dtype=float)
    mu = X.mean(0) if mean is None else mean
    sd = X.std(0) if std is None else std
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd, mu, sd


def standardize_apply(X, mu, sd):
    return (np.asarray(X, dtype=float) - mu) / sd


def one_hot(y, k):
    y = np.asarray(y, dtype=int)
    out = np.zeros((len(y), k))
    out[np.arange(len(y)), y] = 1.0
    return out


def batch_indices(n, batch_size, seed=0, shuffle=True):
    """Yield minibatch index arrays for one epoch."""
    idx = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for i in range(0, n, batch_size):
        yield idx[i:i + batch_size]