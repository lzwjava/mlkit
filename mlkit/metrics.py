"""Evaluation metrics implemented from scratch (classification, regression, clustering)."""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------

def accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float((y_true == y_pred).mean())


def confusion_matrix(y_true, y_pred, labels=None) -> np.ndarray:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    index = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[index[t], index[p]] += 1
    return cm


def precision_recall_f1(y_true, y_pred, average: str = "macro", labels=None):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels)
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    pred = cm.sum(axis=0).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(pred > 0, tp / pred, 0.0)
        recall = np.where(support > 0, tp / support, 0.0)
        f1 = np.where(precision + recall > 0,
                      2 * precision * recall / (precision + recall), 0.0)
    if average == "macro":
        return precision.mean(), recall.mean(), f1.mean()
    if average == "weighted":
        w = support / max(support.sum(), 1)
        return float(precision @ w), float(recall @ w), float(f1 @ w)
    # binary: positive class is the last label
    return precision[-1], recall[-1], f1[-1]


def log_loss(y_true, y_prob, eps: float = 1e-12) -> float:
    y_true = np.asarray(y_true).ravel()
    p = np.clip(np.asarray(y_prob, dtype=float), eps, 1 - eps)
    if p.ndim == 1:
        return float(-(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)).mean())
    return float(-np.log(p[np.arange(len(y_true)), y_true.astype(int)]).mean())


# --------------------------------------------------------------------------
# regression
# --------------------------------------------------------------------------

def mse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


# --------------------------------------------------------------------------
# clustering
# --------------------------------------------------------------------------

def silhouette_score(X, labels) -> float:
    """Mean silhouette coefficient (O(n^2) pairwise distances)."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels).ravel()
    unique = np.unique(labels)
    if len(unique) < 2:
        return 0.0
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    D = np.sqrt(d2)
    n = len(labels)
    sil = np.zeros(n)
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        a = D[i, same].mean() if same.sum() > 0 else 0.0
        b = min(D[i, labels == c].mean() for c in unique if c != labels[i])
        sil[i] = (b - a) / max(a, b, 1e-12)
    return float(sil.mean())


def inertia(X, labels, centers) -> float:
    X = np.asarray(X, dtype=float)
    centers = np.asarray(centers, dtype=float)
    labels = np.asarray(labels).ravel()
    return float(((X - centers[labels]) ** 2).sum())


def explained_variance(X, X_hat) -> float:
    X = np.asarray(X, dtype=float)
    X_hat = np.asarray(X_hat, dtype=float)
    return float(1 - np.sum((X - X_hat) ** 2) / max(np.sum((X - X.mean(0)) ** 2), 1e-12))


def perplexity(nll_per_token: float) -> float:
    """Perplexity = exp(average negative log-likelihood)."""
    return float(np.exp(nll_per_token))


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------

def evaluate(task: str, y_true, y_pred, y_prob=None) -> dict:
    if task == "regression":
        return {"mse": mse(y_true, y_pred), "rmse": rmse(y_true, y_pred),
                "mae": mae(y_true, y_pred), "r2": r2_score(y_true, y_pred)}
    out = {"accuracy": accuracy(y_true, y_pred)}
    p, r, f1 = precision_recall_f1(y_true, y_pred)
    out.update({"precision": p, "recall": r, "f1": f1})
    if y_prob is not None:
        out["log_loss"] = log_loss(y_true, y_prob)
    return out