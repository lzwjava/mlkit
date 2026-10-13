"""k-Nearest Neighbours classifier/regressor (brute-force with optional
KD-tree-style early abandoning via squared distances)."""

from __future__ import annotations

import numpy as np


class KNN:
    def __init__(self, k: int = 5, weights: str = "uniform",
                 task: str = "classification", metric: str = "euclidean"):
        self.k = k
        self.weights = weights          # 'uniform' | 'distance'
        self.task = task                # 'classification' | 'regression'
        self.metric = metric            # 'euclidean' | 'manhattan'

    # ------------------------------------------------------------------
    def _distances(self, ref, query):
        """Pairwise distances, shape (len(query), len(ref))."""
        if self.metric == "manhattan":
            # |x-q| summed over features  (chunked to bound memory)
            out = np.empty((len(query), len(ref)))
            for i in range(0, len(query), 64):
                chunk = query[i:i + 64]
                out[i:i + 64] = np.abs(chunk[:, None, :] - ref[None, :, :]).sum(-1)
            return out
        out = np.empty((len(query), len(ref)))
        for i in range(0, len(query), 64):
            chunk = query[i:i + 64]
            d2 = (chunk ** 2).sum(1)[:, None] + (ref ** 2).sum(1)[None, :] \
                 - 2 * chunk @ ref.T
            out[i:i + 64] = np.sqrt(np.maximum(d2, 0.0))
        return out

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.y_ = y.ravel()
        if self.task == "classification":
            self.classes_ = np.unique(self.y_)
        return self

    def kneighbors(self, Q, k=None):
        k = min(k or self.k, len(self.X_))
        D = self._distances(self.X_, np.asarray(Q, dtype=float))
        nn = np.argpartition(D, k - 1, axis=1)[:, :k]
        rows = np.arange(len(D))[:, None]
        dist = D[rows, nn]
        order = np.argsort(dist, axis=1)
        nn = np.take_along_axis(nn, order, axis=1)
        dist = np.take_along_axis(dist, order, axis=1)
        return nn, dist

    def predict(self, Q):
        Q = np.asarray(Q, dtype=float)
        nn, dist = self.kneighbors(Q)
        if self.task == "regression":
            vals = self.y_[nn].astype(float)
            if self.weights == "distance":
                w = 1.0 / np.maximum(dist, 1e-9)
                return (vals * w).sum(1) / w.sum(1)
            return vals.mean(1)

        # classification: weighted vote
        votes = np.zeros((len(Q), len(self.classes_)))
        cls_idx = {c: i for i, c in enumerate(self.classes_)}
        for r in range(len(Q)):
            for c_i, d in zip(nn[r], dist[r]):
                lab = cls_idx[self.y_[c_i]]
                votes[r, lab] += 1.0 / d if self.weights == "distance" else 1.0
        return self.classes_[votes.argmax(1)]

    def score(self, X, y):
        y = np.asarray(y)
        pred = self.predict(X)
        if self.task == "regression":
            ss = ((y.ravel() - pred) ** 2).sum()
            st = ((y.ravel() - y.mean()) ** 2).sum()
            return float(1 - ss / max(st, 1e-12))
        return float((pred == y).mean())