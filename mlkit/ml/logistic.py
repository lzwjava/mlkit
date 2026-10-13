"""Logistic regression (softmax / multinomial) trained with gradient descent.

Supports binary and multi-class problems, L2 regularisation, mini-batches
and optional class weighting.
"""

from __future__ import annotations

import numpy as np


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    e = np.exp(Z)
    return e / e.sum(axis=1, keepdims=True)


class LogisticRegression:
    def __init__(self, lr: float = 0.5, epochs: int = 300, l2: float = 0.0,
                 batch_size: int = 64, fit_intercept: bool = True,
                 tol: float = 1e-8, patience: int = 30, seed: int = 0,
                 verbose: bool = False):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.batch_size = batch_size
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.patience = patience
        self.seed = seed
        self.verbose = verbose
        self.loss_history_: list[float] = []

    def _add_bias(self, X):
        if not self.fit_intercept:
            return X
        return np.hstack([np.ones((len(X), 1)), X])

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        # map labels to 0..K-1
        self.label_index_ = {c: i for i, c in enumerate(self.classes_)}
        yi = np.array([self.label_index_[v] for v in y])

        A = self._add_bias(X)
        n, d = A.shape
        W = np.zeros((d, K))
        rng = np.random.default_rng(self.seed)

        prev_loss = np.inf
        patience_left = self.patience
        for epoch in range(self.epochs):
            idx = rng.permutation(n)
            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                b = idx[start:start + self.batch_size]
                Ab, yb = A[b], yi[b]
                P = softmax(Ab @ W)
                loss = self._nll(P, yb) + 0.5 * self.l2 * (W ** 2).sum()
                epoch_loss += loss * len(b)
                # gradient of cross-entropy w.r.t. logits
                dW = Ab.T @ (P - np.eye(K)[yb]) / len(b) + self.l2 * W
                W -= self.lr * dW
            epoch_loss /= n
            self.loss_history_.append(float(epoch_loss))
            if self.verbose and (epoch % 50 == 0 or epoch == self.epochs - 1):
                print(f"    epoch {epoch:4d}  loss {epoch_loss:.5f}")
            if abs(prev_loss - epoch_loss) < self.tol:
                patience_left -= 1
                if patience_left <= 0:
                    break
            else:
                patience_left = self.patience
            prev_loss = epoch_loss

        self.W_ = W
        return self

    @staticmethod
    def _nll(P, yb):
        return float(-np.log(P[np.arange(len(yb)), yb] + 1e-12).mean())

    def decision_function(self, X):
        return self._add_bias(np.asarray(X, dtype=float)) @ self.W_

    def predict_proba(self, X):
        return softmax(self.decision_function(X))

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(1)]

    def score(self, X, y):
        return float((self.predict(X) == np.asarray(y)).mean())