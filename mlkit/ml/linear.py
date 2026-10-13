"""Linear regression family: OLS, Ridge, Lasso, ElasticNet.

* OLS        -> closed form via least squares (SVD)
* Ridge      -> closed form  (X'X + aI)^-1 X'y
* Lasso      -> cyclic coordinate descent with soft-thresholding
* ElasticNet -> cyclic coordinate descent with L1 + L2
"""

from __future__ import annotations

import numpy as np


def _add_bias(X):
    return np.hstack([np.ones((len(X), 1)), np.asarray(X, dtype=float)])


class _BaseLinear:
    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True,
                 max_iter: int = 1000, tol: float = 1e-7, l1_ratio: float = 0.5):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.l1_ratio = l1_ratio
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.n_iter_: int = 0

    # shared helpers -------------------------------------------------------
    def _prepare(self, X, y):
        """Center data (closed-form / coordinate-descent solvers assume it)."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if self.fit_intercept:
            self.x_mean_ = X.mean(0)
            self.y_mean_ = float(y.mean())
            return X - self.x_mean_, y - self.y_mean_
        self.x_mean_ = np.zeros(X.shape[1])
        self.y_mean_ = 0.0
        return X, y

    def _finalize(self, w):
        self.coef_ = np.asarray(w, dtype=float)
        self.intercept_ = float(self.y_mean_ - self.x_mean_ @ self.coef_)
        return self

    def _coordinate_descent(self, Xs, ys, l1, l2):
        """Cyclic coordinate descent on
        (1/2n)||y-Xw||² + l1||w||₁ + l2/2||w||²
        (the 1/n makes `alpha` comparable to scikit-learn's convention).
        """
        n, d = Xs.shape
        w = np.zeros(d)
        r = ys.copy()                       # residual = ys - Xs @ w
        col_norm2 = (Xs ** 2).sum(0) / n
        col_norm2 = np.where(col_norm2 < 1e-12, 1.0, col_norm2)
        denom = col_norm2 + l2
        for it in range(self.max_iter):
            max_delta = 0.0
            for j in range(d):
                # rho_j = (1/n)<X_j, ys - Xw> + w_j ||X_j||²/n
                rho = (Xs[:, j] @ r) / n + w[j] * col_norm2[j]
                if abs(rho) <= l1:
                    new_w = 0.0
                else:
                    new_w = (np.sign(rho) * (abs(rho) - l1)) / denom[j]
                delta = new_w - w[j]
                if delta != 0.0:
                    r -= Xs[:, j] * delta
                    w[j] = new_w
                    max_delta = max(max_delta, abs(delta))
            if max_delta < self.tol:
                self.n_iter_ = it + 1
                return w
        self.n_iter_ = self.max_iter
        return w

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        y = np.asarray(y, dtype=float).ravel()
        yp = self.predict(X)
        ss_res = np.sum((y - yp) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return float(1 - ss_res / max(ss_tot, 1e-12))


class LinearRegression(_BaseLinear):
    """Ordinary least squares (closed form, SVD-based ``lstsq``)."""

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        A = _add_bias(X) if self.fit_intercept else X
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        if self.fit_intercept:
            self.intercept_, self.coef_ = float(w[0]), w[1:]
        else:
            self.intercept_, self.coef_ = 0.0, w
        self.n_iter_ = 1
        return self


class Ridge(_BaseLinear):
    """L2-regularised regression: closed form w = (X'X + aI)^-1 X'y."""

    def fit(self, X, y):
        Xs, ys = self._prepare(X, y)
        d = Xs.shape[1]
        A = Xs.T @ Xs + self.alpha * np.eye(d)
        return self._finalize(np.linalg.solve(A, Xs.T @ ys))


class Lasso(_BaseLinear):
    """L1 regression via cyclic coordinate descent with soft-thresholding."""

    @staticmethod
    def soft_threshold(z, gamma):
        return np.sign(z) * np.maximum(np.abs(z) - gamma, 0.0)

    def fit(self, X, y):
        Xs, ys = self._prepare(X, y)
        w = self._coordinate_descent(Xs, ys, l1=self.alpha, l2=0.0)
        return self._finalize(w)


class ElasticNet(_BaseLinear):
    """L1 + L2 regression via cyclic coordinate descent."""

    def fit(self, X, y):
        Xs, ys = self._prepare(X, y)
        a1 = self.alpha * self.l1_ratio
        a2 = self.alpha * (1.0 - self.l1_ratio)
        w = self._coordinate_descent(Xs, ys, l1=a1, l2=a2)
        return self._finalize(w)