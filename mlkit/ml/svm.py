"""Support Vector Machines.

* LinearSVM  -> soft-margin SVM trained with Pegasos-style sub-gradient SGD
                 (hinge loss + L2 regularisation)
* KernelSVM  -> kernel SVM trained with John Platt's SMO algorithm
                 (RBF / polynomial / linear kernels)
"""

from __future__ import annotations

import numpy as np


def make_kernel(K: str, gamma=None, degree: int = 3):
    K = K.lower()

    def linear(X, Z):
        return X @ Z.T

    def rbf(X, Z):
        g = gamma if gamma is not None else 1.0 / max(X.shape[1], 1)
        x2 = np.sum(X ** 2, 1)[:, None]
        z2 = np.sum(Z ** 2, 1)[None, :]
        d2 = x2 + z2 - 2 * X @ Z.T
        return np.exp(-g * np.maximum(d2, 0.0))

    def poly(X, Z):
        return (X @ Z.T + 1.0) ** degree

    return {"linear": linear, "rbf": rbf, "poly": poly,
            "polynomial": poly}[K]


class LinearSVM:
    """Soft-margin linear SVM (Pegasos: sub-gradient descent on the
    primal hinge-loss objective).

    minimise  λ/2 ||w||² + 1/n Σ max(0, 1 - y_i <w, x_i>)
    """

    def __init__(self, C: float = 1.0, epochs: int = 200, seed: int = 0,
                 fit_intercept: bool = True, verbose: bool = False):
        self.C = C            # C ~ 1/(n*lambda); kept for API familiarity
        self.epochs = epochs
        self.seed = seed
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.loss_history_: list[float] = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("LinearSVM supports binary labels only")
        yb = np.where(y == self.classes_[1], 1.0, -1.0)
        n, d = X.shape
        rng = np.random.default_rng(self.seed)

        lam = 1.0 / max(self.C * n, 1e-9)
        w = np.zeros(d)
        b = 0.0
        t = 0
        for epoch in range(self.epochs):
            idx = rng.permutation(n)
            epoch_loss = 0.0
            for i in idx:
                t += 1
                eta = 1.0 / (lam * t)
                margin = yb[i] * (X[i] @ w + b)
                epoch_loss += max(0.0, 1.0 - margin)
                if margin < 1.0:                       # violating sample
                    w = (1 - eta * lam) * w + eta * yb[i] * X[i]
                    b += eta * yb[i]
                else:
                    w = (1 - eta * lam) * w
            loss = 0.5 * lam * (w ** 2).sum() + epoch_loss / n
            self.loss_history_.append(float(loss))
            if self.verbose and (epoch % 50 == 0 or epoch == self.epochs - 1):
                print(f"    epoch {epoch:4d}  objective {loss:.5f}")

        self.coef_ = w
        self.intercept_ = float(b)
        self.support_ = None
        return self

    def decision_function(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def predict(self, X):
        s = self.decision_function(X)
        return np.where(s >= 0, self.classes_[1], self.classes_[0])

    def score(self, X, y):
        return float((self.predict(X) == np.asarray(y)).mean())


class KernelSVM:
    """Kernel SVM trained with a simplified version of Platt's SMO.

    Solves the dual:  max  Σa_i - ½ ΣΣ a_i a_j y_i y_j K(x_i,x_j)
    s.t. 0 ≤ a_i ≤ C,  Σ a_i y_i = 0
    """

    def __init__(self, C: float = 1.0, kernel: str = "rbf", gamma=None,
                 degree: int = 3, tol: float = 1e-3, max_passes: int = 50,
                 seed: int = 0, verbose: bool = False):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.tol = tol
        self.max_passes = max_passes
        self.seed = seed
        self.verbose = verbose

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("KernelSVM supports binary labels only")
        yb = np.where(y == self.classes_[1], 1.0, -1.0)
        n = len(X)
        K = make_kernel(self.kernel, self.gamma, self.degree)(X, X)

        rng = np.random.default_rng(self.seed)
        alpha = np.zeros(n)
        b = 0.0
        E = np.zeros(n)                      # cache of decision errors
        passes = 0
        while passes < self.max_passes:
            num_changed = 0
            for i in range(n):
                f_i = (alpha * yb) @ K[:, i] + b
                E[i] = f_i - yb[i]
                if not ((yb[i] * E[i] < -self.tol and alpha[i] < self.C) or
                        (yb[i] * E[i] > self.tol and alpha[i] > 0)):
                    continue
                j = int(rng.integers(0, n - 1))
                if j >= i:
                    j += 1
                f_j = (alpha * yb) @ K[:, j] + b
                E[j] = f_j - yb[j]

                a_i_old, a_j_old = alpha[i], alpha[j]
                if yb[i] != yb[j]:
                    L = max(0.0, a_j_old - a_i_old)
                    H = min(self.C, self.C + a_j_old - a_i_old)
                else:
                    L = max(0.0, a_i_old + a_j_old - self.C)
                    H = min(self.C, a_i_old + a_j_old)
                if L >= H:
                    continue

                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue
                a_j_new = a_j_old - yb[j] * (E[i] - E[j]) / eta
                a_j_new = min(H, max(L, a_j_new))
                if abs(a_j_new - a_j_old) < 1e-5:
                    continue
                a_i_new = a_i_old + yb[i] * yb[j] * (a_j_old - a_j_new)
                alpha[i], alpha[j] = a_i_new, a_j_new

                b1 = b - E[i] - yb[i] * (a_i_new - a_i_old) * K[i, i] \
                         - yb[j] * (a_j_new - a_j_old) * K[i, j]
                b2 = b - E[j] - yb[i] * (a_i_new - a_i_old) * K[i, j] \
                         - yb[j] * (a_j_new - a_j_old) * K[j, j]
                if 0 < a_i_new < self.C:
                    b = b1
                elif 0 < a_j_new < self.C:
                    b = b2
                else:
                    b = 0.5 * (b1 + b2)
                num_changed += 1
            passes = passes + 1 if num_changed == 0 else 0
            if self.verbose:
                print(f"    pass changed={num_changed}  sv={(alpha > 1e-6).sum()}")

        sv = alpha > 1e-6
        self.support_ = np.where(sv)[0]
        self.dual_coef_ = (alpha * yb)[sv]
        self.X_sv_ = X[sv]
        self.intercept_ = float(b)
        self.n_iter_ = passes
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        K = make_kernel(self.kernel, self.gamma, self.degree)(X, self.X_sv_)
        return K @ self.dual_coef_ + self.intercept_

    def predict(self, X):
        s = self.decision_function(X)
        return np.where(s >= 0, self.classes_[1], self.classes_[0])

    def score(self, X, y):
        return float((self.predict(X) == np.asarray(y)).mean())