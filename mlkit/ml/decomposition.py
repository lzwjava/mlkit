"""Dimensionality reduction: PCA, t-SNE, UMAP (simplified), Linear Discriminant
Analysis (Fisher LDA)."""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# PCA (SVD based)
# --------------------------------------------------------------------------

class PCA:
    def __init__(self, n_components: int = 2):
        self.n_components = n_components

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(0)
        Xc = X - self.mean_
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        k = min(self.n_components, Vt.shape[0])
        self.components_ = Vt[:k]
        self.explained_variance_ = (S[:k] ** 2) / max(len(X) - 1, 1)
        tot = (S ** 2).sum() / max(len(X) - 1, 1)
        self.explained_variance_ratio_ = self.explained_variance_ / max(tot, 1e-12)
        self.singular_values_ = S[:k]
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        return np.asarray(Z, dtype=float) @ self.components_ + self.mean_


# --------------------------------------------------------------------------
# t-SNE (exact O(n^2) implementation: early exaggeration + symmetrised P)
# --------------------------------------------------------------------------

class TSNE:
    def __init__(self, n_components: int = 2, perplexity: float = 30.0,
                 n_iter: int = 800, learning_rate: float = 200.0,
                 early_exaggeration: float = 12.0, init: str = "pca",
                 seed: int = 0, verbose: bool = False):
        self.n_components = n_components
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.early_exaggeration = early_exaggeration
        self.init = init
        self.seed = seed
        self.verbose = verbose

    # ------------------------------------------------------------------
    @staticmethod
    def _binary_search_beta(d2_row, target_entropy, bet_min=1e-12, bet_max=1e4):
        """Find precision beta so that Shannon entropy of the conditional
        distribution equals log2(perplexity)."""
        for _ in range(100):
            beta = 0.5 * (bet_min + bet_max)
            P = np.exp(-d2_row * beta)
            sum_p = P.sum()
            if sum_p < 1e-300:
                bet_max = beta
                continue
            H = np.log(sum_p) + beta * (d2_row * P).sum() / sum_p
            if H > target_entropy:
                bet_min = beta
            else:
                bet_max = beta
            if abs(H - target_entropy) < 1e-5:
                break
        return beta

    def _joint_prob(self, X):
        n = len(X)
        d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
        target = np.log2(max(self.perplexity, 2))
        P = np.zeros((n, n))
        for i in range(n):
            beta = self._binary_search_beta(d2[i], target)
            row = np.exp(-d2[i] * beta)
            row[i] = 0.0
            P[i] = row / max(row.sum(), 1e-300)
        P = (P + P.T) / (2 * n)
        return np.maximum(P, 1e-12)

    def fit_transform(self, X):
        X = np.asarray(X, dtype=float)
        n = len(X)
        rng = np.random.default_rng(self.seed)
        P = self._joint_prob(X)

        if self.init == "pca":
            Y = PCA(self.n_components).fit_transform(X)
            Y *= 0.0001
        else:
            Y = rng.normal(0, 1e-4, size=(n, self.n_components))

        dY = np.zeros_like(Y)
        gain = np.ones_like(Y)
        for it in range(self.n_iter):
            # attractive forces (gradient of KL on P)
            d2 = ((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
            num = 1.0 / (1.0 + d2)
            np.fill_diagonal(num, 0.0)
            Q = num / max(num.sum(), 1e-300)
            Q = np.maximum(Q, 1e-12)
            if it < 100:                       # early exaggeration phase
                PQ = self.early_exaggeration * P - Q
            else:
                PQ = P - Q
            coeff = (PQ * num)                       # (n, n)
            grad = 4.0 * (coeff.sum(1, keepdims=True) * Y - coeff @ Y)

            # adaptive gradient ascent (t-SNE's vandermaaten trick)
            gain = np.where((np.sign(grad) != np.sign(dY)), gain + 0.2, gain * 0.8)
            gain = np.maximum(gain, 1e-3)
            dY = 0.5 * dY - self.learning_rate * gain * grad
            Y += dY
            Y -= Y.mean(0, keepdims=True)

            if self.verbose and (it % 100 == 0 or it == self.n_iter - 1):
                kl = float((P * np.log(P / Q)).sum())
                print(f"    iter {it:4d}  KL {kl:.4f}")
        self.embedding_ = Y
        return Y


# --------------------------------------------------------------------------
# UMAP (simplified: fuzzy kNN graph + negative-sampling SGD layout)
# --------------------------------------------------------------------------

class UMAP:
    def __init__(self, n_components: int = 2, n_neighbors: int = 15,
                 min_dist: float = 0.1, n_epochs: int = 300,
                 learning_rate: float = 1.0, init: str = "spectral",
                 seed: int = 0, verbose: bool = False):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.init = init
        self.seed = seed
        self.verbose = verbose

    @staticmethod
    def _smooth_knn_dist(dists, k):
        """Binary search sigma so that sum of exp(-d/sigma) ~ log2(k) (umap)."""
        target = np.log2(max(k, 2))
        lo, hi = 0.0, float(dists.max()) + 1e-8
        for _ in range(64):
            sig = 0.5 * (lo + hi)
            s = np.exp(-dists / sig).sum()
            if s > target:
                lo = sig
            else:
                hi = sig
        return 0.5 * (lo + hi)

    def fit_transform(self, X):
        X = np.asarray(X, dtype=float)
        n = len(X)
        k = min(self.n_neighbors, n - 1)
        rng = np.random.default_rng(self.seed)

        d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
        idx = np.argsort(d2, axis=1)[:, 1:k + 1]
        rows = np.repeat(np.arange(n), k)
        cols = idx.ravel()
        ds = np.sqrt(d2[rows, cols])

        # fuzzy simplicial set: rho (nearest-neighbour distance) + sigma
        values = np.empty(len(rows))
        for i in range(n):
            di = ds[i * k:(i + 1) * k]
            rho = max(di[0], 1e-8)
            sigma = self._smooth_knn_dist(np.maximum(di - rho, 0.0) + 1e-8, k)
            values[i * k:(i + 1) * k] = np.exp(-(np.maximum(di - rho, 0.0)) / sigma)
        # symmetrise:  a_ij = a_ij + a_ji - a_ij*a_ji
        W = {}
        for r, c, v in zip(rows, cols, values):
            W[(r, c)] = max(W.get((r, c), 0.0), v)
        for (r, c), v in list(W.items()):
            rev = W.get((c, r), 0.0)
            W[(r, c)] = W[(c, r)] = v + rev - v * rev
        # keep 5 edges with largest weight per point (sparsify)
        by_row: dict[int, list] = {}
        for (r, c), v in W.items():
            by_row.setdefault(r, []).append((v, c))
        keep = set()
        for r, lst in by_row.items():
            lst.sort(reverse=True)
            for v, c in lst[:k]:
                keep.add((r, c))
        edges = np.array(sorted(keep))
        ew = np.array([W[(r, c)] for r, c in keep])

        # ---- layout: vectorized negative-sampling SGD on the fuzzy graph ----
        # (UMAP-style objective with a Cauchy kernel: attraction along graph
        #  edges + repulsion from negative samples; both forces are bounded)
        ei, ej = edges[:, 0], edges[:, 1]
        rho = float(np.clip(0.5 * self.min_dist + 0.05, 0.05, 0.5))
        Y = np.zeros((n, self.n_components))
        if self.init == "spectral" and n > 2 * self.n_components + 2:
            Y = self._spectral_init(n, edges, rng)
        else:
            Y = rng.normal(0, 1e-3, size=(n, self.n_components))
            for _ in range(20):              # smooth the random init over the graph
                Y[ei] += 0.1 * (Y[ej] - Y[ei])
            Y -= Y.mean(0, keepdims=True)

        eps = 1e-8
        alpha = self.learning_rate
        for epoch in range(self.n_epochs):
            G = np.zeros_like(Y)
            # attraction along the fuzzy kNN edges
            diff = Y[ei] - Y[ej]
            y = (diff ** 2).sum(-1) + eps
            z = 1.0 / (1.0 + y)
            coef = (4.0 * ew * z)[:, None] * diff
            np.add.at(G, ei, -coef)
            np.add.at(G, ej, coef)
            # repulsion from negative samples (5 per edge)
            na = rng.integers(0, n, size=5 * len(ei))
            nb = rng.integers(0, n, size=5 * len(ei))
            diff = Y[na] - Y[nb]
            y = (diff ** 2).sum(-1) + eps
            z2 = 1.0 / (1.0 + y) ** 2
            coef = (4.0 * rho * z2)[:, None] * diff
            np.add.at(G, na, coef)
            np.add.at(G, nb, -coef)
            Y += alpha * G
            Y -= Y.mean(0, keepdims=True)
            alpha = self.learning_rate * (1.0 - epoch / self.n_epochs)
            if self.verbose and (epoch % 50 == 0 or epoch == self.n_epochs - 1):
                print(f"    epoch {epoch:4d}  edges {len(ei)}")
        self.embedding_ = Y
        return Y

    @staticmethod
    def _spectral_init(n, edges, rng):
        L = np.zeros((n, n))
        for r, c in edges:
            L[r, r] += 1.0
            L[c, c] += 1.0
            L[r, c] -= 1.0
            L[c, r] -= 1.0
        try:
            _, vecs = np.linalg.eigh(L)
            Y = vecs[:, 1:3]
            if np.abs(Y).max() < 1e-12:
                raise np.linalg.LinAlgError
            return Y
        except np.linalg.LinAlgError:
            return rng.normal(0, 1e-4, size=(n, 2))


# --------------------------------------------------------------------------
# Linear Discriminant Analysis (Fisher's criterion)
# --------------------------------------------------------------------------

class LinearDiscriminantAnalysis:
    """LDA used both as a classifier and as a dim-reduction transform."""

    def __init__(self, n_components: int | None = None, shrinkage: float = 0.0):
        self.n_components = n_components
        self.shrinkage = shrinkage

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        n, d = X.shape
        self.priors_ = np.array([(y == c).mean() for c in self.classes_])
        self.means_ = np.array([X[y == c].mean(0) for c in self.classes_])

        # within-class scatter
        Sw = np.zeros((d, d))
        for c in self.classes_:
            Xc = X[y == c] - self.means_[list(self.classes_).index(c)]
            Sw += Xc.T @ Xc
        Sw /= n
        if self.shrinkage > 0:
            mu = np.trace(Sw) / d
            Sw = (1 - self.shrinkage) * Sw + self.shrinkage * mu * np.eye(d)
        self.scaling_ = np.linalg.pinv(Sw)

        # between-class scatter
        overall_mean = X.mean(0)
        Sb = np.zeros((d, d))
        for i, c in enumerate(self.classes_):
            diff = (self.means_[i] - overall_mean).reshape(-1, 1)
            Sb += (y == c).sum() * (diff @ diff.T)

        # generalised eigenproblem via whitening: evals of Sw^-1 Sb
        A = self.scaling_ @ Sb
        eigvals, eigvecs = np.linalg.eig(A)
        order = np.argsort(eigvals.real)[::-1]
        eigvals, eigvecs = eigvals.real[order], eigvecs.real[:, order]
        k = self.n_components or max(1, K - 1)
        self.explained_variance_ = eigvals[:k]
        self.coef_ = eigvecs[:, :k]                       # projection directions
        # per-class projected means (for classification)
        self.proj_means_ = self.means_ @ self.coef_
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float) @ self.coef_

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def predict_proba(self, X):
        Z = self.transform(X)                             # (n, k)
        # nearest class mean in the LDA space, with priors
        d2 = ((Z[:, None, :] - self.proj_means_[None, :, :]) ** 2).sum(-1)
        logp = -0.5 * d2 + np.log(self.priors_ + 1e-300)
        logp -= logp.max(1, keepdims=True)
        p = np.exp(logp)
        return p / p.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]

    def score(self, X, y):
        return float((self.predict(X) == np.asarray(y).ravel()).mean())