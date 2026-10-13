"""Clustering algorithms: K-Means, DBSCAN, Hierarchical (agglomerative), GMM."""

from __future__ import annotations

import numpy as np

from ..metrics import silhouette_score


# --------------------------------------------------------------------------
# K-Means (k-means++ init, Lloyd iterations)
# --------------------------------------------------------------------------

class KMeans:
    def __init__(self, k: int = 3, init: str = "k-means++", n_init: int = 10,
                 max_iter: int = 300, tol: float = 1e-4, seed: int = 0,
                 verbose: bool = False):
        self.k = k
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.verbose = verbose

    # ------------------------------------------------------------------
    @staticmethod
    def _dist2(X, C):
        return np.maximum(((X[:, None, :] - C[None, :, :]) ** 2).sum(-1), 0.0)

    def _init_centers(self, X, rng):
        n = len(X)
        if self.init == "random":
            return X[rng.choice(n, size=self.k, replace=False)].copy()
        # k-means++
        centers = [X[rng.integers(0, n)]]
        for _ in range(1, self.k):
            d2 = self._dist2(X, np.array(centers)).min(1)
            probs = d2 / max(d2.sum(), 1e-12)
            centers.append(X[rng.choice(n, p=probs)])
        return np.array(centers)

    def _lloyd(self, X, rng):
        C = self._init_centers(X, rng)
        labels, inertia = None, np.inf
        for it in range(self.max_iter):
            d2 = self._dist2(X, C)
            labels = d2.argmin(1)
            new_inertia = d2[np.arange(len(X)), labels].sum()
            # update (handle empty clusters by splitting off the worst point)
            for c in range(self.k):
                m = labels == c
                if m.any():
                    C[c] = X[m].mean(0)
                else:
                    worst = d2.max(1).argmax()
                    C[c] = X[worst]
                    labels[worst] = c
            if self.verbose:
                print(f"    iter {it:3d}  inertia {new_inertia:.3f}")
            if abs(inertia - new_inertia) < self.tol:
                inertia = new_inertia
                break
            inertia = new_inertia
        return C, labels, float(inertia), it + 1

    def fit(self, X, seed=None):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.seed if seed is None else seed)
        best = None
        for _ in range(self.n_init):
            C, labels, iner, n_iter = self._lloyd(X, rng)
            if best is None or iner < best[2]:
                best = (C, labels, iner, n_iter)
        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = best
        return self

    def predict(self, X):
        d2 = self._dist2(np.asarray(X, dtype=float), self.cluster_centers_)
        return d2.argmin(1)

    def fit_predict(self, X, **kw):
        return self.fit(X, **kw).labels_

    def silhouette(self, X):
        return silhouette_score(X, self.labels_)


# --------------------------------------------------------------------------
# DBSCAN (density-based clustering with grid bucketing)
# --------------------------------------------------------------------------

class DBSCAN:
    def __init__(self, eps: float = 0.5, min_samples: int = 5,
                 metric: str = "euclidean"):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = len(X)
        eps2 = self.eps ** 2
        # grid bucketing so neighbours are found quickly
        cell = max(self.eps, 1e-9)
        buckets: dict[tuple, list[int]] = {}
        for i, x in enumerate(X):
            key = tuple(np.floor(x / cell).astype(int))
            buckets.setdefault(key, []).append(i)

        def neighbours(i):
            key = np.floor(X[i] / cell).astype(int)
            out = []
            for off in _offsets(X.shape[1]):
                k = tuple(key + np.array(off))
                for j in buckets.get(k, ()):
                    d2 = ((X[i] - X[j]) ** 2).sum()
                    if d2 <= eps2:
                        out.append(j)
            return out

        labels = np.full(n, -2)              # -2 = unvisited, -1 = noise
        cluster = 0
        for i in range(n):
            if labels[i] != -2:
                continue
            nbrs = neighbours(i)
            if len(nbrs) < self.min_samples:
                labels[i] = -1               # noise
                continue
            labels[i] = cluster
            seeds = list(nbrs)
            while seeds:
                j = seeds.pop()
                if labels[j] == -1:
                    labels[j] = cluster      # border point
                if labels[j] != -2:
                    continue
                labels[j] = cluster
                jn = neighbours(j)
                if len(jn) >= self.min_samples:
                    seeds.extend(jn)
            cluster += 1
        self.labels_ = labels
        self.core_sample_indices_ = np.where(
            np.array([len(neighbours(i)) >= self.min_samples
                      for i in range(n)]))[0]
        self.n_clusters_ = cluster
        self.n_noise_ = int((labels == -1).sum())
        return self


def _offsets(d: int):
    """27 (or 3^d) neighbour-cell offsets in d dimensions."""
    import itertools
    return list(itertools.product((-1, 0, 1), repeat=d))


# --------------------------------------------------------------------------
# Agglomerative hierarchical clustering
# --------------------------------------------------------------------------

class HierarchicalClustering:
    def __init__(self, linkage: str = "ward", metric: str = "euclidean",
                 n_clusters: int = 2):
        self.linkage = linkage          # 'single' | 'complete' | 'average' | 'ward'
        self.metric = metric
        self.n_clusters = n_clusters

    @staticmethod
    def _pairwise(X):
        d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
        return np.sqrt(np.maximum(d2, 0.0))

    def fit(self, X):
        """Agglomerative clustering with Lance-Williams distance updates.

        The (2n-1)x(2n-1) distance matrix is maintained incrementally, so
        each merge costs O(n) vectorised ops instead of O(n^2) rescans.
        """
        X = np.asarray(X, dtype=float)
        n = len(X)
        if n == 1:
            self.Z_ = []
            self.labels_ = np.zeros(1, dtype=int)
            self.n_clusters_ = 1
            return self
        d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
        D = np.sqrt(np.maximum(d2, 0.0))
        D = np.pad(D, ((0, n - 1), (0, n - 1)))       # slots for new clusters
        size = np.zeros(2 * n - 1, dtype=np.int64)
        size[:n] = 1
        active = list(range(n))
        next_id = n
        self.Z_ = []                                  # [a, b, dist, size]

        while len(active) > 1:
            act = np.array(active)
            sub = D[np.ix_(act, act)]
            np.fill_diagonal(sub, np.inf)
            i, j = divmod(int(np.argmin(sub)), len(act))
            a, b = int(act[i]), int(act[j])
            sa, sb = int(size[a]), int(size[b])
            self.Z_.append([a, b, float(D[a, b]), sa + sb])

            others = np.array([c for c in active if c != a and c != b],
                              dtype=np.int64)
            sc = size[others].astype(float)
            da, db = D[a, others], D[b, others]
            if self.linkage == "single":
                newd = np.minimum(da, db)
            elif self.linkage == "complete":
                newd = np.maximum(da, db)
            elif self.linkage == "average":
                newd = (sa * da + sb * db) / (sa + sb)
            else:                                      # ward
                newd = np.sqrt(np.maximum(
                    ((sa + sc) * da ** 2 + (sb + sc) * db ** 2
                     - sc * D[a, b] ** 2) / (sa + sb + sc), 0.0))
            D[next_id, others] = newd
            D[others, next_id] = newd
            size[next_id] = sa + sb
            active = [c for c in active if c != a and c != b] + [next_id]
            next_id += 1

        # cut the tree at n_clusters (union-find over the first n-k merges)
        self.n_clusters_ = self.n_clusters
        if self.n_clusters >= n:
            self.labels_ = np.arange(n)
            self.n_clusters_ = n
            return self
        parent = list(range(2 * n - 1))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, row in enumerate(self.Z_[:n - self.n_clusters]):
            a, b = int(row[0]), int(row[1])
            new = n + i                     # id of the merged cluster
            parent[find(a)] = new
            parent[find(b)] = new
        roots = {}
        labels = np.empty(n, dtype=int)
        for i in range(n):
            r = find(i)
            labels[i] = roots.setdefault(r, len(roots))
        self.labels_ = labels
        self.n_clusters_ = len(roots)
        return self

    def fit_predict(self, X, **kw):
        return self.fit(X, **kw).labels_

    def silhouette(self, X):
        return silhouette_score(X, self.labels_)


# --------------------------------------------------------------------------
# DBSCAN (density-based clustering with grid bucketing)
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Gaussian Mixture Models (EM)
# --------------------------------------------------------------------------

class GMM:
    def __init__(self, k: int = 3, covariance: str = "full", max_iter: int = 200,
                 tol: float = 1e-5, reg_covar: float = 1e-6, seed: int = 0,
                 verbose: bool = False):
        self.k = k
        self.covariance = covariance      # 'full' | 'diag'
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.seed = seed
        self.verbose = verbose

    def _e_step(self, X):
        n, d = X.shape
        log_p = np.empty((n, self.k))
        for j in range(self.k):
            diff = X - self.means_[j]
            if self.covariance == "full":
                L = self._prec[j]                      # precision matrix
                maha = np.einsum('ni,ij,nj->n', diff, L, diff)
                log_det = self._logdet[j]
            else:
                v = self.vars_[j]
                maha = ((diff ** 2) / v).sum(1)
                log_det = np.log(v).sum()
            log_p[:, j] = (np.log(self.weights_[j] + 1e-300)
                           - 0.5 * (d * np.log(2 * np.pi) + log_det + maha))
        m = log_p.max(1, keepdims=True)
        log_norm = np.log(np.exp(log_p - m).sum(1, keepdims=True)) + m
        log_resp = log_p - log_norm
        return log_resp, float(log_norm.mean())

    def fit(self, X, seed=None):
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        rng = np.random.default_rng(self.seed if seed is None else seed)

        # init with k-means++
        km = KMeans(k=self.k, seed=int(rng.integers(0, 1 << 30)), n_init=3)
        labels = km.fit_predict(X)
        self.means_ = np.array([X[labels == j].mean(0) if (labels == j).any()
                                else X[rng.integers(0, n)] for j in range(self.k)])
        self.weights_ = np.array([(labels == j).mean() + 1e-9
                                  for j in range(self.k)])
        self.weights_ /= self.weights_.sum()
        if self.covariance == "full":
            global_var = np.cov(X.T) + self.reg_covar * np.eye(d)
            self.covariances_ = np.array([global_var.copy() for _ in range(self.k)])
        else:
            v = X.var(0) + self.reg_covar
            self.covariances_ = np.array([v.copy() for _ in range(self.k)])

        prev_ll = -np.inf
        self.lower_bounds_ = []
        for it in range(self.max_iter):
            self._cache()
            log_resp, ll = self._e_step(X)
            self.lower_bounds_.append(ll)
            # M-step
            resp = np.exp(log_resp)
            Nk = resp.sum(0) + 1e-10
            self.means_ = resp.T @ X / Nk[:, None]
            self.weights_ = Nk / n
            for j in range(self.k):
                diff = X - self.means_[j]
                if self.covariance == "full":
                    wd = resp[:, j][:, None] * diff
                    self.covariances_[j] = wd.T @ diff / Nk[j]
                    self.covariances_[j].flat[:: d + 1] += self.reg_covar
                else:
                    self.covariances_[j] = (resp[:, j] * (diff ** 2)).sum(0) / Nk[j]
                    self.covariances_[j] += self.reg_covar
            if self.verbose and (it % 20 == 0 or it == self.max_iter - 1):
                print(f"    iter {it:4d}  log-lik {ll:.4f}")
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll
        self.n_iter_ = it + 1
        self._cache()
        self.labels_ = self._e_step(X)[0].argmax(1)
        return self

    def _cache(self):
        d = self.means_[0].size
        self._logdet, self._prec = [], []
        for j in range(self.k):
            cov = self.covariances_[j]
            if self.covariance == "full":
                sign, ld = np.linalg.slogdet(cov)
                self._logdet.append(ld if sign > 0 else 0.0)
                self._prec.append(np.linalg.pinv(cov))
            else:
                self._logdet.append(0.0)
                self._prec.append(None)

    def predict_proba(self, X):
        self._cache()
        return np.exp(self._e_step(np.asarray(X, dtype=float))[0])

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

    def score_samples(self, X):
        self._cache()
        return self._e_step(np.asarray(X, dtype=float))[1]