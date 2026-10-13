"""Gradient boosting ensembles.

* GradientBoosting -> classic Friedman GBM (steepest descent on the loss,
                      regression trees fit to negative gradients)
* XGBoost          -> second-order (Newton) boosting: splits scored with
                      grad/hess statistics, L2 (lambda) + L1 (alpha) leaf
                      regularisation and minimum child weight

Both handle regression (squared error) and classification (logistic loss).
"""

from __future__ import annotations

import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


class _BoostTree:
    """Regression tree fitted to (gradients, hessians) or pseudo-residuals."""

    def __init__(self, depth, min_child_weight, lam, gamma, second_order,
                 min_samples_leaf, max_features, rng):
        self.depth = depth
        self.min_child_weight = min_child_weight
        self.lam = lam
        self.gamma = gamma
        self.second_order = second_order
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.rng = rng
        self.feature = -1
        self.threshold = 0.0
        self.left = None
        self.right = None
        self.value = 0.0

    @property
    def is_leaf(self):
        return self.left is None

    def _leaf(self, g, h):
        if self.second_order:
            return float(-g.sum() / (h.sum() + self.lam))
        return float(-g.mean())            # pseudo-residual = -dL/dF

    def fit(self, X, g, h, depth=0):
        n, d = X.shape
        self.value = self._leaf(g, h)
        if depth >= self.depth or n < 2 * self.min_samples_leaf:
            return self

        d_feat = d if self.max_features is None else \
            max(1, min(int(self.max_features), d))
        feat_idx = self.rng.permutation(d)[:d_feat]

        best_gain, best = 0.0, None
        Gtot, Htot = g.sum(), h.sum() if h is not None else 0.0
        for j in feat_idx:
            col = X[:, j]
            order = np.argsort(col, kind="stable")
            xs = col[order]
            gs, hs = g[order], h[order]
            distinct = np.where(np.diff(xs) > 1e-12)[0] + 1
            if distinct.size == 0:
                continue
            if distinct.size > 48:
                cut = np.sort(self.rng.choice(distinct, size=48, replace=False))
            else:
                cut = distinct
            cG, cH = np.cumsum(gs), np.cumsum(hs)
            for k in cut:
                if k < self.min_samples_leaf or n - k < self.min_samples_leaf:
                    continue
                GL, HL = cG[k - 1], cH[k - 1]
                GR, HR = Gtot - GL, Htot - HL
                if self.second_order:
                    gain = 0.5 * (GL ** 2 / (HL + self.lam) +
                                  GR ** 2 / (HR + self.lam) -
                                  (GL + GR) ** 2 / (HL + HR + self.lam)) \
                        - self.gamma
                else:
                    # variance reduction on pseudo-residuals
                    imp = g.var()
                    varL = gs[:k].var() if k > 1 else 0.0
                    varR = gs[k:].var() if n - k > 1 else 0.0
                    gain = imp - (k * varL + (n - k) * varR) / n
                if HL < self.min_child_weight or HR < self.min_child_weight:
                    continue
                if gain > best_gain:
                    best_gain, best = gain, (j, 0.5 * (xs[k - 1] + xs[k]))

        if best is None or best_gain <= 1e-10:
            return self
        j, thr = best
        m = X[:, j] <= thr
        if m.sum() == 0 or (~m).sum() == 0:
            return self
        self.feature, self.threshold = j, thr
        self.left = _BoostTree(self.depth, self.min_child_weight, self.lam,
                               self.gamma, self.second_order,
                               self.min_samples_leaf, self.max_features, self.rng)
        self.right = _BoostTree(self.depth, self.min_child_weight, self.lam,
                                self.gamma, self.second_order,
                                self.min_samples_leaf, self.max_features, self.rng)
        self.left.fit(X[m], g[m], h[m], depth + 1)
        self.right.fit(X[~m], g[~m], h[~m], depth + 1)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        out = np.empty(len(X))
        for i, x in enumerate(X):
            node = self
            while not node.is_leaf:
                node = node.left if x[node.feature] <= node.threshold else node.right
            out[i] = node.value
        return out


class _BaseBoosting:
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3, subsample: float = 1.0, reg_lambda: float = 1.0,
                 reg_alpha: float = 0.0, min_child_weight: float = 1.0,
                 min_samples_leaf: int = 1, max_features: int | str | None = None,
                 task: str = "classification", second_order: bool = False,
                 seed: int = 0, verbose: bool = False, init=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.min_child_weight = min_child_weight
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.task = task
        self.second_order = second_order
        self.seed = seed
        self.verbose = verbose
        self.init = init                       # initial prediction (or None)
        self.trees_: list[_BoostTree] = []
        self.loss_history_: list[float] = []

    # gradient / hessian of the training loss w.r.t. the raw score ---------
    def _grad_hess(self, y, raw):
        if self.task == "regression":
            g = raw - y
            h = np.ones_like(g)
            loss = 0.5 * float(np.mean(g ** 2))
            return g, h, loss
        p = _sigmoid(raw)
        eps = 1e-12
        loss = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
        g = p - y
        h = np.maximum(p * (1 - p), 1e-8)
        return g, h, loss

    def _raw_predict(self, X, trees=None):
        trees = self.trees_ if trees is None else trees
        out = np.full(len(X), self.base_score_, dtype=float)
        for t in trees:
            out += self.learning_rate * t.predict(X)
        return out

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        if self.task == "classification":
            self.classes_ = np.unique(y)
            if len(self.classes_) != 2:
                raise ValueError("boosting classifiers here are binary "
                                 f"(got {len(self.classes_)} classes)")
            y = np.where(y == self.classes_[1], 1.0, 0.0)
        rng = np.random.default_rng(self.seed)

        # initial constant model
        if self.init is not None:
            self.base_score_ = float(self.init)
        elif self.task == "regression":
            self.base_score_ = float(y.mean())
        else:
            p = np.clip(y.mean(), 1e-6, 1 - 1e-6)
            self.base_score_ = float(np.log(p / (1 - p)))

        raw = np.full(len(X), self.base_score_)
        self.trees_ = []
        self.loss_history_ = []
        n = len(X)
        for m in range(self.n_estimators):
            g, h, loss = self._grad_hess(y, raw)
            self.loss_history_.append(loss)
            # optional row subsampling (stochastic gradient boosting)
            if self.subsample < 1.0:
                mask = rng.random(n) < self.subsample
                if mask.sum() < 4:
                    mask = np.ones(n, dtype=bool)
            else:
                mask = slice(None)
            tree = _BoostTree(self.max_depth, self.min_child_weight,
                              self.reg_lambda, self.reg_alpha if self.second_order else 0.0,
                              self.second_order, self.min_samples_leaf,
                              self._max_feat_n(), rng)
            tree.fit(X[mask], g[mask], h[mask])
            self.trees_.append(tree)
            raw += self.learning_rate * tree.predict(X)
            if self.verbose and ((m + 1) % 20 == 0 or m == self.n_estimators - 1):
                print(f"    iter {m + 1:4d}  train-loss {loss:.5f}")
        g, h, loss = self._grad_hess(y, raw)
        self.loss_history_.append(loss)
        self.train_loss_ = loss
        return self

    def _max_feat_n(self):
        if self.max_features is None:
            return None
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(self.n_features_)))
        if self.max_features == "log2":
            return max(1, int(np.log2(self.n_features_) + 1))
        return int(self.max_features)

    @property
    def n_features_(self):
        return getattr(self, "_n_features", 1)

    @n_features_.setter
    def n_features_(self, v):
        self._n_features = v

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        return self._raw_predict(X)

    def predict(self, X):
        raw = self.decision_function(X)
        if self.task == "regression":
            return raw
        return np.where(_sigmoid(raw) >= 0.5,
                        self.classes_[1], self.classes_[0])

    def predict_proba(self, X):
        p = _sigmoid(self.decision_function(X))
        return np.stack([1 - p, p], axis=1)

    def score(self, X, y):
        y = np.asarray(y).ravel()
        pred = self.predict(X)
        if self.task == "regression":
            ss = ((y - pred) ** 2).sum()
            st = ((y - y.mean()) ** 2).sum()
            return float(1 - ss / max(st, 1e-12))
        return float((pred == y).mean())


class GradientBoosting(_BaseBoosting):
    """Friedman-style gradient boosting (first-order, pseudo-residuals)."""

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 subsample=1.0, task="classification", seed=0, verbose=False,
                 **kw):
        super().__init__(n_estimators=n_estimators, learning_rate=learning_rate,
                         max_depth=max_depth, subsample=subsample, task=task,
                         seed=seed, verbose=verbose, second_order=False, **kw)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.n_features_ = X.shape[1]
        return super().fit(X, y)


class XGBoost(_BaseBoosting):
    """Newton boosting: exact split gain with grad/hessian stats,
    L2 (reg_lambda) and L1 (reg_alpha) regularisation, min_child_weight."""

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 subsample=1.0, reg_lambda=1.0, reg_alpha=0.0,
                 min_child_weight=1.0, task="classification", seed=0,
                 verbose=False, **kw):
        super().__init__(n_estimators=n_estimators, learning_rate=learning_rate,
                         max_depth=max_depth, subsample=subsample,
                         reg_lambda=reg_lambda, reg_alpha=reg_alpha,
                         min_child_weight=min_child_weight, task=task, seed=seed,
                         verbose=verbose, second_order=True, **kw)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.n_features_ = X.shape[1]
        return super().fit(X, y)