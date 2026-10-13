"""Decision trees (CART) and Random Forests.

Both support classification (gini / entropy) and regression (mse).
The tree is grown greedily; RandomForest adds bootstrap sampling +
feature subsampling (sqrt/log2 style) per split.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# node helpers
# --------------------------------------------------------------------------

class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self, feature=-1, threshold=0.0, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value          # leaf prediction (class or mean)

    @property
    def is_leaf(self):
        return self.left is None and self.right is None


def _gini(y, w=None):
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(1.0 - (p ** 2).sum())


def _entropy(y):
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _mse(y):
    if len(y) == 0:
        return 0.0
    return float(np.var(y))


def _impurity(y, criterion):
    if criterion == "gini":
        return _gini(y)
    if criterion == "entropy":
        return _entropy(y)
    return _mse(y)


def _leaf_value(y, task):
    if task == "classification":
        vals, counts = np.unique(y, return_counts=True)
        return vals[counts.argmax()]
    return float(np.mean(y))


def _best_split(X, y, feat_idx, criterion, task, min_samples_leaf, rng=None):
    """Exhaustive threshold search over a random subset of features."""
    n = len(y)
    parent = _impurity(y, criterion)
    best_gain, best = 0.0, None
    for j in feat_idx:
        col = X[:, j]
        order = np.argsort(col, kind="stable")
        xs, ys = col[order], y[order]
        # candidate thresholds: midpoints between distinct values
        distinct = np.where(np.diff(xs) > 1e-12)[0] + 1
        if distinct.size == 0:
            continue
        # subsample candidate thresholds for speed on large data
        if distinct.size > 64:
            pick = rng.choice(distinct, size=64, replace=False)
            cut_idx = np.sort(pick)
        else:
            cut_idx = distinct
        # cumulative impurity via prefix sums
        if task == "classification":
            classes = np.unique(y)
            ci = {c: i for i, c in enumerate(classes)}
            ynum = np.array([ci[v] for v in ys])
            K = len(classes)
            onehot = np.zeros((n, K))
            onehot[np.arange(n), ynum] = 1.0
            cum = np.cumsum(onehot, axis=0)
        else:
            cum_y = np.cumsum(ys)
            cum_y2 = np.cumsum(ys ** 2)
        for k in cut_idx:
            nL = k
            nR = n - k
            if nL < min_samples_leaf or nR < min_samples_leaf:
                continue
            if task == "classification":
                cL = cum[k - 1]
                cR = cum[-1] - cL
                pL, pR = cL / nL, cR / nR
                impL = float(1.0 - (pL ** 2).sum()) if criterion == "gini" \
                    else float(-(pL[pL > 0] * np.log2(pL[pL > 0])).sum())
                impR = float(1.0 - (pR ** 2).sum()) if criterion == "gini" \
                    else float(-(pR[pR > 0] * np.log2(pR[pR > 0])).sum())
            else:
                sL, s2L = cum_y[k - 1], cum_y2[k - 1]
                sR, s2R = cum_y[-1] - sL, cum_y2[-1] - s2L
                impL = s2L / nL - (sL / nL) ** 2
                impR = s2R / nR - (sR / nR) ** 2
            gain = parent - (nL * impL + nR * impR) / n
            if gain > best_gain + 1e-12:
                thr = 0.5 * (xs[k - 1] + xs[k])
                best_gain, best = gain, (j, thr)
    return best_gain, best


# --------------------------------------------------------------------------
# decision tree
# --------------------------------------------------------------------------

class DecisionTree:
    def __init__(self, task: str = "classification", criterion: str = "gini",
                 max_depth: int | None = None, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, max_features: int | str | None = None,
                 seed: int = 0):
        self.task = task
        self.criterion = criterion if task == "classification" else "mse"
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.seed = seed
        self.root_: _Node | None = None
        self.n_features_: int = 0

    # ------------------------------------------------------------------
    def _n_feat(self, rng):
        d = self.n_features_
        if self.max_features is None:
            return d
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(d)))
        if self.max_features == "log2":
            return max(1, int(np.log2(d) + 1))
        if isinstance(self.max_features, float) and self.max_features <= 1:
            return max(1, int(self.max_features * d))
        return max(1, min(int(self.max_features), d))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        y = y.ravel()
        if self.task == "classification":
            y = y.astype(int) if np.issubdtype(y.dtype, np.number) else y
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y) if self.task == "classification" else None
        rng = np.random.default_rng(self.seed)
        self.root_ = self._build(X, y, depth=0, rng=rng)
        return self

    def _build(self, X, y, depth, rng):
        node = _Node(value=_leaf_value(y, self.task))
        n = len(y)
        if self.max_depth is not None and depth >= self.max_depth:
            return node
        if n < self.min_samples_split:
            return node
        if self.task == "classification" and len(np.unique(y)) == 1:
            return node
        if self.task == "regression" and np.var(y) < 1e-12:
            return node

        feat_idx = rng.permutation(self.n_features_)[: self._n_feat(rng)]
        gain, split = _best_split(X, y, feat_idx, self.criterion, self.task,
                                  self.min_samples_leaf, rng)
        if split is None or gain <= 1e-12:
            return node
        j, thr = split
        left_m = X[:, j] <= thr
        if left_m.sum() == 0 or (~left_m).sum() == 0:
            return node
        node.feature, node.threshold = j, thr
        node.left = self._build(X[left_m], y[left_m], depth + 1, rng)
        node.right = self._build(X[~left_m], y[~left_m], depth + 1, rng)
        if node.left.is_leaf and node.right.is_leaf and \
           node.left.value == node.right.value:
            return node.left
        return node

    # ------------------------------------------------------------------
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._traverse(x, self.root_) for x in X])

    @staticmethod
    def _traverse(x, node):
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def score(self, X, y):
        y = np.asarray(y)
        pred = self.predict(X)
        if self.task == "regression":
            ss = ((y.ravel() - pred) ** 2).sum()
            st = ((y.ravel() - y.mean()) ** 2).sum()
            return float(1 - ss / max(st, 1e-12))
        return float((pred == y.ravel()).mean())

    def depth(self) -> int:
        def _d(node):
            if node is None or node.is_leaf:
                return 0
            return 1 + max(_d(node.left), _d(node.right))
        return _d(self.root_)


# --------------------------------------------------------------------------
# random forest (bootstrap aggregation + feature subsampling)
# --------------------------------------------------------------------------

class RandomForest:
    def __init__(self, n_estimators: int = 100, task: str = "classification",
                 criterion: str = "gini", max_depth: int | None = None,
                 min_samples_leaf: int = 1, max_features: str | int | None = "sqrt",
                 bootstrap: bool = True, oob: bool = False, seed: int = 0,
                 verbose: bool = False):
        self.n_estimators = n_estimators
        self.task = task
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob = oob
        self.seed = seed
        self.verbose = verbose
        self.trees_: list[DecisionTree] = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        n = len(X)
        rng = np.random.default_rng(self.seed)
        self.classes_ = np.unique(y) if self.task == "classification" else None
        oob_votes = np.zeros((n, len(self.classes_))) if self.oob and \
            self.task == "classification" else None

        self.trees_ = []
        for t in range(self.n_estimators):
            if self.bootstrap:
                idx = rng.integers(0, n, size=n)
            else:
                idx = rng.permutation(n)[:n]
            tree = DecisionTree(task=self.task, criterion=self.criterion,
                                max_depth=self.max_depth,
                                min_samples_leaf=self.min_samples_leaf,
                                max_features=self.max_features, seed=self.seed + t)
            tree.fit(X[idx], y[idx])
            self.trees_.append(tree)

            if oob_votes is not None:
                in_bag = np.zeros(n, dtype=bool)
                in_bag[np.unique(idx)] = True
                oob = np.where(~in_bag)[0]
                if len(oob):
                    pred = tree.predict(X[oob])
                    cls_idx = {c: i for i, c in enumerate(self.classes_)}
                    for row, p in zip(oob, pred):
                        oob_votes[row, cls_idx[p]] += 1
            if self.verbose and (t + 1) % 20 == 0:
                print(f"    tree {t + 1}/{self.n_estimators}")

        if oob_votes is not None:
            valid = oob_votes.sum(1) > 0
            acc = (oob_votes[valid].argmax(1) ==
                   np.searchsorted(self.classes_, y[valid]))
            self.oob_score_ = float(acc.mean()) if valid.any() else None
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self.task == "regression":
            return np.mean([t.predict(X) for t in self.trees_], axis=0)
        votes = np.zeros((len(X), len(self.classes_)))
        cls_idx = {c: i for i, c in enumerate(self.classes_)}
        for tree in self.trees_:
            pred = tree.predict(X)
            for i, p in enumerate(pred):
                votes[i, cls_idx[p]] += 1
        return self.classes_[votes.argmax(1)]

    def score(self, X, y):
        y = np.asarray(y).ravel()
        pred = self.predict(X)
        if self.task == "regression":
            ss = ((y - pred) ** 2).sum()
            st = ((y - y.mean()) ** 2).sum()
            return float(1 - ss / max(st, 1e-12))
        return float((pred == y).mean())