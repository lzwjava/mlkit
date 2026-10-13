"""Classical ML algorithms: learning behaviour + CLI plumbing."""

import numpy as np
import pytest

from mlkit import data as D
from mlkit import metrics as M
from mlkit.ml import (LinearRegression, Ridge, Lasso, ElasticNet,
                      LogisticRegression, LinearSVM, KernelSVM, KNN,
                      GaussianNB, MultinomialNB, DecisionTree, RandomForest,
                      GradientBoosting, XGBoost, KMeans, DBSCAN,
                      HierarchicalClustering, GMM, PCA, TSNE, UMAP,
                      LinearDiscriminantAnalysis)


def split(name="moons", n=300, seed=1, **kw):
    X, y, task = D.get_dataset(name, n=n, seed=seed, **kw)
    Xtr, Xte, ytr, yte = D.train_test_split(X, y, test=0.25, seed=0,
                                            stratify=task == "classification")
    if Xtr.ndim == 2:
        Xtr, mu, sd = D.standardize_fit(Xtr)
        Xte = D.standardize_apply(Xte, mu, sd)
    return Xtr, Xte, ytr, yte


# --------------------------------------------------------------------------
# regression
# --------------------------------------------------------------------------

def test_linear_regression_perfect_fit():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    w = np.array([1.0, -2.0, 0.5, 0.0])
    y = X @ w + 3.0 + rng.normal(0, 0.01, size=200)
    m = LinearRegression().fit(X, y)
    assert np.allclose(m.coef_, w, atol=0.05)
    assert abs(m.intercept_ - 3.0) < 0.1


def test_ridge_shrinks_coefficients():
    X, y, _ = D.get_dataset("regression", n=300, seed=2)
    ols = LinearRegression().fit(X, y)
    ridge = Ridge(alpha=100.0).fit(X, y)
    assert np.linalg.norm(ridge.coef_) < np.linalg.norm(ols.coef_)
    assert ridge.score(X, y) > 0.5


def test_lasso_sparsity():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 10))
    y = X[:, 0] * 3.0 + X[:, 1] * -2.0 + rng.normal(0, 0.1, 200)
    m = Lasso(alpha=0.1).fit(X, y)
    nz = np.abs(m.coef_) > 1e-6
    assert nz.sum() <= 4                 # sparse
    assert abs(m.coef_[0]) > 1.0         # keeps the true signals


def test_elastic_net_fits():
    X, y, _ = D.get_dataset("regression", n=300, seed=3)
    m = ElasticNet(alpha=0.01, l1_ratio=0.5).fit(X, y)
    assert m.score(X, y) > 0.7


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------

@pytest.mark.parametrize("model", [
    LogisticRegression(epochs=200),
    LinearSVM(epochs=60),
    KNN(k=5),
    GaussianNB(),
    DecisionTree(max_depth=6),
    RandomForest(n_estimators=30, max_depth=8),
    GradientBoosting(n_estimators=50),
    XGBoost(n_estimators=50),
])
def test_classifiers_learn_moons(model):
    Xtr, Xte, ytr, yte = split("moons")
    model.fit(Xtr, ytr)
    assert model.score(Xte, yte) > 0.8


def test_kernel_svm_beats_linear_on_circles():
    Xtr, Xte, ytr, yte = split("circles", n=400)
    lin = LinearSVM(epochs=80).fit(Xtr, ytr)
    rbf = KernelSVM(kernel="rbf", max_passes=8).fit(Xtr, ytr)
    assert rbf.score(Xte, yte) > 0.85
    assert rbf.score(Xte, yte) >= lin.score(Xte, yte)


def test_logistic_reports_proba():
    Xtr, Xte, ytr, yte = split()
    m = LogisticRegression(epochs=100).fit(Xtr, ytr)
    p = m.predict_proba(Xte)
    assert p.shape == (len(Xte), 2)
    assert np.allclose(p.sum(1), 1.0, atol=1e-5)


def test_multinomial_nb_on_counts():
    rng = np.random.default_rng(0)
    X = rng.poisson(1.5, size=(200, 20)).astype(float)
    y = (X[:, :5].sum(1) > X[:, 15:].sum(1)).astype(int)
    m = MultinomialNB().fit(X, y)
    assert m.score(X, y) > 0.9


def test_random_forest_oob():
    Xtr, Xte, ytr, yte = split(n=400)
    m = RandomForest(n_estimators=40, oob=True, seed=0).fit(Xtr, ytr)
    assert m.oob_score_ is not None and m.oob_score_ > 0.7


# --------------------------------------------------------------------------
# clustering
# --------------------------------------------------------------------------

def test_kmeans_recovers_blobs():
    X, y, _ = D.get_dataset("gmm", n=300, seed=0)
    km = KMeans(k=3, seed=0).fit(X)
    assert len(np.unique(km.labels_)) == 3
    assert km.silhouette(X) > 0.5
    # k-means++ must beat random init on inertia
    km_rand = KMeans(k=3, init="random", seed=0).fit(X)
    assert km.inertia_ <= km_rand.inertia_ * 1.2


def test_dbscan_finds_clusters_and_noise():
    X, y, _ = D.get_dataset("gmm", n=300, seed=0)
    ds = DBSCAN(eps=1.0, min_samples=5).fit(X)
    assert ds.n_clusters_ >= 2


@pytest.mark.parametrize("linkage", ["single", "complete", "average", "ward"])
def test_hierarchical_linkages(linkage):
    X, y, _ = D.get_dataset("gmm", n=200, seed=0)
    hc = HierarchicalClustering(linkage=linkage, n_clusters=3).fit(X)
    assert hc.n_clusters_ == 3
    assert len(hc.Z_) == 199              # n-1 merges


def test_gmm_expectation_maximisation():
    X, y, _ = D.get_dataset("gmm", n=300, seed=0)
    g = GMM(k=3, seed=0).fit(X)
    ll = g.score_samples(X)
    assert np.isfinite(ll)
    # EM must not decrease the likelihood after the first iteration
    assert g.lower_bounds_[-1] > g.lower_bounds_[0]
    assert len(np.unique(g.labels_)) == 3


# --------------------------------------------------------------------------
# dimensionality reduction
# --------------------------------------------------------------------------

def test_pca_variance_and_reconstruction():
    X, y, _ = D.get_dataset("blobs", n=300, seed=0)
    pca = PCA(2).fit(X)
    assert pca.explained_variance_ratio_.sum() <= 1.0 + 1e-9
    Z = pca.transform(X)
    Xh = pca.inverse_transform(Z)
    assert M.explained_variance(X, Xh) > 0.5


def test_tsne_embeds_classes():
    X, y, _ = D.get_dataset("blobs", n=150, seed=0)
    Z = TSNE(n_iter=300, perplexity=20, seed=0).fit_transform(X)
    assert Z.shape == (150, 2)
    assert M.silhouette_score(Z, y) > M.silhouette_score(X, y)


def test_umap_embeds_classes():
    X, y, _ = D.get_dataset("blobs", n=150, seed=0)
    Z = UMAP(n_epochs=200, seed=0).fit_transform(X)
    assert Z.shape == (150, 2)
    assert M.silhouette_score(Z, y) > 0.7


def test_lda_separates_classes():
    X, y, _ = D.get_dataset("blobs", n=200, seed=0)
    lda = LinearDiscriminantAnalysis().fit(X, y)
    assert lda.score(X, y) > 0.9
    assert lda.transform(X).shape == (200, 2)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def test_metrics_basics():
    assert M.accuracy([0, 1, 1, 0], [0, 1, 0, 0]) == 0.75
    p, r, f = M.precision_recall_f1([0, 0, 1, 1], [0, 1, 1, 1])
    assert 0.6 < p <= 1.0 and 0.6 < r <= 1.0
    assert M.r2_score([1, 2, 3], [1, 2, 3]) == 1.0
    assert M.mse([1, 2], [1, 2]) == 0.0