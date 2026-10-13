"""mlkit command-line interface.

Groups (matching the ML / Deep Learning / LLM breakdown of the README):

* ``mlkit data``    – list / generate datasets
* ``mlkit ml``      – classical machine-learning algorithms
* ``mlkit bench``   – benchmark several algorithms on one dataset
* ``mlkit dl``      – deep learning (MLP/CNN/RNN/ViT/GAN/VAE/diffusion,
                      optimizers, gradient checking)
* ``mlkit llm``     – large language models (train, generate, LoRA,
                      quantization, flash attention, KV cache, speculative
                      decoding, RAG, agents)
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from . import __version__
from . import data as D
from . import metrics as M
from .utils import kv, section, set_seed, table, curve, series, fmt_time


# ==========================================================================
# shared helpers
# ==========================================================================


def _sec(args, title):
    """Print a section header unless machine-readable output was requested."""
    if not getattr(args, "json", False):
        section(title)


def add_data_args(p, default_dataset="moons"):
    g = p.add_argument_group("data")
    g.add_argument("--dataset", default=default_dataset,
                   help=f"synthetic dataset ({sorted(D.DATASETS)})")
    g.add_argument("--csv", default=None, help="CSV file (alternative)")
    g.add_argument("--target", default=None, help="CSV target column name")
    g.add_argument("--n", type=int, default=600, help="samples (synthetic)")
    g.add_argument("--test", type=float, default=0.25, help="test fraction")
    g.add_argument("--no-standardize", action="store_true")
    return p


def load_supervised(args, dataset_kw=None):
    """Return (Xtr, Xte, ytr, yte, task)."""
    if args.csv:
        (Xtr, Xte, ytr, yte), task, target = D.load_csv(
            args.csv, target=args.target, test=args.test,
            standardize=not args.no_standardize, seed=args.seed)
        return Xtr, Xte, ytr, yte, task
    kw = dict(dataset_kw or {})
    X, y, task = D.get_dataset(args.dataset, n=args.n, seed=args.seed, **kw)
    strat = task == "classification"
    Xtr, Xte, ytr, yte = D.train_test_split(X, y, test=args.test,
                                            seed=args.seed, stratify=strat)
    if not args.no_standardize and Xtr.ndim == 2:
        Xtr, mu, sd = D.standardize_fit(Xtr)
        Xte = D.standardize_apply(Xte, mu, sd)
    return Xtr, Xte, ytr, yte, task


def load_unsupervised(args, dataset_kw=None):
    if args.csv:
        (Xtr, _, _, _), task, _ = D.load_csv(args.csv, target=args.target,
                                             test=0.0, seed=args.seed)
        return Xtr
    kw = dict(dataset_kw or {})
    X, _, _ = D.get_dataset(args.dataset, n=args.n, seed=args.seed, **kw)
    return X


def emit(result: dict, args, title: str = "results"):
    if getattr(args, "json", False):
        print(json.dumps(result, indent=2, default=_json_default))
        return
    _sec(args, title)
    kv(result)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def add_common(p):
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--seed", type=int, default=42)
    return p


# ==========================================================================
# data commands
# ==========================================================================

def cmd_data_list(args):
    rows = []
    for name, fn in sorted(D.DATASETS.items()):
        doc = (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else ""
        rows.append([name, doc[:70]])
    _sec(args, "datasets")
    table(["name", "description"], rows)
    print("\n  use: mlkit ml <algo> --dataset <name>   |   "
          "mlkit data make --name <name> --out data.npz")


def cmd_data_make(args):
    X, y, task = D.get_dataset(args.name, n=args.n, seed=args.seed)
    np.savez(args.out, X=X, y=y, task=task)
    emit({"dataset": args.name, "shape": list(X.shape), "task": task,
          "classes": int(len(np.unique(y))) if task == "classification"
          else None, "out": args.out}, args, "dataset written")


def cmd_data_corpus(args):
    text = D.tiny_corpus(args.chars, seed=args.seed)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
    emit({"chars": len(text), "words": len(text.split()),
          "head": text[:120]}, args, "corpus")


# ==========================================================================
# ml commands
# ==========================================================================

def _eval_clf(model, Xtr, Xte, ytr, yte, args, extra=None):
    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    fit_s = time.perf_counter() - t0
    yhat = model.predict(Xte)
    proba = model.predict_proba(Xte) if hasattr(model, "predict_proba") else None
    res = M.evaluate("classification", yte, yhat, proba)
    res["train_acc"] = float((model.predict(Xtr) == ytr).mean())
    res["fit_time_s"] = round(fit_s, 4)
    if extra:
        res.update(extra(model))
    return res, model


def _eval_reg(model, Xtr, Xte, ytr, yte, args, extra=None):
    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    fit_s = time.perf_counter() - t0
    yhat = model.predict(Xte)
    res = M.evaluate("regression", yte, yhat)
    res["fit_time_s"] = round(fit_s, 4)
    if extra:
        res.update(extra(model))
    return res, model


def _handle_unsupervised(args, run, title):
    X = load_unsupervised(args)
    t0 = time.perf_counter()
    out = run(X)
    dt = time.perf_counter() - t0
    res = {"time_s": round(dt, 4)}
    res.update(out)
    emit(res, args, title)


# --- individual algorithms ----------------------------------------------

def ml_linear_regression(args):
    from .ml.linear import LinearRegression
    Xtr, Xte, ytr, yte, _ = load_supervised(args)
    res, _ = _eval_reg(LinearRegression(), Xtr, Xte, ytr, yte, args)
    emit(res, args, "linear regression (OLS)")


def ml_ridge(args):
    from .ml.linear import Ridge
    Xtr, Xte, ytr, yte, _ = load_supervised(args)
    res, _ = _eval_reg(Ridge(alpha=args.alpha), Xtr, Xte, ytr, yte, args,
                       extra=lambda m: {"coef_nonzero": int((np.abs(m.coef_) > 1e-9).sum())})
    emit(res, args, f"ridge (alpha={args.alpha})")


def ml_lasso(args):
    from .ml.linear import Lasso
    Xtr, Xte, ytr, yte, _ = load_supervised(args)
    res, _ = _eval_reg(Lasso(alpha=args.alpha), Xtr, Xte, ytr, yte, args,
                       extra=lambda m: {"coef_nonzero": int((np.abs(m.coef_) > 1e-9).sum()),
                                        "n_iter": m.n_iter_})
    emit(res, args, f"lasso (alpha={args.alpha})")


def ml_elasticnet(args):
    from .ml.linear import ElasticNet
    Xtr, Xte, ytr, yte, _ = load_supervised(args)
    res, _ = _eval_reg(ElasticNet(alpha=args.alpha, l1_ratio=args.l1_ratio),
                       Xtr, Xte, ytr, yte, args,
                       extra=lambda m: {"coef_nonzero": int((np.abs(m.coef_) > 1e-9).sum()),
                                        "n_iter": m.n_iter_})
    emit(res, args, f"elastic net (alpha={args.alpha}, l1_ratio={args.l1_ratio})")


def ml_logistic(args):
    from .ml.logistic import LogisticRegression
    Xtr, Xte, ytr, yte, _ = load_supervised(args)
    model = LogisticRegression(lr=args.lr, epochs=args.epochs, l2=args.l2,
                               batch_size=args.batch)
    res, _ = _eval_clf(model, Xtr, Xte, ytr, yte, args,
                       extra=lambda m: {"epochs_ran": len(m.loss_history_),
                                        "loss_curve": [round(v, 4) for v in m.loss_history_[::max(len(m.loss_history_) // 8, 1)]]})
    emit(res, args, f"logistic regression (lr={args.lr}, epochs={args.epochs})")


def ml_svm(args):
    from .ml.svm import KernelSVM, LinearSVM
    Xtr, Xte, ytr, yte, _ = load_supervised(args)
    if args.kernel == "linear":
        model = LinearSVM(C=args.C, epochs=args.epochs)
        title = f"linear SVM (C={args.C}, subgradient SGD)"
    else:
        model = KernelSVM(C=args.C, kernel=args.kernel, gamma=args.gamma,
                          max_passes=args.passes)
        title = f"kernel SVM ({args.kernel}, C={args.C}, SMO)"
    res, m = _eval_clf(model, Xtr, Xte, ytr, yte, args,
                       extra=lambda m: ({"n_support": int(len(m.support_))}
                                        if getattr(m, "support_", None) is not None
                                        and len(getattr(m, "support_", [])) else {}))
    emit(res, args, title)


def ml_knn(args):
    from .ml.knn import KNN
    Xtr, Xte, ytr, yte, _ = load_supervised(args)
    model = KNN(k=args.k, weights=args.weights,
                task="regression" if args.regression else "classification")
    res, _ = (_eval_reg if args.regression else _eval_clf)(
        model, Xtr, Xte, ytr, yte, args)
    emit(res, args, f"k-NN (k={args.k}, weights={args.weights})")


def ml_naive_bayes(args):
    from .ml.naive_bayes import GaussianNB, MultinomialNB
    Xtr, Xte, ytr, yte, _ = load_supervised(args)
    if args.variant == "multinomial":
        Xtr = np.maximum(Xtr, 0)
        Xte = np.maximum(Xte, 0)
        model = MultinomialNB(alpha=args.alpha)
    else:
        model = GaussianNB()
    res, _ = _eval_clf(model, Xtr, Xte, ytr, yte, args)
    emit(res, args, f"naive bayes ({args.variant})")


def ml_decision_tree(args):
    from .ml.tree import DecisionTree
    Xtr, Xte, ytr, yte, detected = load_supervised(args)
    task = "regression" if (args.regression or detected == "regression") \
        else "classification"
    model = DecisionTree(task=task, criterion=args.criterion,
                         max_depth=args.depth, min_samples_leaf=args.min_leaf,
                         seed=args.seed)
    res, _ = (_eval_reg if task == "regression" else _eval_clf)(
        model, Xtr, Xte, ytr, yte, args,
        extra=lambda m: {"tree_depth": m.depth()})
    emit(res, args, f"decision tree (CART, {args.criterion}, depth={args.depth})")


def ml_random_forest(args):
    from .ml.tree import RandomForest
    Xtr, Xte, ytr, yte, detected = load_supervised(args)
    task = "regression" if (args.regression or detected == "regression") \
        else "classification"
    model = RandomForest(n_estimators=args.n_estimators, task=task,
                         criterion=args.criterion, max_depth=args.depth,
                         max_features=args.max_features, oob=True,
                         seed=args.seed)
    res, _ = (_eval_reg if task == "regression" else _eval_clf)(
        model, Xtr, Xte, ytr, yte, args,
        extra=lambda m: {"oob_score": getattr(m, "oob_score_", None)})
    emit(res, args, f"random forest ({args.n_estimators} trees)")


def ml_gradient_boosting(args):
    from .ml.boosting import GradientBoosting
    Xtr, Xte, ytr, yte, detected = load_supervised(args)
    task = "regression" if (args.regression or detected == "regression") \
        else "classification"
    model = GradientBoosting(n_estimators=args.n_estimators,
                             learning_rate=args.lr, max_depth=args.depth,
                             subsample=args.subsample, task=task,
                             seed=args.seed)
    res, _ = (_eval_reg if task == "regression" else _eval_clf)(
        model, Xtr, Xte, ytr, yte, args,
        extra=lambda m: {"final_train_loss": round(m.loss_history_[-1], 5)})
    emit(res, args, f"gradient boosting ({args.n_estimators} trees, lr={args.lr})")


def ml_xgboost(args):
    from .ml.boosting import XGBoost
    Xtr, Xte, ytr, yte, detected = load_supervised(args)
    task = "regression" if (args.regression or detected == "regression") \
        else "classification"
    model = XGBoost(n_estimators=args.n_estimators, learning_rate=args.lr,
                    max_depth=args.depth, reg_lambda=args.reg_lambda,
                    reg_alpha=args.reg_alpha, min_child_weight=args.min_child_weight,
                    task=task, seed=args.seed)
    res, _ = (_eval_reg if task == "regression" else _eval_clf)(
        model, Xtr, Xte, ytr, yte, args,
        extra=lambda m: {"final_train_loss": round(m.loss_history_[-1], 5)})
    emit(res, args, f"XGBoost (Newton boosting, λ={args.reg_lambda}, α={args.reg_alpha})")


def ml_kmeans(args):
    from .ml.cluster import KMeans
    def run(X):
        km = KMeans(k=args.k, n_init=args.n_init, seed=args.seed).fit(X)
        out = {"k": args.k, "inertia": round(km.inertia_, 3),
               "n_iter": int(km.n_iter_),
               "silhouette": round(km.silhouette(X), 4),
               "cluster_sizes": np.bincount(km.labels_).tolist()}
        return out
    _handle_unsupervised(args, run, f"k-means clustering (k={args.k}, k-means++ init)")


def ml_dbscan(args):
    from .ml.cluster import DBSCAN
    def run(X):
        ds = DBSCAN(eps=args.eps, min_samples=args.min_samples).fit(X)
        labels = ds.labels_
        noise = int((labels == -1).sum())
        sil = M.silhouette_score(X[labels >= 0], labels[labels >= 0]) \
            if len(set(labels)) - (1 if noise else 0) >= 2 else None
        return {"clusters": ds.n_clusters_, "noise_points": noise,
                "silhouette": round(sil, 4) if sil is not None else None,
                "sizes": np.bincount(labels[labels >= 0]).tolist()}
    _handle_unsupervised(args, run,
                         f"DBSCAN (eps={args.eps}, min_samples={args.min_samples})")


def ml_hierarchical(args):
    from .ml.cluster import HierarchicalClustering
    def run(X):
        if len(X) > 1200:
            X = X[:1200]
        hc = HierarchicalClustering(linkage=args.linkage,
                                    n_clusters=args.k).fit(X)
        return {"k": args.k, "linkage": args.linkage,
                "sizes": np.bincount(hc.labels_).tolist(),
                "n_merges": len(hc.Z_),
                "silhouette": round(M.silhouette_score(X, hc.labels_), 4)}
    _handle_unsupervised(args, run,
                         f"agglomerative hierarchical clustering (k={args.k}, {args.linkage} linkage)")


def ml_gmm(args):
    from .ml.cluster import GMM
    def run(X):
        g = GMM(k=args.k, covariance=args.covariance, seed=args.seed).fit(X)
        return {"k": args.k, "covariance": args.covariance,
                "weights": np.round(g.weights_, 3).tolist(),
                "log_likelihood": round(g.score_samples(X), 4),
                "n_iter": int(g.n_iter_),
                "sizes": np.bincount(g.labels_).tolist(),
                "silhouette": round(M.silhouette_score(X, g.labels_), 4)}
    _handle_unsupervised(args, run, f"Gaussian mixture (EM, k={args.k}, {args.covariance})")


def ml_pca(args):
    from .ml.decomposition import PCA
    def run(X):
        pca = PCA(args.components)
        Z = pca.fit_transform(X)
        Xh = pca.inverse_transform(Z)
        return {"explained_variance_ratio": np.round(pca.explained_variance_ratio_, 4).tolist(),
                "cumulative": round(float(pca.explained_variance_ratio_.sum()), 4),
                "reconstruction_error": round(M.mse(X, Xh), 5),
                "embedding_shape": list(Z.shape)}
    _handle_unsupervised(args, run, f"PCA ({args.components} components)")


def ml_tsne(args):
    from .ml.decomposition import TSNE
    def run(X):
        if len(X) > 1000:
            X = X[:1000]
        tsne = TSNE(n_components=args.components, perplexity=args.perplexity,
                    n_iter=args.iterations, seed=args.seed, verbose=not args.json)
        Z = tsne.fit_transform(X)
        out = {"embedding_shape": list(Z.shape), "perplexity": args.perplexity}
        y = _maybe_labels(args)
        if y is not None and len(y) == len(Z):
            out["silhouette"] = round(M.silhouette_score(Z, y), 4)
        return out
    _handle_unsupervised(args, run, "t-SNE (exact, early exaggeration)")


def ml_umap(args):
    from .ml.decomposition import UMAP
    def run(X):
        if len(X) > 1000:
            X = X[:1000]
        u = UMAP(n_components=args.components, n_neighbors=args.neighbors,
                 n_epochs=args.iterations, min_dist=args.min_dist,
                 seed=args.seed, verbose=not args.json)
        Z = u.fit_transform(X)
        out = {"embedding_shape": list(Z.shape),
               "n_neighbors": args.neighbors}
        y = _maybe_labels(args)
        if y is not None and len(y) == len(Z):
            out["silhouette"] = round(M.silhouette_score(Z, y), 4)
        return out
    _handle_unsupervised(args, run, "UMAP (fuzzy kNN graph + SGD layout)")


def ml_lda(args):
    from .ml.decomposition import LinearDiscriminantAnalysis
    Xtr, Xte, ytr, yte, _ = load_supervised(args)
    lda = LinearDiscriminantAnalysis(n_components=args.components)
    t0 = time.perf_counter()
    lda.fit(Xtr, ytr)
    Z = lda.transform(Xte)
    res = {"train_acc": lda.score(Xtr, ytr), "test_acc": lda.score(Xte, yte),
           "embedding_shape": list(Z.shape),
           "explained_variance": np.round(lda.explained_variance_, 4).tolist(),
           "time_s": round(time.perf_counter() - t0, 4)}
    emit(res, args, f"linear discriminant analysis ({args.components} dims)")


def _maybe_labels(args):
    try:
        _, y, _ = D.get_dataset(args.dataset, n=args.n, seed=args.seed)
        return y
    except Exception:                                          # noqa: BLE001
        return None


# --- bench --------------------------------------------------------------

BENCH_ALGOS = [
    ("logistic-reg", lambda a: _mk(a, "LogisticRegression", epochs=200)),
    ("linear-svm", lambda a: _mk(a, "LinearSVM", epochs=60)),
    ("kernel-svm", lambda a: _mk(a, "KernelSVM", kernel="rbf", max_passes=6)),
    ("knn(k=5)", lambda a: _mk(a, "KNN", k=5)),
    ("naive-bayes", lambda a: _mk(a, "GaussianNB")),
    ("decision-tree", lambda a: _mk(a, "DecisionTree", max_depth=6)),
    ("random-forest", lambda a: _mk(a, "RandomForest", n_estimators=50)),
    ("gradient-boosting", lambda a: _mk(a, "GradientBoosting", n_estimators=80)),
    ("xgboost", lambda a: _mk(a, "XGBoost", n_estimators=80)),
]


def _mk(args, cls, **kw):
    import importlib
    import inspect
    mod = importlib.import_module(".ml." + {
        "LogisticRegression": "logistic", "LinearSVM": "svm", "KernelSVM": "svm",
        "KNN": "knn", "GaussianNB": "naive_bayes", "DecisionTree": "tree",
        "RandomForest": "tree", "GradientBoosting": "boosting",
        "XGBoost": "boosting"}[cls], package="mlkit")
    klass = getattr(mod, cls)
    if "seed" in inspect.signature(klass.__init__).parameters:
        kw.setdefault("seed", args.seed)
    return klass(**kw)


def cmd_bench(args):
    Xtr, Xte, ytr, yte, task = load_supervised(args)
    _sec(args, f"benchmark — {args.dataset} (n={len(Xtr)}, d={Xtr.shape[1]}, {task})")
    rows = []
    for name, mk in BENCH_ALGOS:
        model = mk(args)
        try:
            t0 = time.perf_counter()
            model.fit(Xtr, ytr)
            fit_t = time.perf_counter() - t0
            yhat = model.predict(Xte)
            acc = float((yhat == yte).mean())
            rows.append([name, f"{acc:.4f}", f"{fit_t * 1e3:.1f} ms"])
        except Exception as exc:                               # noqa: BLE001
            rows.append([name, f"error: {exc}", "-"])
    table(["algorithm", "test accuracy", "fit time"], rows)


# ==========================================================================
# dl commands
# ==========================================================================

def cmd_dl_mlp(args):
    from .dl.train import build_mlp, fit_model
    Xtr, Xte, ytr, yte, task = load_supervised(args)
    task = "regression" if args.regression else "classification"
    out_dim = 1 if task == "regression" else int(max(ytr.max(), yte.max())) + 1
    hidden = [int(h) for h in args.hidden.split(",")]
    model = build_mlp(Xtr.shape[1], out_dim, hidden=hidden,
                      activation=args.activation, dropout=args.dropout,
                      batch_norm=args.batch_norm)
    _sec(args, f"MLP — {args.dataset} ({task})")
    h = fit_model(model, Xtr, ytr, task=task, optimizer=args.optimizer,
                  lr=args.lr, epochs=args.epochs, batch_size=args.batch,
                  l2=args.l2, clip=args.clip, seed=args.seed,
                  verbose=not args.json)
    # test metrics
    from .dl.tensor import Tensor
    model.eval()
    pred = model(Tensor(Xte.astype(np.float32)))
    if task == "regression":
        yhat = pred.data.ravel()
        res = M.evaluate("regression", yte, yhat)
    else:
        yhat = pred.data.argmax(1)
        res = M.evaluate("classification", yte, yhat)
    res.update({"params": model.num_params(), "optimizer": args.optimizer,
                "epochs": args.epochs, "final_train_loss": round(h["loss"][-1], 4)})
    emit(res, args, "MLP results")


def cmd_dl_cnn(args):
    from .dl.train import CNN, fit_model
    X, y, task = D.get_dataset(args.dataset if args.dataset in ("shapes",)
                               else "shapes", n=args.n, seed=args.seed,
                               size=args.img_size)
    Xtr, Xte, ytr, yte = D.train_test_split(X, y, test=args.test,
                                            seed=args.seed, stratify=True)
    model = CNN(in_ch=X.shape[1], n_classes=int(y.max()) + 1,
                img_size=args.img_size, channels=tuple(
                    int(c) for c in args.channels.split(",")),
                dropout=args.dropout)
    _sec(args, f"CNN — {args.dataset} images {X.shape[1:]}")
    h = fit_model(model, Xtr, ytr, optimizer=args.optimizer, lr=args.lr,
                  epochs=args.epochs, batch_size=args.batch, clip=args.clip,
                  seed=args.seed, verbose=not args.json)
    from .dl.tensor import Tensor
    model.eval()
    yhat = model(Tensor(Xte.astype(np.float32))).data.argmax(1)
    res = M.evaluate("classification", yte, yhat)
    res.update({"params": model.num_params(), "channels": args.channels,
                "final_train_loss": round(h["loss"][-1], 4)})
    emit(res, args, "CNN results")


def cmd_dl_rnn(args):
    from .dl.train import SequenceModel, fit_model
    ds = args.dataset
    if ds == "copy":
        X, y, _ = D.get_dataset("copy", n=args.n, seed=args.seed,
                                length=args.length, vocab=args.vocab)
        vocab = args.vocab
        input_size = 16
        kwargs = {"vocab_size": vocab}
    elif ds == "sine":
        X, y, _ = D.get_dataset("sine", n=args.n, seed=args.seed,
                                seq_len=args.length, horizon=1)
        vocab = None
        input_size = 1
        kwargs = {}
    else:
        X, y, _ = D.get_dataset(ds, n=args.n, seed=args.seed)
        vocab = None
        input_size = X.shape[-1] if X.ndim == 3 else 1
        kwargs = {}
    task = "regression" if y.ndim > 1 and y.shape[1] > 1 or ds == "sine" \
        else "classification"
    out_dim = 1 if task == "regression" else int(np.max(y)) + 1
    Xtr, Xte, ytr, yte = D.train_test_split(X, y, test=args.test,
                                            seed=args.seed,
                                            stratify=task == "classification")
    model = SequenceModel(input_size, args.hidden, out_dim, cell=args.cell,
                          dropout=args.dropout, task=task, **kwargs)
    _sec(args, f"{args.cell.upper()} sequence model — {ds} (task={task})")
    h = fit_model(model, Xtr, ytr, task=task, optimizer=args.optimizer,
                  lr=args.lr, epochs=args.epochs, batch_size=args.batch,
                  clip=args.clip, seed=args.seed,
                  log_every=max(args.epochs // 5, 1), verbose=not args.json)
    from .dl.tensor import Tensor
    model.eval()
    pred = model(Tensor(np.asarray(Xte, dtype=np.float32)))
    if task == "regression":
        res = M.evaluate("regression", yte, pred.data.ravel())
    else:
        res = M.evaluate("classification", yte, pred.data.argmax(1))
    res.update({"cell": args.cell, "params": model.num_params(),
                "final_train_loss": round(h["loss"][-1], 4)})
    emit(res, args, f"{args.cell.upper()} results")


def cmd_dl_vit(args):
    from .dl.train import ViT, fit_model
    X, y, _ = D.get_dataset("shapes", n=args.n, seed=args.seed,
                            size=args.img_size)
    Xtr, Xte, ytr, yte = D.train_test_split(X, y, test=args.test,
                                            seed=args.seed, stratify=True)
    vit = ViT(img_size=args.img_size, patch=args.patch, d_model=args.d_model,
              n_head=args.n_head, depth=args.depth, n_classes=int(y.max()) + 1,
              dropout=args.dropout, seed=args.seed)
    _sec(args, f"Vision Transformer — patches={args.patch}, d={args.d_model}, "
            f"depth={args.depth}")
    h = fit_model(vit, Xtr, ytr, optimizer=args.optimizer, lr=args.lr,
                  epochs=args.epochs, batch_size=args.batch, clip=args.clip,
                  seed=args.seed, log_every=max(args.epochs // 5, 1),
                  verbose=not args.json)
    from .dl.tensor import Tensor
    vit.eval()
    yhat = vit(Tensor(Xte.astype(np.float32))).data.argmax(1)
    res = M.evaluate("classification", yte, yhat)
    res.update({"params": vit.num_params(), "patches": vit.n_patches,
                "final_train_loss": round(h["loss"][-1], 4)})
    emit(res, args, "ViT results")


def cmd_dl_gan(args):
    from .dl.train import train_gan
    X, y, _ = D.get_dataset(args.dataset if args.dataset in ("moons", "circles",
                                                             "blobs", "spirals")
                            else "moons", n=args.n, seed=args.seed)
    _sec(args, f"GAN — 2D data '{args.dataset}'")
    hist = train_gan(X, z_dim=args.z_dim, hidden=args.hidden,
                     epochs=args.epochs, batch_size=args.batch, lr=args.lr,
                     seed=args.seed, verbose=not args.json)
    s = hist["samples"]
    from .ml.knn import KNN
    lab = np.r_[np.zeros(len(X)), np.ones(len(s))]
    acc = KNN(k=5).fit(np.vstack([X, s]), lab).score(np.vstack([X, s]), lab)
    res = {"d_loss": round(hist["d_loss"][-1], 4),
           "g_loss": round(hist["g_loss"][-1], 4),
           "d_acc": round(hist["d_acc"][-1], 3),
           "real_mean": np.round(X.mean(0), 3).tolist(),
           "fake_mean": np.round(s.mean(0), 3).tolist(),
           "real_std": np.round(X.std(0), 3).tolist(),
           "fake_std": np.round(s.std(0), 3).tolist(),
           "two_sample_knn_acc": round(float(acc), 3),
           "note": "two_sample ~0.5 means real vs fake are indistinguishable"}
    emit(res, args, "GAN results")


def cmd_dl_vae(args):
    from .dl.train import train_vae
    X, y, _ = D.get_dataset(args.dataset if args.dataset in ("moons", "circles",
                                                             "blobs")
                            else "moons", n=args.n, seed=args.seed)
    _sec(args, f"VAE — 2D data '{args.dataset}'")
    hist = train_vae(X, latent=args.latent, hidden=args.hidden,
                     epochs=args.epochs, batch_size=args.batch, lr=args.lr,
                     beta=args.beta, seed=args.seed, verbose=not args.json)
    res = {"elbo": round(hist["loss"][-1], 4),
           "recon_loss": round(hist["recon"][-1], 4),
           "kl": round(hist["kl"][-1], 4),
           "latent": args.latent,
           "z_mean_std": np.round(hist["z_mean"].std(0), 3).tolist(),
           "real_mean": np.round(X.mean(0), 3).tolist(),
           "recon_mean": np.round(hist["samples"].mean(0), 3).tolist()}
    emit(res, args, "VAE results")


def cmd_dl_diffusion(args):
    from .dl.train import train_diffusion
    X, y, _ = D.get_dataset(args.dataset if args.dataset in ("blobs", "moons",
                                                             "circles")
                            else "blobs", n=args.n, seed=args.seed)
    _sec(args, f"DDPM diffusion — 2D data '{args.dataset}', T={args.T}, "
            f"schedule={args.schedule}")
    hist = train_diffusion(X, T=args.T, hidden=args.hidden,
                           epochs=args.epochs, batch_size=args.batch,
                           lr=args.lr, schedule=args.schedule, seed=args.seed,
                           verbose=not args.json)
    s = hist["samples"]
    res = {"final_noise_mse": round(hist["loss"][-1], 4),
           "steps": args.T, "schedule": args.schedule,
           "real_mean": np.round(X.mean(0), 3).tolist(),
           "sample_mean": np.round(s.mean(0), 3).tolist(),
           "real_std": np.round(X.std(0), 3).tolist(),
           "sample_std": np.round(s.std(0), 3).tolist()}
    emit(res, args, "diffusion results")


def cmd_dl_optim(args):
    from .dl.optim import rosenbrock, beale
    from .dl.train import run_optimizer_bench
    fn = rosenbrock if args.function == "rosenbrock" else beale
    x0 = np.array([[-1.2, 1.0]], np.float32)
    _sec(args, f"optimizer comparison on {args.function} (lr={args.lr}, "
            f"{args.steps} steps)")
    res = run_optimizer_bench(fn, x0, steps=args.steps, lr=args.lr)
    rows = []
    summary = {}
    for name, v in res.items():
        final = v["final"]
        summary[name] = {"final_loss": round(final, 6),
                         "best_loss": round(min(v["losses"]), 6),
                         "steps_to_1e-3": v["steps_to_1e-3"]}
        rows.append([name, f"{v['losses'][0]:.4f}",
                     f"{final:.6f}", f"{min(v['losses']):.6f}",
                     str(v["steps_to_1e-3"] or "-"),
                     series(v["losses"], 32)])
    table(["optimizer", "start", "final", "best", "<1e-3 @", "loss curve"], rows)
    if args.json:
        print(json.dumps(summary, indent=2))


def cmd_dl_gradcheck(args):
    from .dl.tensor import Tensor, gradcheck_report, cat, stack, gather_rows, \
        cross_entropy, conv2d, maxpool2d, avgpool2d, dropout
    from .dl.nn import attention_gradcheck, layernorm_gradcheck, \
        MultiHeadAttention, TransformerBlock
    rng = np.random.default_rng(0)
    results = []

    def chk(name, fn, params):
        try:
            rep = gradcheck_report(fn, params)
            ok = all(r[3] for r in rep)
            results.append([name, "PASS" if ok else "FAIL",
                            f"{max(r[2] for r in rep):.4f}"])
        except Exception as exc:                               # noqa: BLE001
            results.append([name, f"ERROR {exc}", "-"])

    x = Tensor(rng.normal(size=(4, 5)).astype(np.float32), requires_grad=True, name="x")
    w = Tensor(rng.normal(size=(5, 3)).astype(np.float32), requires_grad=True, name="w")
    b = Tensor(rng.normal(size=3).astype(np.float32), requires_grad=True, name="b")
    chk("linear + mul + tanh", lambda: (((x @ w + b) * (x @ w)) * (x @ w + b).tanh()).mean(), [x, w, b])

    logits = Tensor(rng.normal(size=(6, 4)).astype(np.float32), requires_grad=True, name="logits")
    targets = rng.integers(0, 4, size=6)
    chk("cross-entropy", lambda: cross_entropy(logits, targets) + (logits ** 2).sum(axis=-1).mean(), [logits])

    img = Tensor(rng.normal(size=(2, 3, 8, 8)).astype(np.float32), requires_grad=True, name="img")
    kw = Tensor(rng.normal(size=(4, 3, 3, 3)).astype(np.float32) * 0.3, requires_grad=True, name="conv_w")
    kb = Tensor(rng.normal(size=4).astype(np.float32), requires_grad=True, name="conv_b")
    chk("conv2d + maxpool", lambda: (maxpool2d(conv2d(img, kw, kb, padding=1).relu(), 2) ** 2).mean()
        + avgpool2d(conv2d(img, kw, kb, padding=1), 2).sum() * 0.01, [img, kw, kb])

    emb = Tensor(rng.normal(size=(10, 4)).astype(np.float32), requires_grad=True, name="embedding")
    chk("embedding gather", lambda: (gather_rows(emb, np.array([0, 3, 3, 7])) ** 2).sum(), [emb])

    a = Tensor(rng.normal(size=(3, 4)).astype(np.float32), requires_grad=True, name="a")
    c = Tensor(rng.normal(size=(3, 2)).astype(np.float32), requires_grad=True, name="c")
    chk("concat + stack", lambda: (cat([a, c], axis=1) ** 3).sum(axis=1).mean()
        + (stack([a, a * 2], axis=0) ** 2).mean(), [a, c])

    rep = attention_gradcheck()
    results.append(["attention (SDPA + causal mask)",
                    "PASS" if all(r[3] for r in rep) else "FAIL",
                    f"{max(r[2] for r in rep):.4f}"])
    rep = layernorm_gradcheck()
    results.append(["layer norm", "PASS" if all(r[3] for r in rep) else "FAIL",
                    f"{max(r[2] for r in rep):.4f}"])

    mha = MultiHeadAttention(8, 2, block_size=16, causal=True, bias=True)
    xq = Tensor(rng.normal(size=(3, 5, 8)).astype(np.float32), requires_grad=True, name="x_mha")
    chk("multi-head attention module", lambda: (mha(xq)[0] ** 2).mean(),
        [xq] + mha.parameters())

    blk = TransformerBlock(8, 2, 16, dropout=0.0, causal=True)
    xb = Tensor(rng.normal(size=(2, 6, 8)).astype(np.float32), requires_grad=True, name="x_blk")
    chk("transformer block", lambda: (blk(xb)[0] ** 2).mean(), [xb] + blk.parameters())

    # VAE pieces (target is a detached constant, as in real training)
    from .dl.train import VAE, gaussian_recon, kl_standard_normal
    np.random.seed(0)
    vae = VAE(4, 2, 8)
    xv = Tensor(rng.normal(size=(6, 4)).astype(np.float32), requires_grad=True,
                name="x_vae")
    xv_target = Tensor(xv.data.copy())           # frozen reconstruction target
    def vae_loss():
        np.random.seed(1)
        r2 = np.random.default_rng(1)
        recon, mu, logvar, z = vae(xv, r2)
        return gaussian_recon(recon, xv_target) + kl_standard_normal(mu, logvar)
    chk("VAE (reparameterise + KL)", vae_loss, list(vae.parameters()) + [xv])

    _sec(args, "gradient check: autograd vs central finite differences")
    table(["component", "status", "max rel err"], results)
    n_pass = sum(1 for r in results if r[1] == "PASS")
    print(f"\n  {n_pass}/{len(results)} components passed")
    if args.json:
        print(json.dumps([{"component": r[0], "status": r[1],
                           "max_rel_err": r[2]} for r in results], indent=2))
    sys.exit(0 if n_pass == len(results) else 1)


# ==========================================================================
# llm commands
# ==========================================================================

def _add_llm_common(p):
    p.add_argument("--ckpt", default=None, help="checkpoint directory")
    p.add_argument("--corpus", type=int, default=8000, help="corpus size (chars)")
    p.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    p.add_argument("--train-steps", type=int, default=300)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--block-size", type=int, default=48)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--objective", choices=["clm", "mlm"], default="clm")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p


def cmd_llm_train(args):
    from .llm.checkpoint import train_default_model, save_checkpoint
    _sec(args, f"training TinyGPT — {args.objective.upper()}, "
            f"d={args.d_model}, layers={args.n_layer}, heads={args.n_head}, "
            f"ctx={args.block_size}, tokenizer={args.tokenizer}")
    model, tok, ids, hist = train_default_model(
        objective=args.objective, tokenizer_kind=args.tokenizer,
        corpus_chars=args.corpus, steps=args.train_steps, d_model=args.d_model,
        n_head=args.n_head, n_layer=args.n_layer, block_size=args.block_size,
        lr=args.lr, batch_size=args.batch, seed=args.seed,
        verbose=not (args.quiet or args.json))
    meta = {"corpus_chars": args.corpus, "corpus_seed": args.seed,
            "objective": args.objective, "val_ppl": hist["final_val_ppl"]}
    if args.out:
        save_checkpoint(model, tok, args.out, meta=meta)
    unique = model.num_params(trainable_only=False)
    raw = sum(p.data.size for _, p in model.named_parameters())
    res = {"val_loss": round(hist["final_val_loss"], 4),
           "val_perplexity": round(hist["final_val_ppl"], 3),
           "params": unique, "tied": raw - unique,
           "vocab": tok.vocab_size, "tokens": int(len(ids)),
           "checkpoint": args.out or "(not saved)"}
    emit(res, args, "training complete")


def cmd_llm_evaluate(args):
    from .llm.checkpoint import ensure_model
    from .llm.quantize import evaluate_ppl
    model, tok, ids, src = ensure_model(args)
    ppl = evaluate_ppl(model, ids)
    emit({"source": src, "val_perplexity": round(ppl, 3),
          "tokens": int(len(ids))}, args, "evaluation")


def cmd_llm_tokenize(args):
    from .llm.tokenizer import get_tokenizer
    from .data import tiny_corpus
    text = tiny_corpus(args.corpus, seed=args.seed)
    tok = get_tokenizer(args.tokenizer, text,
                        vocab_size=max(150, args.corpus // 40))
    ids = tok.encode(args.text)
    _sec(args, f"{args.tokenizer} tokenizer")
    kv({"vocab_size": tok.vocab_size, "text": args.text,
        "n_tokens": len(ids), "ids": ids[:40],
        "roundtrip": tok.decode(ids)})
    if args.tokenizer == "bpe" and args.show_merges:
        print(f"  first merges: {tok.merges[:20]}")


def cmd_llm_generate(args):
    from .llm.checkpoint import ensure_model
    from .llm.generate import generate
    model, tok, ids, src = ensure_model(args)
    prompt_ids = tok.encode(args.prompt)
    if not prompt_ids:
        prompt_ids = [tok.bos_id]
    t0 = time.perf_counter()
    out = generate(model, prompt_ids, max_new_tokens=args.tokens,
                   temperature=args.temperature, top_k=args.top_k,
                   top_p=args.top_p, beam_size=args.beam, seed=args.seed,
                   eos_id=tok.eos_id if args.stop_eos else None,
                   use_cache=not args.no_cache)
    dt = time.perf_counter() - t0
    completion = tok.decode(out[len(prompt_ids):])
    if args.json:
        print(json.dumps({"source": src, "prompt": args.prompt,
                          "completion": completion, "ids": out,
                          "seconds": round(dt, 4)}, indent=2))
        return
    _sec(args, f"generate ({src})")
    strategy = []
    if args.beam > 1:
        strategy.append(f"beam={args.beam}")
    if args.temperature <= 0:
        strategy.append("greedy")
    else:
        strategy.append(f"temp={args.temperature}")
        if args.top_k:
            strategy.append(f"top-k={args.top_k}")
        if args.top_p < 1:
            strategy.append(f"top-p={args.top_p}")
    print(f"  strategy : {', '.join(strategy) or 'default'}")
    print(f"  time     : {fmt_time(dt)} for {args.tokens} tokens")
    print(f"\n  prompt     : {args.prompt!r}")
    print(f"  completion : {completion!r}\n")


def cmd_llm_bench_cache(args):
    from .llm.checkpoint import ensure_model
    from .llm.generate import benchmark_cache
    model, tok, ids, src = ensure_model(args)
    prompt = tok.encode(args.prompt) or [tok.bos_id]
    b = benchmark_cache(model, prompt, n_tokens=args.tokens, seed=args.seed)
    res = {"source": src, "tokens": args.tokens,
           "no_cache_ms": round(b["no_cache"] * 1e3, 2),
           "kv_cache_ms": round(b["kv_cache"] * 1e3, 2),
           "speedup": round(b["speedup"], 3),
           "identical_outputs": b["identical"]}
    emit(res, args, "KV-cache benchmark")


def cmd_llm_lora(args):
    from .llm.checkpoint import ensure_model, save_checkpoint
    from .llm.lora import lora_finetune, save_adapter, lora_parameters
    model, tok, ids, src = ensure_model(args)
    _sec(args, f"LoRA fine-tune (rank={args.rank}, alpha={args.alpha}, "
            f"targets={args.targets})")
    hist = lora_finetune(model, ids, tok, rank=args.rank, alpha=args.alpha,
                         targets=tuple(args.targets.split(",")),
                         steps=args.steps, batch_size=args.batch, lr=args.lr,
                         objective=args.objective, seed=args.seed,
                         log_every=max(args.steps // 4, 1),
                         verbose=not args.quiet)
    if args.out:
        save_adapter(model, args.out)
        if args.save_full:
            save_checkpoint(model, tok, args.out + "-full",
                            meta={"lora": True, "rank": args.rank})
    res = {"trainable_params": hist["trainable_params"],
           "total_params": hist["total_params"],
           "percent_trainable": round(
               100 * hist["trainable_params"] / max(hist["total_params"], 1), 2),
           "val_perplexity": round(hist["final_val_ppl"], 3),
           "adapter": args.out or "(not saved)"}
    emit(res, args, "LoRA fine-tune complete")


def cmd_llm_merge(args):
    import re
    from .llm.lora import load_adapter, merge_lora, apply_lora
    from .llm.checkpoint import ensure_model
    from .llm.generate import generate
    model, tok, ids, src = ensure_model(args)
    # discover adapter structure: which layers, which rank
    data = np.load(args.adapter)
    targets, rank = set(), None
    for k in data.files:
        m = re.match(r"blocks\.(\d+)\.attn\.([^.]+)\.(A|B)$", k)
        if m:
            targets.add(m.group(2))
            if m.group(3) == "A":
                rank = data[k].shape[1]
    if not targets or rank is None:
        raise KeyError("adapter file has no recognizable LoRA keys")
    apply_lora(model, targets=tuple(sorted(targets)), rank=rank)
    n = load_adapter(model, args.adapter)
    merged = merge_lora(model)
    prompt = tok.encode(args.prompt) or [tok.bos_id]
    out = generate(model, prompt, max_new_tokens=args.tokens,
                   temperature=0.8, seed=args.seed)
    emit({"tensors_loaded": n, "layers_merged": merged, "rank": rank,
          "sample": tok.decode(out[len(prompt):])}, args,
         "adapter merged into base weights (sample generation follows)")


def cmd_llm_quantize(args):
    from .llm.checkpoint import ensure_model
    from .llm.quantize import (quantize_model, snapshot_weights,
                               evaluate_ppl, iter_linears)
    model, tok, ids, src = ensure_model(args)
    snap = snapshot_weights(model)
    ppl_fp = evaluate_ppl(model, ids)
    rows = []
    report = {"fp32_perplexity": round(ppl_fp, 3), "bits": []}
    for bits in (args.bits.split(",") if args.bits else ["int8", "int4"]):
        rep = quantize_model(model, bits=bits, apply=True)
        ppl_q = evaluate_ppl(model, ids)
        rows.append([bits, f"{ppl_fp:.3f}", f"{ppl_q:.3f}",
                     f"{rep['fp32_bytes']}", f"{rep['quantized_bytes']}",
                     f"{rep['compression']:.2f}x",
                     f"{rep['max_abs_err']:.4f}"])
        report["bits"].append({
            "bits": bits, "perplexity": round(ppl_q, 3),
            "fp32_bytes": rep["fp32_bytes"],
            "quantized_bytes": rep["quantized_bytes"],
            "compression": round(rep["compression"], 2),
            "max_abs_err": round(rep["max_abs_err"], 5)})
        dequantize_model_restore(model, snap)
    if args.json:
        print(json.dumps(report, indent=2))
        return
    _sec(args, f"weight quantization ({src})")
    table(["bits", "ppl fp32", "ppl quant", "bytes fp32", "bytes quant",
           "compression", "max err"], rows)
    print("\n  (weights are quantised then dequantised to simulate inference "
          "at reduced precision)")


def dequantize_model_restore(model, snap):
    from .llm.quantize import dequantize_model
    dequantize_model(model, snap)


def cmd_llm_flash(args):
    from .llm.flash import benchmark
    res = benchmark(B=1, H=args.heads, T=args.seq_len, d=args.dim,
                    causal=not args.bidirectional)
    if args.json:
        print(json.dumps(res, indent=2))
        return
    _sec(args, "flash attention vs naive attention")
    kv({"shape": res["shape"], "causal": res["causal"],
        "naive": f"{res['naive_ms']:.3f} ms",
        "flash (tiled, online softmax)": f"{res['flash_ms']:.3f} ms",
        "max abs difference": res["max_abs_err"],
        "note": "identical outputs up to float rounding; flash tiles the "
                "S×S score matrix (memory O(B·H·block_q·block_k) instead of "
                "O(B·H·T·T)); pure-numpy loop overhead can make it slower "
                "at small T"})


def cmd_llm_speculative(args):
    from .llm.checkpoint import ensure_model, train_default_model, TinyGPT
    from .llm import speculative as S
    model, tok, ids, src = ensure_model(args)
    # draft: smaller model trained on the same corpus (or a saved ckpt)
    if args.draft_ckpt:
        from .llm.checkpoint import load_checkpoint
        draft, dtok, _ = load_checkpoint(args.draft_ckpt)
    else:
        from .llm.model import train_lm
        draft = TinyGPT(tok.vocab_size, d_model=max(args.d_model // 2, 16),
                        n_head=max(args.n_head // 2, 1), n_layer=1,
                        block_size=args.block_size, seed=args.seed + 1)
        train_lm(draft, ids, steps=args.train_steps,
                 batch_size=args.batch, lr=args.lr,
                 log_every=9999, verbose=False, seed=args.seed + 1)
    prompt = tok.encode(args.prompt) or [tok.bos_id]
    sb = S.benchmark(model, draft, prompt, n_tokens=args.tokens, k=args.k,
                     temperature=1.0, seed=args.seed)
    res = {"tokens": args.tokens, "k": args.k,
           "baseline_s": round(sb["baseline_s"], 4),
           "speculative_s": round(sb["speculative_s"], 4),
           "wall_clock_speedup": round(sb["speedup"], 2),
           "acceptance_rate": round(sb["acceptance_rate"], 3),
           "target_forwards_baseline": sb["target_forwards_baseline"],
           "target_forwards_speculative": sb["target_forwards_speculative"],
           "draft_forwards": sb["draft_forwards"],
           "note": "speculative decoding preserves the target distribution "
                   "exactly; speedup appears when the target forward pass "
                   "dominates cost (fewer, bigger target calls)"}
    emit(res, args, "speculative decoding")


def cmd_llm_rag(args):
    from .llm.rag import RAG
    docs = [(f"doc{i+1}", text) for i, text in enumerate(_rag_docs())]
    rag = RAG(index_kind=args.index, embedder=args.embedder,
              chunk_words=args.chunk, overlap=args.overlap // 2,
              dim=args.dim)
    if args.embedder == "lm":
        from .llm.checkpoint import ensure_model
        model, tok, ids, src = ensure_model(args)
        rag.model, rag.tokenizer = model, tok
    t0 = time.perf_counter()
    rag.index_documents(docs,
                           verbose=not (args.quiet or args.json))
    index_t = time.perf_counter() - t0
    q = args.query
    hits = rag.retrieve(q, k=args.k)
    if args.generate:
        from .llm.checkpoint import ensure_model
        if rag.model is None:
            model, tok, ids, src = ensure_model(args)
            rag.model, rag.tokenizer = model, tok
        ans = rag.answer_generate(q, k=args.k, max_new_tokens=args.tokens,
                                  seed=args.seed)
    else:
        ans = rag.answer_extractive(q, k=args.k)
    if args.json:
        print(json.dumps({"index": args.index, "query": q, "hits": hits,
                          "answer": ans["answer"],
                          "index_seconds": round(index_t, 4)}, indent=2,
                         default=_json_default))
        return
    _sec(args, f"RAG — index={args.index}, embedder={args.embedder}, "
            f"query={q!r}")
    table(["#", "score", "source", "passage"],
          [[h["rank"], f"{h['score']:.3f}", h["source"], h["text"][:64] + "…"]
           for h in hits])
    print(f"\n  answer ({ans['mode']}): {ans['answer'][:400]}")
    print(f"\n  index built in {fmt_time(index_t)} over {len(rag.chunks)} chunks")


def _rag_docs():
    """Small built-in knowledge base for the RAG demo."""
    return [
        "The Transformer architecture was introduced in the paper Attention "
        "Is All You Need in 2017. It relies entirely on self-attention and "
        "dispenses with recurrence and convolutions. Every Transformer layer "
        "contains multi-head attention and a feed-forward network.",

        "Large language models are pre-trained with objectives such as causal "
        "language modelling, where the model predicts the next token, and "
        "masked language modelling, where the model reconstructs masked "
        "tokens using bidirectional context.",

        "Fine-tuning techniques include supervised fine-tuning, RLHF, and "
        "direct preference optimisation. Parameter-efficient methods like "
        "LoRA freeze the base weights and inject trainable low-rank "
        "matrices into attention projections.",

        "Inference efficiency comes from KV caching, quantisation to int8 or "
        "int4, flash attention, and speculative decoding where a small draft "
        "model proposes tokens verified by the large target model.",

        "Retrieval augmented generation combines a retriever with a "
        "generator. Documents are chunked, embedded into vectors, stored in "
        "an index such as flat search, IVF, or HNSW, and the top-k chunks "
        "are placed into the prompt context.",

        "K-means partitions data into k clusters by iteratively assigning "
        "points to the nearest centroid and recomputing centroids. DBSCAN "
        "finds density-based clusters and marks noise. Gaussian mixtures "
        "use expectation maximisation to estimate soft cluster membership.",

        "Gradient boosting builds trees sequentially, each fitting the "
        "residual or gradient of the loss. XGBoost additionally uses "
        "second-order hessians, L1 and L2 regularisation, and histogram "
        "based splitting.",
    ]


def cmd_llm_agent(args):
    from .llm.agent import react, chain_of_thought
    _sec(args, f"agent — mode={args.mode}")
    questions = args.question or [
        "calculate 12 * (8 + 3) - 45 / 5",
        "convert 100 km to mi",
        "what is photosynthesis",
        "convert 25 c to f",
    ]
    results = []
    quiet = args.json
    if not quiet:
        print()
    for q in questions:
        if not quiet:
            print(f"  Q: {q}")
        if args.mode == "cot":
            out = chain_of_thought(q, verbose=not quiet)
        else:
            out = react(q, verbose=not quiet)
        results.append({"question": q, "answer": out["answer"]})
        if not quiet:
            print()
    if args.json:
        print(json.dumps(results, indent=2))


def cmd_llm_kv(args):
    """KV-cache internals demo: show cache growth per step."""
    from .llm.checkpoint import ensure_model
    model, tok, ids, src = ensure_model(args)
    prompt = tok.encode(args.prompt) or [tok.bos_id]
    model.eval()
    _sec(args, "KV cache growth")
    rows = []
    logits, _, kv = model(np.array([prompt], dtype=np.int64), use_cache=True)
    rows.append([0, len(prompt), _kv_elems(kv), "prefill"])
    for step in range(1, args.steps + 1):
        nxt = int(logits.data[0, -1].argmax())
        logits, _, kv = model(np.array([[nxt]]), past_kv=kv, use_cache=True)
        rows.append([step, len(prompt) + step, _kv_elems(kv), "decode"])
    table(["step", "seq_len", "cached K/V floats", "phase"], rows)
    total = rows[-1][2] * 4  # bytes (fp32)
    print(f"\n  final cache: {rows[-1][2]:,} floats "
          f"({total / 1024:.1f} KiB fp32) — avoids recomputing "
          f"keys/values for the {rows[-1][1]} context tokens")


def _kv_elems(kv) -> int:
    if not kv:
        return 0
    n = 0
    for k, v in kv:
        n += int(k.data.size) + int(v.data.size)
    return n


# ==========================================================================
# argument parsing
# ==========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mlkit",
        description="ML / Deep Learning / LLM algorithms from scratch "
                    "(pure NumPy) with a unified CLI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
               "  mlkit ml random-forest --dataset moons\n"
               "  mlkit bench --dataset circles\n"
               "  mlkit dl mlp --dataset spirals --epochs 150\n"
               "  mlkit dl gradcheck\n"
               "  mlkit llm train --out ckpt/\n"
               "  mlkit llm generate --ckpt ckpt/ --prompt 'the '\n"
               "  mlkit llm rag --query 'what is lora'\n")
    p.add_argument("--version", action="version", version=f"mlkit {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    # ---- data
    sp = sub.add_parser("data", help="datasets")
    ssub = sp.add_subparsers(dest="sub", required=True)
    q = ssub.add_parser("list", help="list built-in datasets")
    q.set_defaults(func=cmd_data_list)
    q = ssub.add_parser("make", help="generate a dataset .npz")
    q.add_argument("--name", default="moons")
    q.add_argument("--n", type=int, default=600)
    q.add_argument("--out", default="dataset.npz")
    add_common(q)
    q.set_defaults(func=cmd_data_make)
    q = ssub.add_parser("corpus", help="generate the text corpus")
    q.add_argument("--chars", type=int, default=8000)
    q.add_argument("--out", default=None)
    add_common(q)
    q.set_defaults(func=cmd_data_corpus)

    # ---- ml
    sp = sub.add_parser("ml", help="classical machine-learning algorithms")
    msub = sp.add_subparsers(dest="sub", required=True)

    def mlp(name, func, dataset="moons", **kw):
        q = msub.add_parser(name, help=name)
        add_data_args(q, default_dataset=dataset)
        add_common(q)
        for k, v in kw.items():
            q.add_argument(*v[:-1], **v[-1])
        q.set_defaults(func=func)
        return q

    mlp("linear-regression", ml_linear_regression, dataset="regression")
    mlp("ridge", ml_ridge, dataset="regression",
        **{"--alpha": ("--alpha", dict(type=float, default=1.0, help="L2"))})
    mlp("lasso", ml_lasso, dataset="regression",
        **{"--alpha": ("--alpha", dict(type=float, default=0.05, help="L1"))})
    mlp("elasticnet", ml_elasticnet, dataset="regression",
        **{"--alpha": ("--alpha", dict(type=float, default=0.05)),
           "--l1-ratio": ("--l1-ratio", dict(type=float, default=0.5))})
    mlp("logistic-regression", ml_logistic,
        **{"--lr": ("--lr", dict(type=float, default=0.5)),
           "--epochs": ("--epochs", dict(type=int, default=300)),
           "--l2": ("--l2", dict(type=float, default=0.0)),
           "--batch": ("--batch", dict(type=int, default=64))})
    mlp("svm", ml_svm,
        **{"--kernel": ("--kernel", dict(choices=["linear", "rbf", "poly"],
                                         default="rbf")),
           "--C": ("--C", dict(type=float, default=1.0)),
           "--gamma": ("--gamma", dict(type=float, default=None)),
           "--epochs": ("--epochs", dict(type=int, default=60)),
           "--passes": ("--passes", dict(type=int, default=8))})
    mlp("knn", ml_knn,
        **{"--k": ("--k", dict(type=int, default=5)),
           "--weights": ("--weights", dict(choices=["uniform", "distance"],
                                           default="uniform")),
           "--regression": ("--regression", dict(action="store_true"))})
    mlp("naive-bayes", ml_naive_bayes,
        **{"--variant": ("--variant", dict(choices=["gaussian", "multinomial"],
                                           default="gaussian")),
           "--alpha": ("--alpha", dict(type=float, default=1.0))})
    mlp("decision-tree", ml_decision_tree, dataset="circles",
        **{"--depth": ("--depth", dict(type=int, default=None)),
           "--criterion": ("--criterion", dict(choices=["gini", "entropy"],
                                               default="gini")),
           "--min-leaf": ("--min-leaf", dict(type=int, default=1)),
           "--regression": ("--regression", dict(action="store_true"))})
    mlp("random-forest", ml_random_forest, dataset="circles",
        **{"--n-estimators": ("--n-estimators", dict(type=int, default=100)),
           "--depth": ("--depth", dict(type=int, default=None)),
           "--criterion": ("--criterion", dict(choices=["gini", "entropy"],
                                               default="gini")),
           "--max-features": ("--max-features",
                              dict(default="sqrt",
                                   help="sqrt | log2 | int | float")),
           "--regression": ("--regression", dict(action="store_true"))})
    mlp("gradient-boosting", ml_gradient_boosting, dataset="regression",
        **{"--n-estimators": ("--n-estimators", dict(type=int, default=100)),
           "--lr": ("--lr", dict(type=float, default=0.1)),
           "--depth": ("--depth", dict(type=int, default=3)),
           "--subsample": ("--subsample", dict(type=float, default=1.0)),
           "--regression": ("--regression", dict(action="store_true"))})
    mlp("xgboost", ml_xgboost, dataset="regression",
        **{"--n-estimators": ("--n-estimators", dict(type=int, default=100)),
           "--lr": ("--lr", dict(type=float, default=0.1)),
           "--depth": ("--depth", dict(type=int, default=3)),
           "--reg-lambda": ("--reg-lambda", dict(type=float, default=1.0)),
           "--reg-alpha": ("--reg-alpha", dict(type=float, default=0.0)),
           "--min-child-weight": ("--min-child-weight",
                                  dict(type=float, default=1.0)),
           "--regression": ("--regression", dict(action="store_true"))})
    mlp("kmeans", ml_kmeans, dataset="gmm",
        **{"--k": ("--k", dict(type=int, default=3)),
           "--n-init": ("--n-init", dict(type=int, default=10))})
    mlp("dbscan", ml_dbscan, dataset="gmm",
        **{"--eps": ("--eps", dict(type=float, default=0.8)),
           "--min-samples": ("--min-samples", dict(type=int, default=5))})
    mlp("hierarchical", ml_hierarchical, dataset="gmm",
        **{"--k": ("--k", dict(type=int, default=3)),
           "--linkage": ("--linkage",
                         dict(choices=["single", "complete", "average", "ward"],
                              default="ward"))})
    mlp("gmm", ml_gmm, dataset="gmm",
        **{"--k": ("--k", dict(type=int, default=3)),
           "--covariance": ("--covariance",
                            dict(choices=["full", "diag"], default="full"))})
    mlp("pca", ml_pca, dataset="blobs",
        **{"--components": ("--components", dict(type=int, default=2))})
    mlp("tsne", ml_tsne, dataset="blobs",
        **{"--components": ("--components", dict(type=int, default=2)),
           "--perplexity": ("--perplexity", dict(type=float, default=30.0)),
           "--iterations": ("--iterations", dict(type=int, default=600))})
    mlp("umap", ml_umap, dataset="blobs",
        **{"--components": ("--components", dict(type=int, default=2)),
           "--neighbors": ("--neighbors", dict(type=int, default=15)),
           "--min-dist": ("--min-dist", dict(type=float, default=0.1)),
           "--iterations": ("--iterations", dict(type=int, default=300))})
    mlp("lda", ml_lda, dataset="blobs",
        **{"--components": ("--components", dict(type=int, default=2))})

    # ---- bench
    q = sub.add_parser("bench", help="benchmark classical ML algorithms")
    add_data_args(q, default_dataset="moons")
    add_common(q)
    q.set_defaults(func=cmd_bench)

    # ---- dl
    sp = sub.add_parser("dl", help="deep learning models & tools")
    dsub = sp.add_subparsers(dest="sub", required=True)

    def dl(name, func, dataset=None, **kw):
        q = dsub.add_parser(name, help=name)
        if dataset is not None:
            add_data_args(q, default_dataset=dataset)
        else:
            q.add_argument("--dataset", default="shapes")
            q.add_argument("--n", type=int, default=400)
            q.add_argument("--test", type=float, default=0.2)
        add_common(q)
        q.add_argument("--optimizer", default="adam",
                       choices=["sgd", "adam", "adamw", "rmsprop"])
        q.add_argument("--lr", type=float, default=3e-3)
        q.add_argument("--epochs", type=int, default=40)
        q.add_argument("--batch", type=int, default=64)
        q.add_argument("--dropout", type=float, default=0.0)
        q.add_argument("--clip", type=float, default=1.0,
                       help="gradient clipping norm (0 = off)")
        q.add_argument("--l2", type=float, default=0.0)
        for k, v in kw.items():
            q.add_argument(*v[:-1], **v[-1])
        q.set_defaults(func=func)
        return q

    dl("mlp", cmd_dl_mlp, dataset="spirals",
       **{"--hidden": ("--hidden", dict(default="64,64",
                                        help="comma-separated hidden sizes")),
          "--activation": ("--activation",
                           dict(choices=["relu", "tanh", "gelu", "sigmoid"],
                                default="relu")),
          "--batch-norm": ("--batch-norm", dict(action="store_true")),
          "--regression": ("--regression", dict(action="store_true"))})
    dl("cnn", cmd_dl_cnn, dataset=None,
       **{"--img-size": ("--img-size", dict(type=int, default=16)),
          "--channels": ("--channels", dict(default="8,16"))})
    dl("rnn", cmd_dl_rnn, dataset="copy",
       **{"--cell": ("--cell", dict(choices=["rnn", "gru", "lstm"],
                                    default="lstm")),
          "--hidden": ("--hidden", dict(type=int, default=32)),
          "--length": ("--length", dict(type=int, default=8)),
          "--vocab": ("--vocab", dict(type=int, default=6))})
    dl("vit", cmd_dl_vit, dataset=None,
       **{"--img-size": ("--img-size", dict(type=int, default=16)),
          "--patch": ("--patch", dict(type=int, default=4)),
          "--d-model": ("--d-model", dict(type=int, default=32)),
          "--n-head": ("--n-head", dict(type=int, default=4)),
          "--depth": ("--depth", dict(type=int, default=2))})
    dl("gan", cmd_dl_gan, dataset="moons",
       **{"--z-dim": ("--z-dim", dict(type=int, default=8)),
          "--hidden": ("--hidden", dict(type=int, default=32))})
    dl("vae", cmd_dl_vae, dataset="moons",
       **{"--latent": ("--latent", dict(type=int, default=2)),
          "--hidden": ("--hidden", dict(type=int, default=32)),
          "--beta": ("--beta", dict(type=float, default=1.0))})
    dl("diffusion", cmd_dl_diffusion, dataset="blobs",
       **{"--T": ("--T", dict(type=int, default=100)),
          "--hidden": ("--hidden", dict(type=int, default=64)),
          "--schedule": ("--schedule",
                         dict(choices=["linear", "cosine"], default="linear"))})

    q = dsub.add_parser("optim", help="optimizer comparison")
    q.add_argument("--function", choices=["rosenbrock", "beale"],
                   default="rosenbrock")
    q.add_argument("--steps", type=int, default=400)
    q.add_argument("--lr", type=float, default=1e-3)
    add_common(q)
    q.set_defaults(func=cmd_dl_optim)

    q = dsub.add_parser("gradcheck",
                        help="verify autograd against finite differences")
    add_common(q)
    q.set_defaults(func=cmd_dl_gradcheck)

    # ---- llm
    sp = sub.add_parser("llm", help="large language models")
    lsub = sp.add_subparsers(dest="sub", required=True)

    q = lsub.add_parser("train", help="pre-train a TinyGPT (clm/mlm)")
    _add_llm_common(q)
    q.add_argument("--out", default=None, help="checkpoint directory")
    q.set_defaults(func=cmd_llm_train)

    q = lsub.add_parser("evaluate", help="perplexity of a model")
    _add_llm_common(q)
    q.set_defaults(func=cmd_llm_evaluate)

    q = lsub.add_parser("tokenize", help="tokenizer encode/decode demo")
    q.add_argument("--text", default="the old king found the map")
    q.add_argument("--tokenizer", choices=["char", "bpe"], default="bpe")
    q.add_argument("--corpus", type=int, default=8000)
    q.add_argument("--show-merges", action="store_true")
    add_common(q)
    q.set_defaults(func=cmd_llm_tokenize)

    q = lsub.add_parser("generate", help="sample text from a model")
    _add_llm_common(q)
    q.add_argument("--prompt", default="the ")
    q.add_argument("--tokens", type=int, default=60)
    q.add_argument("--temperature", type=float, default=0.8,
                   help="0 = greedy")
    q.add_argument("--top-k", type=int, default=None)
    q.add_argument("--top-p", type=float, default=1.0)
    q.add_argument("--beam", type=int, default=1, help="beam size (>1 = beam search)")
    q.add_argument("--no-cache", action="store_true", help="disable KV cache")
    q.add_argument("--stop-eos", action="store_true")
    q.set_defaults(func=cmd_llm_generate)

    q = lsub.add_parser("bench-cache", help="KV-cache speed benchmark")
    _add_llm_common(q)
    q.add_argument("--prompt", default="the ")
    q.add_argument("--tokens", type=int, default=40)
    q.set_defaults(func=cmd_llm_bench_cache)

    q = lsub.add_parser("kv", help="show KV cache growth per decode step")
    _add_llm_common(q)
    q.add_argument("--prompt", default="the ")
    q.add_argument("--steps", type=int, default=8)
    q.set_defaults(func=cmd_llm_kv)

    q = lsub.add_parser("lora", help="LoRA parameter-efficient fine-tuning")
    _add_llm_common(q)
    q.add_argument("--rank", type=int, default=4)
    q.add_argument("--alpha", type=float, default=1.0)
    q.add_argument("--targets", default="c_attn",
                   help="comma separated linear attrs (c_attn,c_proj)")
    q.add_argument("--steps", type=int, default=200)
    q.add_argument("--out", default="adapter.npz")
    q.add_argument("--save-full", action="store_true")
    q.set_defaults(func=cmd_llm_lora)

    q = lsub.add_parser("merge", help="merge a LoRA adapter into the base model")
    _add_llm_common(q)
    q.add_argument("--adapter", required=True)
    q.add_argument("--prompt", default="the ")
    q.add_argument("--tokens", type=int, default=40)
    q.set_defaults(func=cmd_llm_merge)

    q = lsub.add_parser("quantize", help="int8/int4 weight quantization")
    _add_llm_common(q)
    q.add_argument("--bits", default="int8,int4",
                   help="comma list: int8,int4")
    q.set_defaults(func=cmd_llm_quantize)

    q = lsub.add_parser("flash", help="flash attention benchmark")
    q.add_argument("--seq-len", type=int, default=256)
    q.add_argument("--heads", type=int, default=8)
    q.add_argument("--dim", type=int, default=32)
    q.add_argument("--bidirectional", action="store_true")
    add_common(q)
    q.set_defaults(func=cmd_llm_flash)

    q = lsub.add_parser("speculative", help="speculative decoding benchmark")
    _add_llm_common(q)
    q.add_argument("--prompt", default="the ")
    q.add_argument("--tokens", type=int, default=32)
    q.add_argument("--k", type=int, default=4, help="draft tokens per step")
    q.add_argument("--draft-ckpt", default=None)
    q.set_defaults(func=cmd_llm_speculative)

    q = lsub.add_parser("rag", help="retrieval-augmented generation demo")
    _add_llm_common(q)
    q.add_argument("--index", choices=["flat", "ivf", "hnsw"], default="flat")
    q.add_argument("--embedder", choices=["tfidf", "lm"], default="tfidf")
    q.add_argument("--dim", type=int, default=256)
    q.add_argument("--chunk", type=int, default=40)
    q.add_argument("--overlap", type=int, default=20)
    q.add_argument("--query", default="how does speculative decoding work?")
    q.add_argument("--k", type=int, default=4)
    q.add_argument("--generate", action="store_true",
                   help="prompt-stuff + LM generation (needs/ trains a model)")
    q.add_argument("--tokens", type=int, default=60)
    q.set_defaults(func=cmd_llm_rag)

    q = lsub.add_parser("agent", help="ReAct / Chain-of-Thought agent loops")
    q.add_argument("--mode", choices=["react", "cot"], default="react")
    q.add_argument("--question", action="append",
                   help="repeatable; defaults to a demo set")
    add_common(q)
    q.set_defaults(func=cmd_llm_agent)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # defaults for commands without --seed
    if not hasattr(args, "seed"):
        args.seed = 42
    set_seed(getattr(args, "seed", 42))
    try:
        args.func(args)
    except BrokenPipeError:
        return 0
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())