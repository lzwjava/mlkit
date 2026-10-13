"""Naive Bayes classifiers.

* GaussianNB     -> P(x|c) = Π N(x_j; mu_cj, var_cj)   (continuous features)
* MultinomialNB  -> P(x|c) = Π P(w_j|c)^x_j             (count features, text)
"""

from __future__ import annotations

import numpy as np


class GaussianNB:
    def __init__(self, var_smoothing: float = 1e-9, priors=None):
        self.var_smoothing = var_smoothing
        self.priors = priors

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        self.classes_, counts = np.unique(y, return_counts=True)
        K = len(self.classes_)
        n, d = X.shape
        self.class_prior_ = (counts / n if self.priors is None
                             else np.asarray(self.priors, dtype=float))
        self.theta_ = np.zeros((K, d))          # per-class feature means
        self.var_ = np.zeros((K, d))            # per-class feature variances
        eps = self.var_smoothing * X.var(axis=0).max()
        for i, c in enumerate(self.classes_):
            Xc = X[y == c]
            self.theta_[i] = Xc.mean(0)
            self.var_[i] = Xc.var(0) + eps
        return self

    def _log_likelihood(self, X):
        X = np.asarray(X, dtype=float)
        # (n, K, d) gaussian log-pdf, summed over features
        ll = np.empty((len(X), len(self.classes_)))
        for i in range(len(self.classes_)):
            mu, var = self.theta_[i], self.var_[i]
            log_pdf = -0.5 * (np.log(2 * np.pi * var) + ((X - mu) ** 2) / var)
            ll[:, i] = log_pdf.sum(1)
        return ll

    def predict_log_proba(self, X):
        ll = self._log_likelihood(X)
        return ll + np.log(self.class_prior_)

    def predict_proba(self, X):
        logp = self.predict_log_proba(X)
        logp -= logp.max(1, keepdims=True)
        p = np.exp(logp)
        return p / p.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_log_proba(X).argmax(1)]

    def score(self, X, y):
        return float((self.predict(X) == np.asarray(y)).mean())


class MultinomialNB:
    """Multinomial Naive Bayes with Laplace smoothing (classic text classifier)."""

    def __init__(self, alpha: float = 1.0, priors=None):
        self.alpha = alpha
        self.priors = priors

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        self.classes_, counts = np.unique(y, return_counts=True)
        K = len(self.classes_)
        d = X.shape[1]
        self.class_log_prior_ = np.log(
            counts / len(y) if self.priors is None else np.asarray(self.priors))
        self.feature_log_prob_ = np.zeros((K, d))
        for i, c in enumerate(self.classes_):
            Xc = X[y == c].sum(0)
            self.feature_log_prob_[i] = np.log(
                (Xc + self.alpha) / (Xc.sum() + self.alpha * d))
        return self

    def _joint_log_likelihood(self, X):
        return np.asarray(X, dtype=float) @ self.feature_log_prob_.T \
            + self.class_log_prior_

    def predict_proba(self, X):
        j = self._joint_log_likelihood(X)
        j -= j.max(1, keepdims=True)
        p = np.exp(j)
        return p / p.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[self._joint_log_likelihood(X).argmax(1)]

    def score(self, X, y):
        return float((self.predict(X) == np.asarray(y)).mean())