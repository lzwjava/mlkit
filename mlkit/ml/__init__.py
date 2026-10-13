"""Classical machine-learning algorithms (pure NumPy, sklearn-style API)."""

from .linear import LinearRegression, Ridge, Lasso, ElasticNet
from .logistic import LogisticRegression
from .svm import LinearSVM, KernelSVM
from .knn import KNN
from .naive_bayes import GaussianNB, MultinomialNB
from .tree import DecisionTree, RandomForest
from .boosting import GradientBoosting, XGBoost
from .cluster import KMeans, DBSCAN, HierarchicalClustering, GMM
from .decomposition import PCA, TSNE, UMAP, LinearDiscriminantAnalysis

__all__ = [
    "LinearRegression", "Ridge", "Lasso", "ElasticNet", "LogisticRegression",
    "LinearSVM", "KernelSVM", "KNN", "GaussianNB", "MultinomialNB",
    "DecisionTree", "RandomForest", "GradientBoosting", "XGBoost",
    "KMeans", "DBSCAN", "HierarchicalClustering", "GMM",
    "PCA", "TSNE", "UMAP", "LinearDiscriminantAnalysis",
]