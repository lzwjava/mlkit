"""Deep-learning subpackage: autograd engine, layers, optimizers, trainers."""

from . import tensor, nn, optim, train          # noqa: F401
from .tensor import Tensor, cross_entropy, gradcheck_report   # noqa: F401