# include all packages, including those needed for the children classes
from abc import ABCMeta
from collections.abc import Mapping
from copy import copy

import numpy as np
from numpy.linalg import pinv, norm, lstsq
import tensorly as tl
from tensorly.cp_tensor import CPTensor
from tensorly.tenalg import khatri_rao, mode_dot, multi_mode_dot


def calcR2X(X, Xhat):
    if (Xhat.ndim == 2) and (X.ndim == 1):
        X = X.reshape(-1, 1)
    assert X.shape == Xhat.shape
    mask = np.isfinite(X)
    xIn = np.nan_to_num(X)
    top = norm(Xhat * mask - xIn) ** 2.0
    bottom = norm(xIn) ** 2.0
    return 1 - top / bottom


class PLSTensor(Mapping, metaclass=ABCMeta):
    """ Base class for all variants of tensor PLS """
    def __init__(self, n_components:int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Parameters
        self.n_components = n_components

    def __getitem__(self, index):
        if index == 0:
            return self.X_factors
        elif index == 1:
            return self.Y_factors
        else:
            raise IndexError

    def __iter__(self):
        yield self.X_factors
        yield self.Y_factors

    def __len__(self):
        return 2

    def copy(self):
        return copy(self)

    def preprocess(self):
        self.X_mean = np.mean(self.X, axis=0)
        self.Y_mean = np.mean(self.Y, axis=0)
        self.X -= self.X_mean
        self.Y -= self.Y_mean

    def _init_factors(self, X, Y):
        assert X.shape[0] == Y.shape[0]
        assert Y.ndim <= 2
        self.X_dim = X.ndim
        self.original_X = X.copy()
        self.original_Y = Y.copy()
        self.X = X
        self.Y = Y
        self.X_factors = [np.zeros((l, self.n_components)) for l in X.shape]
        self.Y_factors = [np.zeros((l, self.n_components)) for l in Y.shape]

    def fit(self, X, Y):
        raise NotImplementedError

    def _mean_center(self, to_predict):
        return to_predict - self.X_mean

    def predict(self, to_predict):
        assert self.X.shape[1:] == to_predict.shape[1:], \
            f"Training tensor shape is {self.X.shape}, while new tensor " \
            f"shape is {to_predict.shape}"
        to_predict = self._mean_center(to_predict)
        factors_kr = khatri_rao(self.X_factors, skip_matrix=0)
        unfolded = tl.unfold(to_predict, 0)
        scores, _, _, _ = lstsq(factors_kr, unfolded.T, rcond=-1)

        return scores.T @ self.Y_factors[1]

    def X_reconstructed(self):
        return CPTensor((None, self.X_factors)).to_tensor()

    def Y_reconstructed(self):
        if len(self.Y_factors) >= 2:
            return CPTensor((None, self.Y_factors)).to_tensor()
        else:
            return self.Y_factors[0]
