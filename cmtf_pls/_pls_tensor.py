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

def factors_to_tensor(factors):
    return CPTensor((None, factors)).to_tensor()


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

    def preprocess(self, X, Y):
        # check input integrity
        assert X.shape[0] == Y.shape[0]
        assert Y.ndim <= 2, "Only a matrix (2-mode tensor) Y is acceptable."
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        # mean center the data; set up factors
        self.X_dim = X.ndim
        self.original_X = X.copy()
        self.original_Y = Y.copy()
        self.X_factors = [np.zeros((l, self.n_components)) for l in X.shape]
        self.Y_factors = [np.tile(Y[:, [0]], self.n_components), np.zeros((Y.shape[1], self.n_components))]
            # U takes the 1st column of Y

        self.X_mean = np.mean(X, axis=0)
        self.Y_mean = np.mean(Y, axis=0)
        return X - self.X_mean, Y - self.Y_mean


    def fit(self, X, Y):
        X, Y = self.preprocess(X, Y)
        raise NotImplementedError


    def predict(self, to_predict):
        assert self.original_X.shape[1:] == to_predict.shape[1:], \
            f"Training tensor shape is {self.original_X.shape}, while new tensor " \
            f"shape is {to_predict.shape}"
        to_predict -= self.X_mean
        factors_kr = khatri_rao(self.X_factors, skip_matrix=0)
        unfolded = tl.unfold(to_predict, 0)
        scores, _, _, _ = lstsq(factors_kr, unfolded.T, rcond=-1)

        return scores.T @ self.Y_factors[1]

    def X_reconstructed(self):
        return factors_to_tensor(self.X_factors) + self.X_mean

    def Y_reconstructed(self):
        return factors_to_tensor(self.Y_factors) + self.Y_mean

    def mean_centered_R2X(self):
        return calcR2X(self.original_X - self.X_mean, factors_to_tensor(self.X_factors))

    def mean_centered_R2Y(self):
        return calcR2X(self.original_Y - self.Y_mean, factors_to_tensor(self.Y_factors))




