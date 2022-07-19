# include all packages, including those needed for the children classes
from abc import ABCMeta
from collections.abc import Mapping
from copy import copy

import numpy as np
from numpy.linalg import pinv, norm, lstsq
from sklearn.model_selection import LeaveOneOut
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
    def __init__(self, num_comp:int, *args, **kwargs):
        super().__init__()
        # Parameters
        self.num_comp = num_comp

        # Variables
        self.original_X = None
        self.original_Y = None
        self.X = None
        self.Y = None
        self.Xdim = 0
        self.X_mean = None
        self.Y_mean = None

        # Factors
        self.Xfacs = None
        self.Yfacs = None

    def __getitem__(self, index):
        if index == 0:
            return self.Xfacs
        elif index == 1:
            return self.Yfacs
        else:
            raise IndexError

    def __iter__(self):
        yield self.Xfacs
        yield self.Yfacs

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
        self.Xdim = X.ndim
        self.original_X = X
        self.original_Y = Y
        self.X = X
        self.Y = Y
        self.Xfacs = [np.zeros((l, self.num_comp)) for l in X.shape]
        self.Yfacs = [np.zeros((l, self.num_comp)) for l in Y.shape]

    def fit(self, X, Y):
        raise NotImplementedError

    def _mean_center(self, to_predict):
        return to_predict - self.X_mean

    def predict(self, to_predict):
        assert self.X.shape[1:] == to_predict.shape[1:], \
            f"Training tensor shape is {self.X.shape}, while new tensor " \
            f"shape is {to_predict.shape}"
        to_predict = self._mean_center(to_predict)
        factors_kr = khatri_rao(self.Xfacs, skip_matrix=0)
        unfolded = tl.unfold(to_predict, 0)
        scores, _, _, _ = lstsq(factors_kr, unfolded.T, rcond=-1)

        return scores.T @ self.Yfacs[1]

    def x_recover(self):
        return CPTensor((None, self.Xfacs)).to_tensor()

    def y_recover(self):
        if len(self.Yfacs) >= 2:
            return CPTensor((None, self.Yfacs)).to_tensor()
        else:
            return self.Yfacs[0]
