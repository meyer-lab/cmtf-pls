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
        self._init_factors(X, Y)
        self.preprocess()
        X, Y = self.X.copy(), self.Y.copy()
        assert X.ndim == 3
        assert Y.ndim == 1
        self.Xfacs = [np.zeros((l, self.num_comp)) for l in X.shape]
        assert X.shape[0] == Y.shape[0]
        for a in range(self.num_comp):
            svd_res = np.linalg.svd(mode_dot(X, Y, 0))
            self.Xfacs[1][:, a] = svd_res[0][:, 0]
            self.Xfacs[2][:, a] = svd_res[2].T[:, 0]
            Tt = X.copy()
            Tt = mode_dot(Tt, self.Xfacs[1][:, a], 1)
            Tt = mode_dot(Tt, self.Xfacs[2][:, a], 1)
            self.Xfacs[0][:, a] = Tt
            X = X - CPTensor((None, self.Xfacs)).to_tensor()
            Y = Y - self.Xfacs[0] @ pinv(self.Xfacs[0]) @ Y

        Y_loading, _, _, _ = np.linalg.lstsq(self.Xfacs[0], self.Y, rcond=-1)
        self.Yfacs = [
            self.Xfacs[0] @ pinv(self.Xfacs[0]) @ self.Y,
            Y_loading
        ]

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
        if self.Y.ndim >= 2:
            return CPTensor((None, self.Yfacs)).to_tensor()
        else:
            return self.Yfacs[0]


def get_q2y(pls_tensor):
    """
    Calculates Q2Y for a fitted PLS tensor.

    Args:
        pls_tensor (PLSTensor): fitted PLS tensor.

    Returns:
        Q2Y (float): Q2Y of PLS tensor applied to fitted dataset.
    """
    assert pls_tensor.original_X is not None, \
        'PLS Tensor must be fit prior to calculating Q2Y'
    X = pls_tensor.original_X
    Y = pls_tensor.original_Y
    q2y_plsr = PLSTensor(pls_tensor.num_comp)

    loo = LeaveOneOut()
    Y_pred = np.zeros(Y.shape)
    Y_actual = np.zeros(Y.shape)
    for train_index, test_index in loo.split(X, Y):
        X_train, Y_train = X[train_index], Y[train_index]
        X_test, Y_test = X[test_index], Y[test_index]
        q2y_plsr.fit(X_train, Y_train)

        Y_pred[test_index] = q2y_plsr.predict(X_test)
        Y_actual[test_index] = Y_test

    numerator = (Y_pred - Y_actual) ** 2
    denominator = (Y_actual) ** 2
    return 1 - numerator.sum() / denominator.sum()
