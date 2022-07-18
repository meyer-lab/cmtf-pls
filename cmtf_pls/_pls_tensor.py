# include all packages, including those needed for the children classes
import numpy as np
from collections.abc import Mapping
from abc import ABCMeta
from tensorly.cp_tensor import CPTensor
from numpy.linalg import pinv, norm
from tensorly.tenalg import mode_dot, multi_mode_dot


def calcR2X(X, Xhat):
    if (Xhat.ndim == 2) and (X.dim == 1):
        X = X.reshape(-1, 1)
    assert X.shape == Xhat.shape
    mask = np.isfinite(X)
    xIn = np.nan_to_num(X)
    top = norm(Xhat * mask - xIn) ** 2.0
    bottom = norm(xIn) ** 2.0
    return 1 - top / bottom

class PLSTensor(Mapping, metaclass=ABCMeta):
    """ Base class for all variants of tensor PLS """
    def __init__(self, X:np.ndarray, Y:np.ndarray, num_comp:int, *args, **kwargs):
        super().__init__()
        assert X.shape[0] == Y.shape[0]
        assert Y.ndim <= 2
        self.Xdim = X.ndim
        self.X = X
        self.Y = Y
        self.Xfacs = [np.zeros((l, num_comp)) for l in X.shape]
        self.Yfacs = [np.zeros((l, num_comp)) for l in Y.shape]
        self.num_comp = num_comp

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

    def preprocess(self):
        self.X -= np.mean(self.X, axis=0)
        self.Y -= np.mean(self.Y, axis=0)

    def fit(self):
        raise NotImplementedError

    def predict(self, Xnew):
        assert self.X.shape[1:] == Xnew.shape[1:], "X shape is {}, while new X shape is {}".format(self.X.shape, Xnew.shape)
        raise NotImplementedError

    def x_recover(self):
        return CPTensor((None, self.Xfacs)).to_tensor()

    def y_recover(self):
        if len(self.Yfacs) >= 2:
            return CPTensor((None, self.Yfacs)).to_tensor()
        else:
            return self.Yfacs[0]
