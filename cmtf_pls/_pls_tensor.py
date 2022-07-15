# include all packages, including those needed for the children class
import numpy as np
from collections.abc import Mapping
from abc import ABCMeta
from tensorly.cp_tensor import CPTensor
from tensorly.tenalg import mode_dot, multi_mode_dot
from numpy.linalg import pinv, norm

def calcR2X(X, Xhat):
    mask = np.isfinite(X)
    xIn = np.nan_to_num(X)
    top = np.linalg.norm(Xhat * mask - xIn) ** 2.0
    bottom = np.linalg.norm(xIn) ** 2.0
    return 1 - top / bottom

class PLSTensor(Mapping, metaclass=ABCMeta):
    """ Base class for all variants of tensor PLS """

    def __init__(self, X:np.ndarray, Y:np.ndarray, num_comp:int):
        super().__init__()
        assert X.shape[0] == Y.shape[0]
        assert Y.ndim <= 2
        self.Xdim = X.ndim
        self.X = X
        self.Y = Y
        self.Xfacs = [np.zeros((l, num_comp)) for l in X.shape]
        self.Yfacs = [np.zeros((l, num_comp)) for l in Y.shape]
        self.num_comp = num_comp

    def preprocess(self):
        self.X -= np.mean(self.X, axis=0)
        self.Y -= np.mean(self.Y, axis=0)
        return NotImplementedError

    def fit(self):
        return NotImplementedError

    def predict(self, Xnew):
        return NotImplementedError

    def x_recovered(self):
        return CPTensor((None, self.Xfacs)).to_tensor()

    def y_recovered(self):
        return CPTensor((None, self.Yfacs)).to_tensor()