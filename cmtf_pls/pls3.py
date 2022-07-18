import numpy as np
import tensorly as tl
from tensorly.tenalg import khatri_rao

from ._pls_tensor import *


class ThreeModePLS(PLSTensor):
    def __init__(self, *args):
        super().__init__(*args)

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
