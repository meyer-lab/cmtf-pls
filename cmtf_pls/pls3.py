import numpy as np
import tensorly as tl
from tensorly.tenalg import khatri_rao

from ._pls_tensor import *


class ThreeModePLS(PLSTensor):
    def __init__(self, *args):
        super().__init__(*args)

    def fit(self, X, Y):
        X, Y = self.preprocess(X, Y)
        Y = Y.flatten()
        assert X.ndim == 3
        assert Y.ndim == 1
        self.X_factors = [np.zeros((l, self.n_components)) for l in X.shape]
        assert X.shape[0] == Y.shape[0]
        for a in range(self.n_components):
            svd_res = np.linalg.svd(mode_dot(X, Y, 0))
            self.X_factors[1][:, a] = svd_res[0][:, 0]
            self.X_factors[2][:, a] = svd_res[2].T[:, 0]
            Tt = X.copy()
            Tt = mode_dot(Tt, self.X_factors[1][:, a], 1)
            Tt = mode_dot(Tt, self.X_factors[2][:, a], 1)
            self.X_factors[0][:, a] = Tt
            X = X - CPTensor((None, self.X_factors)).to_tensor()
            Y = Y - self.X_factors[0] @ pinv(self.X_factors[0]) @ Y

        Y_loading, _, _, _ = np.linalg.lstsq(self.X_factors[0], self.original_Y - self.Y_mean, rcond=-1)
        self.Y_factors = [
            self.X_factors[0] @ pinv(self.X_factors[0]) @ (self.original_Y - self.Y_mean),
            Y_loading
        ]

    def fit_v2(self, X, Y, tol=1e-10, max_iter=100):
        X, Y = self.preprocess(X, Y)
        assert X.ndim == 3
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        assert Y.ndim == 2
        self.X_factors = [np.zeros((l, self.n_components)) for l in X.shape]
        self.Y_factors = [np.tile(Y[:, [0]], self.n_components), np.zeros((Y.shape[1], self.n_components))]  # U takes 1st col of Y

        assert X.shape[0] == Y.shape[0]
        for a in range(self.n_components):
            oldU = np.ones_like(self.Y_factors[0][:, a]) * np.inf

            for iter in range(max_iter):

                svd_res = np.linalg.svd(mode_dot(X, self.Y_factors[0][:, a], 0))
                self.X_factors[1][:, a] = svd_res[0][:, 0]
                self.X_factors[2][:, a] = svd_res[2].T[:, 0]
                Tt = X.copy()
                Tt = mode_dot(Tt, self.X_factors[1][:, a], 1)
                Tt = mode_dot(Tt, self.X_factors[2][:, a], 1)
                self.X_factors[0][:, a] = Tt


                self.Y_factors[1][:, a] = Y.T @ self.X_factors[0][:, a]
                self.Y_factors[1][:, a] /= norm(self.Y_factors[1][:, a])
                self.Y_factors[0][:, a] = Y @ self.Y_factors[1][:, a]
                if norm(oldU - self.Y_factors[0][:, a]) < tol:
                    print("Comp {}: converged after {} iterations".format(a, iter))
                    break
                oldU = self.Y_factors[0][:, a].copy()

            X -= CPTensor((None, [f[:, a].reshape(-1, 1) for f in self.X_factors])).to_tensor()
            Y -= self.X_factors[0] @ pinv(self.X_factors[0]) @ self.Y_factors[0][:, [a]] @ self.Y_factors[1][:, [a]].T  # T pinv(T) u q'
