import numpy as np
import tensorly as tl
from tensorly.tenalg import khatri_rao

from ._pls_tensor import *


class ThreeModePLS(PLSTensor):
    def __init__(self, *args):
        super().__init__(*args)

    def fit(self, X, Y, tol=1e-10, max_iter=100):
        self.preprocess()
        X, Y = self.X.copy(), self.Y.copy()
        assert X.ndim == 3
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        assert Y.ndim == 2
        self.Xfacs = [np.zeros((l, self.n_components)) for l in X.shape]
        self.Yfacs = [np.tile(Y[:, [0]], self.n_components), np.zeros((Y.shape[1], self.n_components))]  # U takes 1st col of Y

        assert X.shape[0] == Y.shape[0]
        for a in range(self.n_components):
            oldU = np.ones_like(self.Yfacs[0][:, a]) * np.inf

            for iter in range(max_iter):

                svd_res = np.linalg.svd(mode_dot(X, self.Yfacs[0][:, a], 0))
                self.Xfacs[1][:, a] = svd_res[0][:, 0]
                self.Xfacs[2][:, a] = svd_res[2].T[:, 0]
                Tt = X.copy()
                Tt = mode_dot(Tt, self.Xfacs[1][:, a], 1)
                Tt = mode_dot(Tt, self.Xfacs[2][:, a], 1)
                self.Xfacs[0][:, a] = Tt


                self.Yfacs[1][:, a] = Y.T @ self.Xfacs[0][:, a]
                self.Yfacs[1][:, a] /= norm(self.Yfacs[1][:, a])
                self.Yfacs[0][:, a] = Y @ self.Yfacs[1][:, a]
                if norm(oldU - self.Yfacs[0][:, a]) < tol:
                    print("Comp {}: converged after {} iterations".format(a, iter))
                    break
                oldU = self.Yfacs[0][:, a].copy()

            X -= CPTensor((None, [f[:, a].reshape(-1, 1) for f in self.Xfacs])).to_tensor()
            Y -= self.Xfacs[0] @ pinv(self.Xfacs[0]) @ self.Yfacs[0][:, [a]] @ self.Yfacs[1][:, [a]].T  # T pinv(T) u q'
