from tensorly.decomposition import parafac
from ._pls_tensor import *

class NModePLS(PLSTensor):
    def __init__(self, *args):
        super().__init__(*args)

    def fit(self, tol=1e-10, max_iter=100):
        self.preprocess()
        X, Y = self.X.copy(), self.Y.copy()
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1).shape
        assert Y.ndim == 2

        self.Xfacs = [np.zeros((l, self.num_comp)) for l in X.shape]  # T, ...
        self.Yfacs = [np.tile(Y[:, [0]], self.num_comp), np.zeros((Y.shape[1], self.num_comp))]  # U, Q

        for a in range(self.num_comp):
            oldU = np.ones_like(self.Yfacs[0][:, a]) * np.inf
            for iter in range(100):
                Z = np.einsum("i...,i...->...", X, self.Yfacs[0][:, a])
                Z_CP = parafac(Z, 1)  # TODO: CP cannot guarantee maximization
                for ii in range(Z.ndim):
                    self.Xfacs[ii + 1][:, a] = Z_CP.factors[ii][:, 0] / norm(Z_CP.factors[ii][:, 0])

                self.Xfacs[0][:, a] = multi_mode_dot(X, [ff[:, 0] for ff in self.Xfacs[1:]], range(1, X.ndim))
                self.Yfacs[1][:, a] = Y.T @ self.Xfacs[0][:, a]
                self.Yfacs[1][:, a] /= norm(self.Yfacs[1][:, a])
                self.Yfacs[0][:, a] = Y @ self.Yfacs[1][:, a]
                if norm(oldU - self.Yfacs[0][:, a]) < tol:
                    print("Comp {}: converged after {} iterations".format(a, iter))
                    break
                oldU = self.Yfacs[0][:, a].copy()

            X = X - CPTensor((None, self.Xfacs)).to_tensor()
            Y = Y - self.Xfacs[0] @ np.outer((pinv(self.Xfacs[0]) @ oldU), (self.Yfacs[1][:, a]).T)
