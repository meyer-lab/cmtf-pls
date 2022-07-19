from tensorly.decomposition import tucker
from ._pls_tensor import *

class NModePLS(PLSTensor):
    def __init__(self, *args):
        super().__init__(*args)

    def fit(self, X, Y, tol=1e-10, max_iter=100):
        X, Y = self.preprocess(X, Y)

        for a in range(self.n_components):
            oldU = np.ones_like(self.Y_factors[0][:, a]) * np.inf
            for iter in range(max_iter):
                Z = np.einsum("i...,i...->...", X, self.Y_factors[0][:, a])
                Z_comp = tucker(Z, 1)[1] if Z.ndim >= 2 else [Z / norm(Z)]
                for ii in range(Z.ndim):
                    self.X_factors[ii + 1][:, a] = Z_comp[ii].flatten()

                self.X_factors[0][:, a] = multi_mode_dot(X, [ff[:, a] for ff in self.X_factors[1:]], range(1, X.ndim))
                self.Y_factors[1][:, a] = Y.T @ self.X_factors[0][:, a]
                self.Y_factors[1][:, a] /= norm(self.Y_factors[1][:, a])
                self.Y_factors[0][:, a] = Y @ self.Y_factors[1][:, a]
                if norm(oldU - self.Y_factors[0][:, a]) < tol:
                    print("Comp {}: converged after {} iterations".format(a, iter))
                    break
                oldU = self.Y_factors[0][:, a].copy()

            X -= factors_to_tensor([ff[:, a].reshape(-1, 1) for ff in self.X_factors])
            Y -= self.X_factors[0] @ pinv(self.X_factors[0]) @ self.Y_factors[0][:, [a]] @\
                 self.Y_factors[1][:, [a]].T    # Y -= T pinv(T) u q'
            

