# include all packages, including those needed for the children classes
from abc import ABCMeta
from collections.abc import Mapping
from copy import copy

import numpy as np
from numpy.linalg import pinv, norm, lstsq
import tensorly as tl
from tensorly.cp_tensor import CPTensor
from tensorly.tenalg import khatri_rao, mode_dot, multi_mode_dot, kronecker
from tensorly.decomposition import tucker
from tensorly.decomposition._cp import parafac
from tensorly import unfold

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


class tPLS(Mapping, metaclass=ABCMeta):
    """ Base class for all variants of tensor PLS """
    def __init__(self, n_components:int):
        super().__init__()
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
        self.X_shape = X.shape
        self.original_X = X.copy()
        self.original_Y = Y.copy()
        self.X_factors = [np.zeros((l, self.n_components)) for l in X.shape]
        self.Y_factors = [np.tile(Y[:, [0]], self.n_components), np.zeros((Y.shape[1], self.n_components))]
            # U takes the 1st column of Y

        self.X_mean = np.mean(X, axis=0)
        self.Y_mean = np.mean(Y, axis=0)
        self.coef_ = np.zeros((self.n_components, self.n_components))
        return X - self.X_mean, Y - self.Y_mean


    def fit(self, X, Y, tol=1e-8, max_iter=1, verbose=0, method="cp"):
        self.preprocess(X, Y)
        for a in range(self.n_components):
            oldU = np.ones_like(self.Y_factors[0][:, a]) * np.inf
            for iter in range(max_iter):
                Z = np.einsum("i...,i...->...", X, self.Y_factors[0][:, a])
                Z_comp = [Z / norm(Z)]
                if Z.ndim >= 2:
                    if method == "cp":
                        Z_comp = parafac(Z, 1, tol=tol, init="svd", normalize_factors=True)[1]
                    elif method == "tucker": #Tucker has to be non-negative Tucker
                        Z_comp = tucker(Z, [1] * Z.ndim)[1]
                    else:
                        raise NotImplementedError
                for ii in range(Z.ndim):
                    self.X_factors[ii + 1][:, a] = Z_comp[ii].flatten()

                self.X_factors[0][:, a] = multi_mode_dot(X, [ff[:, a] for ff in self.X_factors[1:]], range(1, X.ndim))
                self.Y_factors[1][:, a] = Y.T @ self.X_factors[0][:, a]
                self.Y_factors[1][:, a] /= norm(self.Y_factors[1][:, a])
                self.Y_factors[0][:, a] = Y @ self.Y_factors[1][:, a]
                if norm(oldU - self.Y_factors[0][:, a]) < tol:
                    if verbose:
                        print("Comp {}: converged after {} iterations".format(a, iter))
                    break
                oldU = self.Y_factors[0][:, a].copy()

            X = X - factors_to_tensor([ff[:, a].reshape(-1, 1) for ff in self.X_factors])
            Y = Y - self.X_factors[0] @ pinv(self.X_factors[0]) @ self.Y_factors[0][:, [a]] @ \
                self.Y_factors[1][:, [a]].T  # Y -= T pinv(T) u q' = T lstsq(T, u) q'
            self.coef_[0:a + 1, a] = (pinv(self.X_factors[0][:, 0:a + 1]) @ self.Y_factors[0][:, [a]]).reshape(a + 1)
            if (a != self.n_components - 1):
                self.Y_factors[0][:, a + 1] = Y[:, 0]


    def predict(self, X):
        if self.X_shape[1:] != X.shape[1:]:
            raise ValueError(f"Training X has shape {self.X_shape}, while the new X has shape {X.shape}")

        X_size = X.shape
        X_projections = np.zeros((X_size[0],self.n_components))
        X_e = X.copy()
        X_e = X_e.reshape((X.shape[0],np.prod(X.shape[1:])),order='F')
        for Factor in range(1, self.n_components+1):
            weights_unfolded = kronecker([self.X_factors[m][:,Factor-1] for m in reversed(range(1,len(X.shape)))])
            weights_unfolded = weights_unfolded.reshape(weights_unfolded.shape[0], 1)
            X_projections[:,Factor-1] = np.matmul(X_e, weights_unfolded).reshape(X_e.shape[0])
            X_e -= X_projections[:,Factor-1].reshape(X_e.shape[0],1) @ weights_unfolded.T
        F = np.arange(0, self.n_components)
        return X_projections[:,F] @ self.coef_[F[:,None], F] @ self.Y_factors[1][:,F].T



    def transform(self, X, Y=None):
        if self.X_shape[1:] != X.shape[1:]:
            raise ValueError(f"Training X has shape {self.X_shape}, while the new X has shape {X.shape}")
        X = X.copy()
        #X -= self.X_mean
        X_scores = np.zeros((X.shape[0], self.n_components))

        for a in range(self.n_components):
            X_scores[:, a] = multi_mode_dot(X, [ff[:, a] for ff in self.X_factors[1:]], range(1, X.ndim))
            X -= CPTensor((None, [X_scores[:, a].reshape((-1, 1))] + [ff[:, a].reshape((-1, 1)) for ff in self.X_factors[1:]])).to_tensor()

        if Y is not None:
            Y = Y.copy()
            # Check on the shape of Y
            if (Y.ndim != 1) and (Y.ndim != 2):
                raise ValueError("Only a matrix (2-mode tensor) Y is allowed.")
            if Y.ndim == 1:
                Y = Y.reshape((-1, 1))
            if self.Y_shape[1:] != Y.shape[1:]:
                raise ValueError(f"Training Y has shape {self.Y_shape}, while the new Y has shape {Y.shape}")

            Y -= self.Y_mean
            Y_scores = np.zeros((Y.shape[0], self.n_components))
            for a in range(self.n_components):
                Y_scores[:, a] = Y @ self.Y_factors[1][:, a]
                Y -= X_scores @ pinv(X_scores) @ Y_scores[:, [a]] @ self.Y_factors[1][:, [a]].T
                    # Y -= T pinv(T) u q' = T lstsq(T, u) q'
            return X_scores, Y_scores

        return X_scores

    def X_reconstructed(self):
        return factors_to_tensor(self.X_factors)

    def Y_reconstructed(self):
        return self.predict(self.original_X)

    def mean_centered_R2X(self):
        return calcR2X(self.original_X, factors_to_tensor(self.X_factors))

    def mean_centered_R2Y(self):
        return calcR2X(self.original_Y, self.predict(self.original_X))