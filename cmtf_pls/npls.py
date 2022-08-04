# include all packages, including those needed for the children classes
from abc import ABCMeta
from collections.abc import Mapping
from copy import copy

import numpy as np
from numpy.linalg import pinv, norm, lstsq
import tensorly as tl
from tensorly.cp_tensor import CPTensor
from tensorly.tenalg import khatri_rao, mode_dot, multi_mode_dot
from tensorly.decomposition import tucker


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


class NPLS(Mapping, metaclass=ABCMeta):
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
        assert any(Y.var(axis=0) > 0), "No variance in any response variable " \
                                       "detected"
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
        return X - self.X_mean, Y - self.Y_mean


    def fit(self, X, Y, tol=1e-10, max_iter=100, verbose=0):
        X, Y = self.preprocess(X, Y)

        for a in range(self.n_components):
            oldU = np.ones_like(self.Y_factors[0][:, a]) * np.inf
            for iter in range(max_iter):
                Z = np.einsum("i...,i...->...", X, self.Y_factors[0][:, a])
                Z_comp = tucker(Z, [1] * Z.ndim)[1] if Z.ndim >= 2 else [Z / norm(Z)]
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

            X -= factors_to_tensor([ff[:, a].reshape(-1, 1) for ff in self.X_factors])
            Y -= self.X_factors[0] @ pinv(self.X_factors[0]) @ self.Y_factors[0][:, [a]] @ \
                 self.Y_factors[1][:, [a]].T  # Y -= T pinv(T) u q'


    def predict(self, X):
        if self.X_shape[1:] != X.shape[1:]:
            raise ValueError(f"Training X has shape {self.X_shape}, while the new X has shape {X.shape}")
        X -= self.X_mean
        factors_kr = khatri_rao(self.X_factors, skip_matrix=0)
        unfolded = tl.unfold(X, 0)
        scores = lstsq(factors_kr, unfolded.T, rcond=-1)[0]
        estimators = lstsq(self.X_factors[0], self.Y_factors[0], rcond=-1)[0]

        return scores.T @ estimators @ self.Y_factors[1].T


    def transform(self, X, Y=None):
        if self.X_shape[1:] != X.shape[1:]:
            raise ValueError(f"Training X has shape {self.X_shape}, while the new X has shape {X.shape}")
        X = X.copy()
        X -= self.X_mean
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
        return factors_to_tensor(self.X_factors) + self.X_mean

    def Y_reconstructed(self):
        return factors_to_tensor(self.Y_factors) + self.Y_mean

    def mean_centered_R2X(self):
        return calcR2X(self.original_X - self.X_mean, factors_to_tensor(self.X_factors))

    def mean_centered_R2Y(self):
        return calcR2X(self.original_Y - self.Y_mean, factors_to_tensor(self.Y_factors))




