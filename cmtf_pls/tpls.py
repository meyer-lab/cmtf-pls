# include all packages, including those needed for the children classes
from abc import ABCMeta
from collections.abc import Mapping
from copy import copy
from functools import reduce

import numpy as np
from numpy.linalg import norm, lstsq
from tensorly.cp_tensor import CPTensor
from tensorly.tenalg import multi_mode_dot
from tensorly.decomposition._cp import parafac

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


def miss_tensordot(X, u, missX=None):
    # Equivalent to np.einsum("i...,i...->...", X, u), but X with missing values at missX
    Xdim = X.shape
    assert Xdim[0] == u.shape[0]
    if missX is None:
        missX = np.isnan(X)
    X = X.reshape(Xdim[0], -1)
    missX = missX.reshape(Xdim[0], -1)
    w = np.zeros((X.shape[1],))
    for i in range(X.shape[1]):
        m = np.where(~missX[:, i])[0]
        if len(m) > 0:
            w[i] = X[m, i].T @ u[m] / len(m) * Xdim[0]
    return w.reshape(Xdim[1:])

def miss_mmodedot(X, facs, missX=None):
    # Equivalent to multi_mode_dot(X, fac, range(1, X.ndim)), but X with missing values at missX
    # facs ~= [ff[:, a] for ff in self.X_factors[1:]]
    Xdim = X.shape
    assert all([(Xdim[i+1], ff.shape[0]) for (i, ff) in enumerate(facs)])
    if missX is None:
        missX = np.isnan(X)
    X = X.reshape(Xdim[0], -1)
    missX = missX.reshape(Xdim[0], -1)
    t = np.zeros((Xdim[0],))
    wkron = reduce(np.kron, facs)
    for i in range(Xdim[0]):
        m = np.where(~missX[i, :])[0]
        t[i] = X[i, m] @ wkron[m]
    return t


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
        elif index == 2:
            return self.coef_
        else:
            raise IndexError

    def __iter__(self):
        yield self.X_factors
        yield self.Y_factors
        yield self.coef_

    def __len__(self):
        return 3

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
        self.Y_shape = Y.shape
        self.original_X = X.copy()
        self.original_Y = Y.copy()
        self.X_factors = [np.zeros((l, self.n_components)) for l in X.shape]
        self.Y_factors = [np.zeros((l, self.n_components)) for l in Y.shape]
            # U takes the 1st column of Y

        self.X_hasMiss = np.any(np.isnan(X))
        if self.X_hasMiss:
            print("X has missing values")
        self.X_miss = np.isnan(X)   # positions of missing value, not the opposite

        self.X_mean = np.nanmean(X, axis=0)
        self.Y_mean = np.nanmean(Y, axis=0)
        self.coef_ = np.zeros((self.n_components, self.n_components))   # a upper triangular matrix
        return X - self.X_mean, Y - self.Y_mean


    def fit(self, X, Y, tol=1e-8, max_iter=100, verbose=0):
        X, Y = self.preprocess(X, Y)
        for a in range(self.n_components):
            oldU = np.ones_like(self.Y_factors[0][:, a]) * np.inf
            self.Y_factors[0][:, a] = Y[:, 0]
            for iter in range(max_iter):
                if self.X_hasMiss:
                    Z = miss_tensordot(X, self.Y_factors[0][:, a], self.X_miss)
                else:
                    Z = np.einsum("i...,i...->...", X, self.Y_factors[0][:, a])
                # missing value in X: solve einsum customedly; missing value in Y: PCA-NIPALS
                Z_comp = [Z / norm(Z)]
                if Z.ndim >= 2:
                    Z_comp = parafac(Z, 1, tol=tol, init="svd", normalize_factors=True)[1]
                for ii in range(Z.ndim):
                    self.X_factors[ii + 1][:, a] = Z_comp[ii].flatten()

                if self.X_hasMiss:
                    self.X_factors[0][:, a] = miss_mmodedot(X, [ff[:, a] for ff in self.X_factors[1:]], self.X_miss)
                else:
                    self.X_factors[0][:, a] = multi_mode_dot(X, [ff[:, a] for ff in self.X_factors[1:]], range(1, self.X_dim))
                self.Y_factors[1][:, a] = Y.T @ self.X_factors[0][:, a]
                self.Y_factors[1][:, a] /= norm(self.Y_factors[1][:, a])
                self.Y_factors[0][:, a] = Y @ self.Y_factors[1][:, a]
                if norm(oldU - self.Y_factors[0][:, a]) < tol:
                    if verbose:
                        print("Comp {}: converged after {} iterations".format(a, iter))
                    break
                oldU = self.Y_factors[0][:, a].copy()

            X -= factors_to_tensor([ff[:, [a]] for ff in self.X_factors])
            self.coef_[:, a] = lstsq(self.X_factors[0], self.Y_factors[0][:, a], rcond=-1)[0]
            Y -= self.X_factors[0] @ self.coef_[:, [a]] @ self.Y_factors[1][:, [a]].T
            # Y -= T b q' = T pinv(T) u q' = T lstsq(T, u) q'; b = inv(T'T) T' u = pinv(T) u


    def predict(self, X):
        if self.X_shape[1:] != X.shape[1:]:
            raise ValueError(f"Training X has shape {self.X_shape}, while the new X has shape {X.shape}")

        X = X.copy() - self.X_mean
        X_projection = np.zeros((X.shape[0], self.n_components))
        for a in range(self.n_components):
            X_projection[:, a] = multi_mode_dot(X, [ff[:, a] for ff in self.X_factors[1:]], range(1, self.X_dim))
            X -= factors_to_tensor([X_projection[:, [a]]] + [ff[:, [a]] for ff in self.X_factors[1:]])
        return X_projection @ self.coef_ @ self.Y_factors[1].T + self.Y_mean


    def transform(self, X, Y=None):
        if self.X_shape[1:] != X.shape[1:]:
            raise ValueError(f"Training X has shape {self.X_shape}, while the new X has shape {X.shape}")
        X = X.copy() - self.X_mean
        X_scores = np.zeros((X.shape[0], self.n_components))

        for a in range(self.n_components):
            X_scores[:, a] = multi_mode_dot(X, [ff[:, a] for ff in self.X_factors[1:]], range(1, self.X_dim))
            X -= factors_to_tensor([X_scores[:, [a]]] + [ff[:, [a]] for ff in self.X_factors[1:]])

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
                Y -= X_scores @ self.coef_[:, [a]] @ self.Y_factors[1][:, [a]].T
            return X_scores, Y_scores

        return X_scores

    def X_reconstructed(self):
        return factors_to_tensor(self.X_factors) + self.X_mean

    def Y_reconstructed(self):
        return self.predict(self.original_X) + self.Y_mean

    def R2X(self):
        # defined as after mean-centering
        return calcR2X(self.original_X - self.X_mean, factors_to_tensor(self.X_factors))

    def R2Y(self):
        # defined as after mean-centering
        return calcR2X(self.original_Y - self.Y_mean, self.predict(self.original_X) - self.Y_mean)