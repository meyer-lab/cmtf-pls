# include all packages, including those needed for the children classes
from abc import ABCMeta
from collections.abc import Mapping
from copy import copy

import numpy as np
from numpy.linalg import norm, lstsq
from tensorly.tenalg import multi_mode_dot
from tensorly.decomposition._cp import parafac

from .tpls import calcR2X, factors_to_tensor


class ctPLS(Mapping, metaclass=ABCMeta):
    """ Coupled tensor PLS """
    def __init__(self, n_components:int):
        super().__init__()
        # Parameters
        self.n_components = n_components

    def __getitem__(self, index):
        if index == 0:
            return self.Xs_factors
        elif index == 1:
            return self.Y_factors
        elif index == 2:
            return self.coef_
        else:
            raise IndexError

    def __iter__(self):
        yield self.Xs_factors
        yield self.Y_factors
        yield self.coef_

    def __len__(self):
        return 3

    def copy(self):
        return copy(self)

    def preprocess(self, Xs, Y):
        # check input integrity
        assert isinstance(Xs, list)

        for X in Xs:
            assert X.shape[0] == Y.shape[0]
            assert X.ndim >= 2
        assert Y.ndim <= 2, "Only a matrix (2-mode tensor) Y is acceptable."
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        # mean center the data; set up factors
        self.Xs_len = len(Xs)
        self.Xs_dim = [X.ndim for X in Xs]
        self.Xs_shape = [X.shape for X in Xs]
        self.Y_shape = Y.shape
        self.original_Xs = [X.copy() for X in Xs]
        self.original_Y = Y.copy()
        self.factor_T = np.zeros((self.Y_shape[0], self.n_components))
        self.Xs_factors = [[self.factor_T] + [np.zeros((l, self.n_components)) for l in X.shape[1:]] for X in Xs]
        self.Y_factors = [np.zeros((l, self.n_components)) for l in Y.shape]

        self.Xs_mean = [np.mean(X, axis=0) for X in Xs]
        self.Y_mean = np.mean(Y, axis=0)
        self.coef_ = np.zeros((self.n_components, self.n_components))   # a upper triangular matrix
        return [X - self.Xs_mean[i] for (i, X) in enumerate(Xs)], Y - self.Y_mean


    def fit(self, Xs, Y, tol=1e-8, max_iter=100, verbose=0):
        Xs, Y = self.preprocess(Xs, Y)
        for a in range(self.n_components):
            oldU = np.ones_like(self.Y_factors[0][:, a]) * np.inf
            self.Y_factors[0][:, a] = Y[:, 0]
            for iter in range(max_iter):
                for (ti, X) in enumerate(Xs):
                    Z = np.einsum("i...,i...->...", X, self.Y_factors[0][:, a])
                    Z_comp = [Z / norm(Z)]
                    if Z.ndim >= 2:
                        Z_comp = parafac(Z, 1, tol=tol, init="svd", normalize_factors=True)[1]
                    for ii in range(Z.ndim):
                        self.Xs_factors[ti][ii + 1][:, a] = Z_comp[ii].flatten()

                Ts = [multi_mode_dot(Xs[ti],
                                     [ff[:, a] for ff in self.Xs_factors[ti][1:]],
                                     range(1, self.Xs_dim[ti])) for ti in range(self.Xs_len)]
                self.factor_T[:, a] = np.average(Ts, axis=0)
                self.Y_factors[1][:, a] = Y.T @ self.factor_T[:, a]
                self.Y_factors[1][:, a] /= norm(self.Y_factors[1][:, a])
                self.Y_factors[0][:, a] = Y @ self.Y_factors[1][:, a]
                if norm(oldU - self.Y_factors[0][:, a]) < tol:
                    if verbose:
                        print("Comp {}: converged after {} iterations".format(a, iter))
                    break
                oldU = self.Y_factors[0][:, a].copy()

            for (ti, X) in enumerate(Xs):
                X -= factors_to_tensor([ff[:, [a]] for ff in self.Xs_factors[ti]])
            self.coef_[:, a] = lstsq(self.factor_T, self.Y_factors[0][:, a], rcond=-1)[0]
            Y -= self.factor_T @ self.coef_[:, [a]] @ self.Y_factors[1][:, [a]].T
            # Y -= T b q' = T pinv(T) u q' = T lstsq(T, u) q'; b = inv(T'T) T' u = pinv(T) u


    def predict(self, Xs):
        Xs = [X.copy() for X in Xs]
        for (ti, X) in enumerate(Xs):
            if self.Xs_shape[ti][1:] != X.shape[1:]:
                raise ValueError(f"Training X{ti} has shape {self.Xs_shape[ti]}, while the new X has shape {X.shape}")
            Xs[ti] -= self.Xs_mean[ti]
        X_projection = np.zeros((Xs[0].shape[0], self.n_components))
        for a in range(self.n_components):
            X_projection[:, a] = np.average([multi_mode_dot(Xs[ti],
                                                            [ff[:, a] for ff in self.Xs_factors[ti][1:]],
                                                            range(1, self.Xs_dim[ti])) for ti in range(self.Xs_len)],
                                            axis=0)
            for (ti, X) in enumerate(Xs):
                X -= factors_to_tensor([X_projection[:, [a]]] + [ff[:, [a]] for ff in self.Xs_factors[ti][1:]])
        return X_projection @ self.coef_ @ self.Y_factors[1].T + self.Y_mean


    def transform(self, Xs, Y=None):
        Xs = [X.copy() for X in Xs]
        for (ti, X) in enumerate(Xs):
            if self.Xs_shape[ti][1:] != X.shape[1:]:
                raise ValueError(f"Training X{ti} has shape {self.Xs_shape[ti]}, while the new X has shape {X.shape}")
            Xs[ti] -= self.Xs_mean[ti]
        X_scores = np.zeros((Xs[0].shape[0], self.n_components))

        for a in range(self.n_components):
            Ts = [multi_mode_dot(Xs[ti],
                                 [ff[:, a] for ff in self.Xs_factors[ti][1:]],
                                 range(1, self.Xs_dim[ti])) for ti in range(self.Xs_len)]
            X_scores[:, a] = np.average(Ts, axis=0)
            for (ti, X) in enumerate(Xs):
                X -= factors_to_tensor([X_scores[:, [a]]] + [ff[:, [a]] for ff in self.Xs_factors[ti][1:]])

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

    def R2X(self, idx):
        # defined as after mean-centering
        return calcR2X(self.original_Xs[idx] - self.Xs_mean[idx], factors_to_tensor(self.Xs_factors[idx]))

    def R2Xs(self):
        return np.array([self.R2X(i) for i in range(self.Xs_len)])

    def R2Y(self):
        # defined as after mean-centering
        return calcR2X(self.original_Y - self.Y_mean, self.predict(self.original_Xs) - self.Y_mean)