import numpy as np
from numpy.linalg import norm

def wold_nipals0(X, Y, num_comp = 2, eps = 1e-7, max_iter = 100):
    X -= np.mean(X, axis=0)
    Y -= np.mean(Y, axis=0)
    n, k, m = X.shape[0], X.shape[1], Y.shape[1]
    assert n == Y.shape[0]
    T = np.zeros((n, num_comp))
    U = np.zeros((n, num_comp))
    W = np.zeros((k, num_comp))
    P = np.zeros((k, num_comp))
    C = np.zeros((m, num_comp))
    for a in range(num_comp):
        t_old = np.zeros(n)
        U[:, a] = Y[:, 0]
        for _ in range(max_iter):
            W[:, a] = X.T @ U[:, a] / norm(U[:, a])**2
            W[:, a] /= norm(W[:, a])
            T[:, a] = X @ W[:, a]
            C[:, a] = Y.T @ T[:, a] / norm(T[:, a])**2
            U[:, a] = Y @ C[:, a] / norm(C[:, a])**2
            if norm(t_old - T[:, a])/norm(T[:, a]) < eps:
                break
            t_old = T[:, a].copy()
        P[:, a] = X.T @ T[:, a] / norm(T[:, a])**2
        X -= np.outer(T[:, a], P[:, a])
        Y -= np.outer(T[:, a], C[:, a])
    return (T, P), (U, C), W

def wold_nipals(X, Y, num_comp = 2):
    """
    X = T P' + errors
    Y = U Q' + errors = T C' + errors
    """
    X -= np.mean(X, axis=0)
    Y -= np.mean(Y, axis=0)
    n, k, m = X.shape[0], X.shape[1], Y.shape[1]
    assert n == Y.shape[0]
    T = np.zeros((n, num_comp))
    U = np.zeros((n, num_comp))
    W = np.zeros((k, num_comp))
    P = np.zeros((k, num_comp))
    C = np.zeros((m, num_comp))
    Q = np.zeros((m, num_comp))
    for a in range(num_comp):
        W[:, a] = np.linalg.svd(X.T @ Y)[0][:, 0]
        T[:, a] = X @ W[:, a]
        P[:, a] = X.T @ T[:, a] / norm(T[:, a]) ** 2
        C[:, a] = Y.T @ T[:, a] / norm(T[:, a]) ** 2
        U[:, a] = Y @ C[:, a]
        Q[:, a] = Y.T @ U[:, a] / norm(U[:, a]) ** 2
        X -= np.outer(T[:, a], P[:, a])
        Y -= np.outer(T[:, a], C[:, a])
    return T, U, W, P, C, Q

def tensorPLS(X, Y):
    pass