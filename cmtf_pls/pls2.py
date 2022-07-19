from ._pls_tensor import *

class TwoModePLS(PLSTensor):
    def __init__(self, *args):
        super().__init__(*args)

    def fit(self):
        """
        Adapted from Wold's NIPALS algorithm
        X = T P' + errors
        Y = U Q' + errors = T C' + errors
        """
        self.preprocess()
        X, Y = self.X.copy(), self.Y.copy()
        n, k, m = X.shape[0], X.shape[1], Y.shape[1]
        assert n == Y.shape[0]
        T = np.zeros((n, self.n_components))
        U = np.zeros((n, self.n_components))
        W = np.zeros((k, self.n_components))
        P = np.zeros((k, self.n_components))
        C = np.zeros((m, self.n_components))
        Q = np.zeros((m, self.n_components))
        for a in range(self.n_components):
            W[:, a] = np.linalg.svd(X.T @ Y)[0][:, 0]
            T[:, a] = X @ W[:, a]
            P[:, a] = X.T @ T[:, a] / norm(T[:, a]) ** 2
            C[:, a] = Y.T @ T[:, a] / norm(T[:, a]) ** 2
            U[:, a] = Y @ C[:, a]
            Q[:, a] = Y.T @ U[:, a] / norm(U[:, a]) ** 2
            X -= np.outer(T[:, a], P[:, a])
            Y -= np.outer(T[:, a], C[:, a])

        self.Xfacs = [T, P]
        self.Yfacs = [U, Q]
        self.W = W
        self.C = C

    def fit_v2(self, tol=1e-7, max_iter=100):
        """ Kept here only for testing. Tested equivalent to self.fit() """
        self.preprocess()
        X, Y = self.X.copy(), self.Y.copy()
        n, k, m = X.shape[0], X.shape[1], Y.shape[1]
        assert n == Y.shape[0]
        T = np.zeros((n, self.n_components))
        U = np.zeros((n, self.n_components))
        W = np.zeros((k, self.n_components))
        P = np.zeros((k, self.n_components))
        C = np.zeros((m, self.n_components))
        Q = np.zeros((m, self.n_components))
        for a in range(self.n_components):
            t_old = np.zeros(n)
            U[:, a] = Y[:, 0]
            for _ in range(max_iter):
                W[:, a] = X.T @ U[:, a] / norm(U[:, a]) ** 2
                W[:, a] /= norm(W[:, a])
                T[:, a] = X @ W[:, a]
                C[:, a] = Y.T @ T[:, a] / norm(T[:, a]) ** 2
                U[:, a] = Y @ C[:, a] / norm(C[:, a]) ** 2
                if norm(t_old - T[:, a]) / norm(T[:, a]) < tol:
                    break
                t_old = T[:, a].copy()
            P[:, a] = X.T @ T[:, a] / norm(T[:, a]) ** 2
            X -= np.outer(T[:, a], P[:, a])
            Y -= np.outer(T[:, a], C[:, a])
        self.Xfacs = [T, P]
        self.Yfacs = [U, Q]
        self.W = W
        self.C = C

    def predict(self, Xnew):
        B = self.W @ pinv(self.Xfacs[1].T @ self.W) @ self.Xfacs[0].T @ self.Y
        return Xnew @ B
