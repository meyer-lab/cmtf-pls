from ._pls_tensor import *

class ThreeModePLS(PLSTensor):
    def __init__(self, *args):
        super().__init__(*args)

    def fit(self):
        self.preprocess()
        X, Y = self.X.copy(), self.Y.copy()
        assert X.ndim == 3
        assert Y.ndim == 1
        Xfacs = [np.zeros((l, self.num_comp)) for l in X.shape]
        assert X.shape[0] == Y.shape[0]
        for a in range(self.num_comp):
            svd_res = np.linalg.svd(mode_dot(X, Y, 0))
            Xfacs[1][:, a] = svd_res[0][:, 0]
            Xfacs[2][:, a] = svd_res[2].T[:, 0]
            Tt = X.copy()
            Tt = mode_dot(Tt, Xfacs[1][:, a], 1)
            Tt = mode_dot(Tt, Xfacs[2][:, a], 1)
            Xfacs[0][:, a] = Tt
            X -= CPTensor((None, Xfacs)).to_tensor()
            Y = Y - Xfacs[0] @ pinv(Xfacs[0]) @ Y
        return Xfacs, [Xfacs[0] @ pinv(Xfacs[0]) @ Y]
