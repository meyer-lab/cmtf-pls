from cmtf_pls.atpls import *
import pytest

@pytest.mark.parametrize("n_y", [1, 3, 5])
def test_tPLS(n_y):
    np.random.seed(25)
    X = np.random.rand(20, 8, 6, 4)
    Y = np.random.rand(20, n_y)
    R2Xs, R2Ys = [], []
    for r in range(1, 12):
        tpls = tPLS(r)
        tpls.fit(X, Y)
        R2Xs.append(tpls.mean_centered_R2X())
        R2Ys.append(tpls.mean_centered_R2Y())
    R2Xds = np.array([R2Xs[i + 1] - R2Xs[i] for i in range(len(R2Xs) - 1)])
    R2Yds = np.array([R2Ys[i + 1] - R2Ys[i] for i in range(len(R2Ys) - 1)])
    print(R2Xs, R2Ys)
    assert np.all(np.array(R2Xds) > 0.0), "R2X is not monotonically increasing"
    assert np.all(np.array(R2Yds) > 0.0), \
        f"R2Y is not monotonically increasing. " \
        f"Streak till {np.where(R2Yds <= 0.0)[0][0] + 1}-th component, " \
        f"R2Y = {R2Ys[np.where(R2Yds <= 0.0)[0][0]]}. Y shape = {Y.shape}"


def test_kron_equivalence():
    np.random.seed(25)
    X = np.random.rand(20, 8, 6, 4)
    Y = np.random.rand(20, 5)
    tpls = tPLS(3)
    tpls.fit(X, Y)
    a = 1

    # Method 1: Kronecker method
    weight = [ff[:, a] for ff in tpls.X_factors[1:]]
    weight = kronecker([weight[m] for m in reversed(range(len(weight)))])
    weight = np.expand_dims(weight, axis=1)  # (X_shape[1:], 1)
    X_fac = np.expand_dims(tpls.X_factors[0][:, a], axis=1)  # (X_shape[0], 1)
    X_subtract = (X_fac @ weight.T).reshape(*X.shape, order='F')

    # Method 2: Original tensor product
    X_orig = factors_to_tensor([ff[:, a].reshape(-1, 1) for ff in tpls.X_factors])

    assert np.all(np.isclose(X_subtract, X_orig))


