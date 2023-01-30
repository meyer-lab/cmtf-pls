from cmtf_pls.tpls import *
from tensorly.regression.cp_plsr import *


def test_tensorly_same_results():
    X = np.random.rand(10,8,7,6)
    Y = np.random.rand(10,4)

    tly = CP_PLSR(3, tol=1e-15)
    tpack = tPLS(3)
    tly.fit(X, Y)
    tpack.fit(X, Y, tol=1e-15, method="cp")

    for ii in range(len(X.shape)):
        assert np.all(np.logical_or(
            np.all(np.isclose(tly.X_factors[ii], tpack.X_factors[ii], rtol=1e-5), axis=0),
            np.all(np.isclose(tly.X_factors[ii], -tpack.X_factors[ii], rtol=1e-5), axis=0),
        )), f"Agreement with Tensorly CP_PLSR: X factor {ii} are not the same."
