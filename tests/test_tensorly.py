from cmtf_pls.tpls import *
from tensorly.regression.cp_plsr import *

"""
def test_tensorly_same_results():
    # Test that tPLS here in CP mode is the same as the Tensorly version
    X = np.random.rand(10,8,7,6)
    Y = np.random.rand(10,4)

    tly = CP_PLSR(3, tol=1e-15)
    tpack = tPLS(3)
    tly.fit(X, Y)
    tpack.fit(X, Y, tol=1e-15)

    for ii in range(len(X.shape)):
        assert np.all(np.logical_or(
            np.all(np.isclose(tly.X_factors[ii], tpack.X_factors[ii], rtol=1e-5), axis=0),
            np.all(np.isclose(tly.X_factors[ii], -tpack.X_factors[ii], rtol=1e-5), axis=0),
        )), f"Agreement with Tensorly CP_PLSR: X factor {ii} are not the same."

    for ii in range(len(Y.shape)):
        assert np.all(np.logical_or(
            np.all(np.isclose(tly.Y_factors[ii], tpack.Y_factors[ii], rtol=1e-5), axis=0),
            np.all(np.isclose(tly.Y_factors[ii], -tpack.Y_factors[ii], rtol=1e-5), axis=0),
        )), f"Agreement with Tensorly CP_PLSR: Y factor {ii} are not the same."

"""