""" Test the functions for dealing with missing values """
from numpy.linalg import norm
from tensorly.tenalg import multi_mode_dot

from cmtf_pls.tpls import tPLS
from cmtf_pls.missingvals import *
from cmtf_pls.synthetic import import_synthetic



def test_miss_tensordot():
    # Test it is equivalent to np.einsum("i...,i...->...", X, u)
    X = np.random.rand(10, 5, 4, 3)
    X[np.random.rand(*X.shape) < 0.1] = np.nan
    missX = np.isnan(X)
    u = np.random.rand(10)
    w = miss_tensordot(X, u, missX.reshape(X.shape[0], -1))
    w2 = np.einsum("i...,i...->...", X, u)
    assert np.allclose(w * ~np.isnan(w2), np.nan_to_num(w2))

    for _ in range(10):
        X = np.random.rand(20, 1) @ np.random.rand(8, 1).T
        u = np.random.rand(20)
        w = X.T @ u
        X[np.random.rand(*X.shape) < 0.2] = np.nan
        w1 = miss_tensordot(X, u)
        assert norm(w - w1) / norm(w) < 0.2

def test_miss_mmodedot():
    # Test it is equivalent to multi_mode_dot(X, fac, range(1, X.ndim))
    for _ in range(10):
        X = np.random.rand(10, 9, 8, 7)
        facs = [np.random.rand(l) for l in X.shape[1:]]
        t = multi_mode_dot(X, facs, range(1, X.ndim))
        X[np.random.rand(*X.shape) < 0.1] = np.nan
        missX = np.isnan(X)
        t2 = miss_mmodedot(X, facs, missX)
        assert norm(t - t2) / norm(t) < 0.15

def test_miss_X_random():
    X = np.random.rand(10, 9, 8, 7)
    Y = np.random.rand(10, 4)
    tpls = tPLS(5)
    tpls.fit(X, Y)
    X[5, 4, 3, 2] = np.nan
    tpls1 = tPLS(5)
    tpls1.fit(X, Y)
    for i in range(X.ndim):
        fac = tpls.X_factors[i]
        fac1 = tpls1.X_factors[i]
        assert (np.sum(~np.isclose(fac, fac1, rtol=0.2)) / np.prod(fac.shape)) < 0.2
    for i in range(Y.ndim):
        fac = tpls.Y_factors[i]
        fac1 = tpls1.Y_factors[i]
        assert (np.sum(~np.isclose(fac, fac1, rtol=0.2)) / np.prod(fac.shape)) < 0.2

def test_miss_X_synthetic():
    X, Y, _ = import_synthetic((10, 9, 8, 7), 4, 5)
    tpls = tPLS(5)
    tpls.fit(X, Y)
    X[np.random.rand(*X.shape) < 0.01] = np.nan
    tpls1 = tPLS(5)
    tpls1.fit(X, Y)
    for i in range(X.ndim):
        fac = tpls.X_factors[i]
        fac1 = tpls1.X_factors[i]
        assert (np.sum(~np.isclose(fac, fac1, rtol=0.1)) / np.prod(fac.shape)) < 0.4
    for i in range(Y.ndim):
        fac = tpls.Y_factors[i]
        fac1 = tpls1.Y_factors[i]
        assert (np.sum(~np.isclose(fac, fac1, rtol=0.1)) / np.prod(fac.shape)) < 0.4


