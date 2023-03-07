import numpy as np
from cmtf_pls.cmtf import *
from cmtf_pls.tpls import tPLS

def test_multi_mmodedot():
    X = np.random.rand(10, 9, 8, 7)
    Y = np.random.rand(10, 5)
    X -= np.mean(X, axis=0)
    Y -= np.mean(Y, axis=0)
    n_components = 4
    X_factors = [np.zeros((l, n_components)) for l in X.shape]
    Z = np.einsum("i...,i...->...", X, Y[:, 0])
    Z_comp = parafac(Z, 1, tol=1e-8, init="svd", normalize_factors=True)[1]
    for ii in range(Z.ndim):
        X_factors[ii + 1][:, 0] = Z_comp[ii].flatten()

    oldT = multi_mode_dot(X, [ff[:, 0] for ff in X_factors[1:]], range(1, X.ndim))
    newT = multi_mmodedot([X], [[ff[:, 0] for ff in X_factors[1:]]])
    assert np.allclose(oldT, newT)

def test_oneX_equivalence():
    X = np.random.rand(10, 9, 8, 7)
    Y = np.random.rand(10, 5)
    for r in range(1, 6):
        pls0 = tPLS(r)
        pls0.fit(X, Y)
        pls1 = ctPLS(r)
        pls1.fit([X], Y)
        assert np.allclose(pls0.R2X(), pls1.R2X(0))

def test_ctPLS():
    from cmtf_pls.cmtf import *

    dims = [(10, 9, 8, 7), (10, 8, 6)]
    Xs = [np.random.rand(*d) for d in dims]
    Y = np.random.rand(10, 5)
    pls = ctPLS(3)
    pls.fit(Xs, Y)




    n_comp = 4
    facss = [[np.random.rand(di, 1) for di in d[1:]] for d in dims]