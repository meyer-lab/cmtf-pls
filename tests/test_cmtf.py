import pytest
import numpy as np
from cmtf_pls.cmtf import ctPLS
from cmtf_pls.tpls import tPLS, factors_to_tensor


def test_tPLS_equivalence():
    X = np.random.rand(10, 9, 8, 7)
    Y = np.random.rand(10, 5)
    for r in range(1, 6):
        pls0 = tPLS(r)
        pls0.fit(X, Y)
        pls1 = ctPLS(r)
        pls1.fit([X], Y)
        assert np.allclose(pls0.R2X(), pls1.R2X(0))

@pytest.mark.parametrize("X0dim", [(10, 9, 8, 7), (10, 9, 8, 7, 6)])
@pytest.mark.parametrize("X1dim", [(10, 8, 7), (10, 9, 8, 7)])
@pytest.mark.parametrize("X2dim", [(10, 8), (10, 9, 8)])
def test_ctPLS_dimensions(X0dim, X1dim, X2dim):
    dims = [X0dim, X1dim, X2dim]
    Xs = [np.random.rand(*d) for d in dims]
    Y = np.random.rand(10, 5)
    pls = ctPLS(3)
    pls.fit(Xs, Y)
    assert np.allclose(pls.factor_T, pls.transform(Xs))

def test_ctPLS_increasing_R2Y():
    dims = [(10, 9, 8, 7), (10, 8, 7)]
    Xs = [np.random.rand(*d) for d in dims]
    Y = np.random.rand(10, 5)
    oldR2Xs, oldR2Y = np.array([0.0 for _ in dims]), 0.0
    for r in range(1, 6):
        pls = ctPLS(r)
        pls.fit(Xs, Y)
        assert np.all(pls.R2Xs() > oldR2Xs)
        assert pls.R2Y() > oldR2Y
        oldR2Xs = pls.R2Xs()
        oldR2Y = pls.R2Y()

def test_ctPLS_increasing_R2Y():
    dims = [(10, 9, 8, 7), (10, 8, 7)]
    n_latent = 4
    Xs = [factors_to_tensor([np.random.rand(d, n_latent) for d in ds]) for ds in dims]
    Y = np.random.rand(10, 4) @ np.random.rand(5, 4).T
    oldR2Xs, oldR2Y = np.array([0.0 for _ in dims]), 0.0
    for r in range(1, 5):
        # TODO: figure out how to keep R2Xs increasing
        pls = ctPLS(r)
        pls.fit(Xs, Y)
        assert np.all(pls.R2Xs() > oldR2Xs)
        assert pls.R2Y() >= oldR2Y
        oldR2Xs = pls.R2Xs()
        oldR2Y = pls.R2Y()


def test_ctPLS_transform():
    dims = [(10, 9, 8, 7), (10, 8, 7)]
    Xs = [np.random.rand(*d) for d in dims]
    Y = np.random.rand(10, 5)
    pls = ctPLS(3)
    pls.fit(Xs, Y)
    assert np.allclose(pls.factor_T, pls.transform(Xs))
