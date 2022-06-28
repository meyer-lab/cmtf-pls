from cmtf_pls import __version__
import numpy as np


def test_version():
    assert __version__ == '0.1.0'


def test_wold_nipals():
    X = np.random.rand(10, 6)
    Y = np.random.rand(10, 4)
    X -= np.mean(X, axis=0)
    Y -= np.mean(Y, axis=0)

    (T1, P1), (U1, C1), W1 = wold_nipals(X, Y, num_comp = 2)

    np.random.shuffle(X.T)
    np.random.shuffle(Y.T)
    (T2, P2), (U2, C2), W2 = wold_nipals(X, Y, num_comp=2)

    pass