import numpy as np
from cmtf_pls.cmtf import ctPLS

def test_ctPLS():
    from cmtf_pls.cmtf import *

    dims = [(10, 9, 8, 7), (10, 8, 6)]
    Xs = [np.random.rand(*d) for d in dims]
    Y = np.random.rand(10, 5)
    pls = ctPLS(3)
    pls.fit(Xs, Y)




    n_comp = 4
    facss = [[np.random.rand(di, 1) for di in d[1:]] for d in dims]