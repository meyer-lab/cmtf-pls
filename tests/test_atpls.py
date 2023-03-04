from cmtf_pls.atpls import *

def test_tPLS():
    np.random.seed(25)
    X = np.random.rand(20, 8, 6, 4)
    Y = np.random.rand(20, 1)
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