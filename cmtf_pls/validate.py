import pandas as pd
import seaborn as sns
from sklearn.model_selection import LeaveOneOut

from cmtf_pls.tpls import *
from cmtf_pls.pls3 import ThreeModePLS


def plotTensorPLS(pls, X, Y, n = 6):
    R2Xs, R2Ys, Q2Ys = [], [], []
    looXs = [X[np.arange(X.shape[0]) != i,...] for i in range(X.shape[0])]
    looYs = [Y[np.arange(Y.shape[0]) != i,...] for i in range(Y.shape[0])]
    for rr in range(1, n + 1):
        plso = pls(X, Y, rr)
        plso.fit()
        R2Xs.append(calcR2X(X, plso.X_reconstructed()))
        R2Ys.append(calcR2X(Y, plso.Y_reconstructed()))
        predY = np.zeros_like(Y)
        try:
            for j in range(X.shape[0]):
                pls_loo = pls(looXs[j], looYs[j], rr)
                pls_loo.fit()
                predY[j, :] = pls_loo.predict(X[[j], :])
            Q2Ys.append(calcR2X(Y, predY))
        except NotImplementedError:
            pass
    try:
        pls_loo.predict(X)
    except NotImplementedError:
        print("Prediction has not been implemented yet")
        df = pd.DataFrame({"Component": range(1, n + 1), "R2X": R2Xs, "R2Y": R2Ys})
    else:
        df = pd.DataFrame({"Component": range(1, n + 1), "R2X": R2Xs, "R2Y": R2Ys, "Q2Y": Q2Ys})

    ax = sns.lineplot(data=pd.melt(df, ["Component"]), x="Component", y="value", hue="variable")
    ax.set(ylabel='Variance Explained', title="PLSR X and Y Explained")
    return ax


def get_q2y(pls_tensor):
    """
    Calculates Q2Y for a fitted PLS tensor.

    Args:
        pls_tensor (PLSTensor): fitted PLS tensor.

    Returns:
        Q2Y (float): Q2Y of PLS tensor applied to fitted dataset.
    """
    assert pls_tensor.original_X is not None, \
        'PLS Tensor must be fit prior to calculating Q2Y'
    X = pls_tensor.original_X
    Y = pls_tensor.original_Y
    q2y_plsr = ThreeModePLS(pls_tensor.n_components)

    loo = LeaveOneOut()
    Y_pred = np.zeros(Y.shape)
    Y_actual = np.zeros(Y.shape)
    for train_index, test_index in loo.split(X, Y):
        X_train, Y_train = X[train_index], Y[train_index]
        X_test, Y_test = X[test_index], Y[test_index]
        q2y_plsr.fit(X_train, Y_train)

        Y_pred[test_index] = q2y_plsr.predict(X_test)
        Y_actual[test_index] = Y_test

    numerator = (Y_pred - Y_actual) ** 2
    denominator = (Y_actual) ** 2
    return 1 - numerator.sum() / denominator.sum()
