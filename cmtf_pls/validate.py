from .tpls import *
import pandas as pd
import seaborn as sns

def plotTensorPLS(pls, X, Y, n = 6):
    R2Xs, R2Ys, Q2Ys = [], [], []
    looXs = [X[np.arange(X.shape[0]) != i,...] for i in range(X.shape[0])]
    looYs = [Y[np.arange(Y.shape[0]) != i,...] for i in range(Y.shape[0])]
    for rr in range(1, n + 1):
        plso = pls(X, Y, rr)
        plso.fit()
        R2Xs.append(calcR2X(X, plso.x_recover()))
        R2Ys.append(calcR2X(Y, plso.y_recover()))
        predY = np.zeros_like(Y)
        try:
            for j in range(X.shape[0]):
                pls_loo = pls(looXs[j], looYs[j], rr)
                pls_loo.fit()
                predY[j, :] = pls_loo.predict(X[j, :])
            Q2Ys.append(calcR2X(Y, predY))
        except NotImplementedError:
            pass
    try:
        pls_loo.predict(X[0, :])
    except NotImplementedError:
        print("Prediction has not been implemented yet")
        df = pd.DataFrame({"Component": range(1, n + 1), "R2X": R2Xs, "R2Y": R2Ys})
    else:
        df = pd.DataFrame({"Component": range(1, n + 1), "R2X": R2Xs, "R2Y": R2Ys, "Q2Y": Q2Ys})

    ax = sns.lineplot(data=pd.melt(df, ["Component"]), x="Component", y="value", hue="variable")
    ax.set(ylabel='Variance Explained', title="PLSR X and Y Explained")
    return ax
