import numpy as np
import pandas as pd
import seaborn as sns
from .tpls import *
from numpy.linalg import pinv

def plotTensorPLS(X, Y, method = wold_nipals, n = 6):
    R2Xs, R2Ys, Q2Ys = [], [], []
    RQ2Ys = []
    looXs = [X[np.arange(X.shape[0]) != i,:] for i in range(X.shape[0])]
    looYs = [Y.reshape(Y.shape[0], -1)[np.arange(Y.shape[0]) != i,:] for i in range(Y.shape[0])]
    for i in range(1, n+1):
        if method == tensorPLS:
            pass
        else:
            T, U, W, P, C, Q = method(X, Y, num_comp = i)
            R2Xs.append(calcR2X(X, T @ P.T))
            R2Ys.append(calcR2X(Y, T @ C.T))
            B = W @ pinv(P.T @ W) @ T.T @ Y
            RQ2Ys.append(calcR2X(Y, X @ B))
            predY = np.zeros_like(Y)
            for j in range(X.shape[0]):
                T, U, W, P, C, Q = method(looXs[j], looYs[j], num_comp=i)
                B = W @ pinv(P.T @ W) @ T.T @ looYs[j]
                predY[j, :] = X[i, :] @ B
            Q2Ys.append(calcR2X(Y, predY))

    df = pd.DataFrame({"Component": range(1, n+1), "R2X": R2Xs, "R2Y": R2Ys, "Q2Y": Q2Ys, "RQ2Y": RQ2Ys})
    ax = sns.lineplot(data=pd.melt(df, ["Component"]), x="Component", y="value", hue="variable")
    ax.set(ylabel='Variance Explained', title="PLSR X and Y Explained")
    return ax
