import numpy as np
import pandas as pd
import seaborn as sns
from .tpls import *
from numpy.linalg import pinv


def genRandXY(n=10, k=6, m=4):
    X = np.random.rand(n, k)
    Y = np.random.rand(n, m)
    X -= np.mean(X, axis=0)
    Y -= np.mean(Y, axis=0)
    return X, Y


def R2Xplots(X, Y, method = wold_nipals, n = 6):
    R2Xs, R2Ys, Q2Ys = [], [], []
    RQ2Ys = []
    looXs = [X[np.arange(X.shape[0]) != i,:] for i in range(X.shape[0])]
    looYs = [Y[np.arange(Y.shape[0]) != i,:] for i in range(Y.shape[0])]
    for i in range(1, n+1):
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
    ax = sns.lineplot(data=pd.melt(df, ["Component"], ["R2X", "R2Y", "Q2Y", "RQ2Y"]), x="Component", y="value", hue="variable")
    ax.set(ylabel='Variance Explained', title="PLSR X and Y Explained")
    return ax
