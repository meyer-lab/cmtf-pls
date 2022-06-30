import numpy as np
import pandas as pd
import seaborn as sns
from .tpls import *


def genRandXY(n=10, k=6, m=4):
    X = np.random.rand(n, k)
    Y = np.random.rand(n, m)
    X -= np.mean(X, axis=0)
    Y -= np.mean(Y, axis=0)
    return X, Y


def R2Xplots(X, Y, method = wold_nipals, n = 6):
    R2Xs, R2Ys, Q2Ys = [], [], []
    for i in range(1, n+1):
        T, U, W, P, C, Q = method(X, Y, num_comp = i)
        R2Xs.append(calcR2X(X, T @ P.T))
        R2Ys.append(calcR2X(Y, U @ Q.T))
        Q2Ys.append(calcR2X(Y, T @ C.T))
    df = pd.DataFrame({"Component": range(1, n+1), "R2X": R2Xs, "R2Y": R2Ys, "Q2Y": Q2Ys})

    ax = sns.lineplot(data=pd.melt(df, ["Component"], ["R2X", "R2Y", "Q2Y"]), x="Component", y="value", style="variable")
    ax.set(ylabel='Variance Explained', title="PLSR X and Y Explained")
    return ax
