from tensorly.regression import cp_plsr
from tensorly.cp_tensor import CPTensor
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_auc_score
from importlib.abc import PathEntryFinder
import os
from os.path import dirname, join
from pathlib import Path
import ast
import textwrap
from types import CellType
import pandas as pd
import numpy as np
import warnings
import xarray as xa
from copy import copy
import tensorly
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
from tensorly.regression import CP_PLSR
from scipy.stats import pearsonr
from tensorly.decomposition import parafac
from tensorpack.cmtf import perform_CP
from tensorly import norm
from tensorpack.cmtf import cp_normalize
import tensorpack 

def PLS_IA_Imputation(X,Y,tol,numComp,max_iter):
  """
  Iteratively doing PLS between X and Y to impute missing data in both X and Y. 

  Inputs:
  - X: Input tensor with missing data
  - Y: Response Tensor with missing data
  - tol: Convergence tolerance
  - numComp: Number of Components for doing tPLS decomposition
  - max_iter: Maximum number of iterations

  Return:
  - X_intermediate[-1]: X with imputed data
  - Y_intermediate[-1]: Y with imputed data

  """
    if tol==None:
        tol=1e-6

    if max_iter==None:
        max_iter=100

    NOT_missing=np.isfinite(X)
    Xmissing=np.isnan(X).astype("int")


    Xmean = np.nanmean(X, axis=0)
    Xmeant = np.repeat(Xmean[np.newaxis, :, :], X.shape[0], axis=0)
    Ximpute = Xmeant * Xmissing + np.nan_to_num(X) * (1 - Xmissing)

    Ymissing=np.isnan(Y)
    Ynotmissing=np.isfinite(Y)
    Yimpute=Y.copy()
    Yimpute[Ymissing]=np.nanmean(Yimpute)
    
    X_intermediate =[]
    Y_intermediate=[]

    X_intermediate.append(Ximpute)
    Y_intermediate.append(Yimpute)

    diff=1
    n=-1

    while ((diff>tol) and (n<=max_iter)):
        


        model = CP_PLSR(numComp)  #Need to find an optimal components number
        model.fit(X_intermediate[n], Y_intermediate[n])
        Xnew = CPTensor((None, model.X_factors)).to_tensor()
        X = Xnew * Xmissing + np.nan_to_num(X) * (1 - Xmissing)
        Y=model.predict(X.values)

        X_intermediate.append(X)
        Y_intermediate.append(Y)

        if n>=1:
            diff = np.linalg.norm(X_intermediate[n] - X_intermediate[n-1])
        
        n+=1

        print(diff)
    
    return X_intermediate[-1],Y_intermediate[-1]

def CP_IA_Imputation(X,Y,tol,numComp,max_iter):
  """
  Iteratively doing CP between X and Y to first impute missing data in X. After CP iteration,apply tPLS between 
  recovered X and Y to impute missing data in Y.

  Inputs:
  - X: Input tensor with missing data
  - Y: Response Tensor with missing data
  - tol: Convergence tolerance
  - numComp: Number of Components for doing tPLS decomposition
  - max_iter: Maximum number of iterations

  Return:
  - X_intermediate[-1]: X with imputed data
  - Yimpute: Y with imputed data

  """

    if tol==None:
        tol=1e-6
    
    if max_iter==None:
        max_iter=100

    Xmissing=np.isfinite(X).astype("int")


    Ymissing=np.isnan(Y)
    Ynotmissing=np.isfinite(Y)
    Yimpute=Y.copy()
    
    X_intermediate =[]

    X_intermediate.append(X.values)


    diff=1
    n=0

    while ((diff>tol) and (n<=max_iter)):
        Xfac= perform_CP(X_intermediate[n],numComp,tol=1e-7, maxiter=1000)  #Need to find an optimal components number
        Xnew = CPTensor((Xfac[0], Xfac[1])).to_tensor()

        X_intermediate.append(Xnew)

        if n>=2:
            diff = norm(X_intermediate[n]-X_intermediate[n-1])
        

        n+=1



        print(diff)
    
    model = CP_PLSR(numComp)  #Need to find an optimal components number
    model.fit(X_intermediate[-1][Ynotmissing], Yimpute[Ynotmissing])
    Yimpute[Ymissing]=(model.predict(X_intermediate[-1][Ymissing])).reshape(1,len(model.predict(X_intermediate[-1][Ymissing])))[0]
    Yimpute[Ymissing]=np.exp(Yimpute[Ymissing])/(1+np.exp(Yimpute[Ymissing]))
    Yimpute[Yimpute>=0.5]=1
    Yimpute[Yimpute<0.5]=0
    
    return X_intermediate[-1],Yimpute

def CP_PLS_IA_Imputation(X,Y,tol,numComp,max_iter):
  """
  Iteratively doing CP and tPLS between X and Y to impute missing data in X and Y. 

  Inputs:
  - X: Input tensor with missing data
  - Y: Response Tensor with missing data
  - tol: Convergence tolerance
  - numComp: Number of Components for doing tPLS decomposition
  - max_iter: Maximum number of iterations

  Return:
  - X_intermediate[-1]: X with imputed data
  - Yimpute: Y with imputed data

  """

    if tol==None:
        tol=1e-6
    
    if max_iter==None:
        max_iter=1000

    Ymissing=np.isnan(Y)
    Ynotmissing=np.isfinite(Y)
    Yimpute=Y.copy()
    Y_intermediate=[]
    Yimpute[Ymissing]=np.nanmean(Yimpute)
    Y_intermediate.append(Yimpute)

    Xmissing=np.isfinite(X).astype("int")
    X_intermediate=[]
    X_intermediate.append(X.values)
    X_output=[]


    diff=1
    n=0

    while ((diff>tol) and (n<=max_iter)):
        Xfac= perform_CP(X_intermediate[n],numComp,tol=1e-7, maxiter=1000)  #Need to find an optimal components number
        X_temp=CPTensor((Xfac[0],Xfac[1])).to_tensor()
        model = CP_PLSR(numComp)  #Need to find an optimal components number
        model.fit(X_temp[Ynotmissing], Y_intermediate[n][Ynotmissing])

        Yimpute[Ymissing]=(model.predict(X_temp[Ymissing]).reshape(1,len(model.predict(X_temp[Ymissing]))))[0]
 
        X_intermediate.append(X_temp)
        Y_intermediate.append(Yimpute)

        if n>=2:
            diff = norm(X_intermediate[n]-X_intermediate[n-1])
        

        n+=1



        print(diff)
    Y_intermediate[-1][Ymissing]=np.exp(Y_intermediate[-1][Ymissing])/(1+np.exp(Y_intermediate[-1][Ymissing]))
    Yimpute = Y_intermediate[-1]
    Yimpute[Yimpute>=0.5]=1
    Yimpute[Yimpute<0.5]=0
    return X_intermediate[-1],Yimpute

    