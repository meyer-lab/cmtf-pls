import numpy as np
from scipy.linalg import inv, sqrtm
from tensorly.tenalg import multi_mode_dot,mode_dot
import tensorly as tl
import numpy.linalg as la
from numpy.linalg import inv
from scipy.linalg import sqrtm
import scipy
from tensorly.tucker_tensor import tucker_to_tensor

def ttt(x, y, ms):
    """
    Calculate the matrix product of two unfolded tensors along a specific mode

    Inputs:
    - x: Input tensor #1
    - y: Input tensor #2
    - ms: The specified mode 

    Return:
    - tl.tensor(mat): The matrix product of x and y along ms.

    Reference: 
    Zeng J, Wang W, Zhang X (2021). “TRES: An R Package for Tensor Regression and Envelope Algorithms.” 
    Journal of Statistical Software, 99(12), 1–31. doi:10.18637/jss.v099.i12.
    """
    s1 = x.shape
    s2 = y.shape
    idx_1 = [i for i in range(len(s1)) if i not in ms]
    idx_2 = [i for i in range(len(s2)) if i not in ms]
    mat_1 = tl.unfold(x, idx_1[0]) 
    mat_2 = tl.unfold(y, idx_2[0]) 
    mat = np.dot(mat_1, mat_2.T)
    return tl.tensor(mat)

def kroncov(Tn, tol=1e-6, maxiter=10): 
  """
  Calculate the MLE of the covariance matrix for tensor normal distribution, where the covariance has a separable 
  Kronecker structure. 

  Inputs:
  - Tn: The tensor
  - tol: Convergence tolerance for estimating the covariance matrix
  - maxiter: Maximum number of iterations

  IMPORTANT:
  - The last dimension of the input tensor should be the sample size dimension.

  Return a dictionary of:
  - lambda: Normalizing constant
  - S: a list containing each normalized covariance matrix. 

  Reference: 
  1. Zeng J, Wang W, Zhang X (2021). “TRES: An R Package for Tensor Regression and Envelope Algorithms.” 
  Journal of Statistical Software, 99(12), 1–31. doi:10.18637/jss.v099.i12.

  2. Zhang, X. and Li, L., 2017. Tensor envelope partial least-squares regression. 
  Technometrics, 59(4), pp.426-436.

  """
    ss = Tn.shape
    if len(ss) <= 1:
        raise ValueError("The dimension of Tn should be larger than one.")
    n = ss[-1]
    r = ss[:-1]
    m = len(r)
    prodr = np.prod(r)
    mu = np.mean(Tn, axis=len(ss)-1)
    Tn = Tn - mu
    lambda_ = 1
    S = [np.diag(np.ones(ri)) for ri in r]
    Sinvhalf = np.array(S)
    tol = [tol] * m
    if m > 1:
        flag = 0
        for iter_ in range(maxiter):
            for i in range(m):
                Si0 = S[i]
                idx = np.arange(m+1)
                idx=np.delete(idx,i)
                len_ = len(idx)
                Tsn = multi_mode_dot(Tn.values, Sinvhalf[idx[:len_-1]],modes=idx[:len_-1]) 
                idxprod = (r[i] / n) / prodr
                TsnTsn = ttt(Tsn, Tsn, ms=idx) * idxprod
                S[i] = TsnTsn / np.sqrt(np.sum(TsnTsn ** 2))
                Sinvhalf[i] = inv(sqrtm(S[i]))
                if np.sqrt(np.sum((Si0 - S[i]) ** 2)) < tol[i]:
                    flag = 1
                    break
            if flag == 1:
                break
        Tsn = multi_mode_dot(Tn.values, Sinvhalf,modes=np.arange(m)) 
        lambda_ = np.sum((Tsn ** 2)) / np.prod([*r, n])
    else:
        lambda_ = 1
        S[m - 1] = np.einsum(Tn, [list(range(len(r))), list(range(len(r)))]) * (1 / n)
    return {"lambda": lambda_, "S": S}

def simplsMU(M, U, u):
  """
  Generalization of the SIMPLS algorithm for estimating orthogonal basis of the envelope subspace.

  Inputs:
  - M: The positive definite matrix M of pxp as defined in the envelope objective function.
  - U: The positive semi-definite matrix U pxp as defined in the envelope objective function.
  - u: An integer between 0 and n representing the envelope dimension.

  Return:
  - Gamma: The estimated orthogonal basis of the envelope subspace.

  Reference: 
  1. Zeng J, Wang W, Zhang X (2021). “TRES: An R Package for Tensor Regression and Envelope Algorithms.” 
  Journal of Statistical Software, 99(12), 1–31. doi:10.18637/jss.v099.i12.

  2. Zhang, X. and Li, L., 2017. Tensor envelope partial least-squares regression. 
  Technometrics, 59(4), pp.426-436.

  """
  dimM = np.shape(M)
  dimU = np.shape(U)
  p = dimM[0]

  if dimM[0] != dimM[1] and dimU[0] != dimU[1]:
    raise Exception("M and U should be square matrices.")
  if dimM[0] != dimU[0]:
    raise Exception("M and U should have the same dimension.")
  if np.linalg.matrix_rank(M) < p:
    raise Exception("M should be positive definite.")
  if u > p and u < 0:
    raise Exception("u should be between 0 and p.")
  if u == p:
    return np.diag(p)
  else:
    W = np.zeros((p, u + 1))
    for k in range(u):
      Wk = W[:, :(k + 1)]
      Ek = np.matmul(M, Wk)
      temp = np.matmul(Ek.T, Ek)
      QEK = np.diag(np.ones(p)) - np.matmul(Ek, np.linalg.pinv(temp)@(Ek.T))
      
      eigen_vals, eigen_vecs = scipy.linalg.eig((np.matmul(QEK, np.matmul(U, QEK))).T,)
      W[:, k + 1] = eigen_vecs[:, np.argmax(eigen_vals)].real
    Gamma = np.linalg.qr(W[:, 1:u + 1])[0]
    return Gamma


def TPR_fit(x, y, u=None, method='PLS', Gamma_init=None):
  """
  Algorithm for Tensor Envelope PLS.

  Inputs:
  - x: The input tensor predictors.
  - y: The response tensor. (The size must be equal to the sample size (last) dimension of x)
  - u: The dimension of the tensor envelope subspace. Each integer in u corresponds to each (x's) mode's  
  dimension of the tensor envelope subspace.

  IMPORTANT:
  - The last dimension of the input tensor should be the sample size dimension.

  Return a dictionary containing:
  - fitted_values: The predicted responses from the fitted TEPLS model. 
  - coefficients: The estimation of regression coefficient tensor. 
  - Gamma: The estimation of envelope subspace basis. 
  - residuals: The residuals matrix.
  - Sigma: A lists of estimated covariance matrices at each mode for the tensor predictors.

  Reference: 
  1. Zeng J, Wang W, Zhang X (2021). “TRES: An R Package for Tensor Regression and Envelope Algorithms.” 
  Journal of Statistical Software, 99(12), 1–31. doi:10.18637/jss.v099.i12.

  2. Zhang, X. and Li, L., 2017. Tensor envelope partial least-squares regression. 
  Technometrics, 59(4), pp.426-436.

  """

  '''
    if y is None:
        tmp = x
        if isinstance(tmp, list):
            if tmp.__dict__.get('names') is not None:
                x = tmp['x']
                y = tmp['y']
            else:
                if len(x) < 2:
                    raise Exception("x or y is missing.")
                x = tmp[0]
                y = tmp[1]
        else:
            raise Exception("y is None, x should be a list.")
        if x is None or y is None:
            raise Exception("x or y is missing. Check names(x).")
    if not isinstance(y, np.matrix):
        if isinstance(y, np.ndarray):
            y = np.transpose(np.asmatrix(y))
        else:
            raise Exception("y should be numpy array or numpy matrix.")
    if not isinstance(x, np.ndarray):
        if isinstance(x, np.matrix) or isinstance(x, np.ndarray):
            x = np.asarray(x)
        else:
            raise Exception("x should be numpy matrix, numpy array or numpy ndarray.")
  '''
    x_old = x
    y_old = y
    ss = x.shape
    len_ = len(ss)
    n = ss[len_-1]
    '''if n != y.shape[1]:
        raise Exception("Unmatched dimension.")'''
    p = ss[0:len_-1]
    m = len(p)
    r = y.shape[0]

    ##center the data
    muy = np.mean(y)
    y = y - muy
    mux = np.mean(x, axis=len_-1)
    ttmp2 = x - mux
    ###

    x = ttmp2
    vecx = tl.base.unfold(x.values,mode=len_-1)
    res = kroncov(x)
    lambda_ = res['lambda']
    Sigx = res['S']
    Sigx[0] = lambda_*Sigx[0]

    if method == 'standard':
        Sigxinv = la.inv(Sigx)
        Bhat = np.dot(np.dot(Sigxinv, x), y) / n
        Gamma = None
    else:
        if u is None:
            raise ValueError('A user-defined u is required.')
        Sinvhalf = [la.inv(sqrtm(Sigx[i])) for i in range(m)]
        Sigy = (n-1)*np.cov(y.T)/n
        Sigy=np.array(Sigy)
        Sigy=np.reshape(Sigy,(1,1))
        Sinvhalf.append(la.inv(sqrtm(Sigy)))
        C = mode_dot(x.values, y, mode=m) / n
        C = np.expand_dims(C,axis=len(C.shape))
        Gamma = []
        PGamma = []
        for i in range(m):
            M = Sigx[i]
            idx = [j for j in range(m+1) if j != i]
            idx=np.array(idx)
            Ck = multi_mode_dot(C, np.array(Sinvhalf)[idx], modes=idx)
            U = tl.base.unfold(Ck,mode=i)
            idxprod = p[i]*n/r/np.prod(p)
            Uk = np.dot(U, U.T) * idxprod
            Gamma.append(simplsMU(M, Uk, u[i]))
            tmp = np.dot(Gamma[i].T, np.dot(Sigx[i], Gamma[i]))
            PGamma.append(np.dot(Gamma[i],(np.dot(la.inv(tmp),np.dot(Gamma[i].T,Sigx[i])))))
        PGamma.append(y)
        Bhat = multi_mode_dot(x.values,PGamma,modes=np.arange(m+1))/n
        Bhat = np.expand_dims(Bhat,axis=len(Bhat.shape))
    tp1 = tl.base.unfold(Bhat,mode=len_-1)
    tp2 = tl.base.unfold(x_old.values,mode=len_-1)
    fitted_values = np.dot(tp2,tp1.T)
    residuals= y_old - fitted_values
    output={}
    output["fitted_values"]=fitted_values
    output["coefficients"]=Bhat
    output["Gamma"]=Gamma
    output["residuals"]=residuals
    output["Sigma"]=Sigx
    output["PGamma"]=PGamma

    return output

def TPR_reconstruct_X(output,x, y):
    """
      Reconstruct x based on the tucker factors from Tensor Envelope PLS.

      Inputs:
      - output: The output from TPR_fit.
      - x: The input tensor predictors.
      - y: The response tensor. (The size must be equal to the sample size (last) dimension of x)

      IMPORTANT:
      - The last dimension of the input tensor should be the sample size dimension.

      Return:
      - x_reconstructed: The reconstructed x with the same dimension of x by converting the Tucker tensor from TEPLS
      (output["PGamma"]) back into the full  tensor.

      Reference: 
      1. Zeng J, Wang W, Zhang X (2021). “TRES: An R Package for Tensor Regression and Envelope Algorithms.” 
      Journal of Statistical Software, 99(12), 1–31. doi:10.18637/jss.v099.i12.

      2. Zhang, X. and Li, L., 2017. Tensor envelope partial least-squares regression. 
      Technometrics, 59(4), pp.426-436.

    """
    factors=[]

    for i in range(len(output["PGamma"])):
        factors.append(output["PGamma"][i].T)

    x_reconstructed = tucker_to_tensor((x.T.values,factors))

    return x_reconstructed

def TPR_transform_X(output,x, y):
    """
    Reduce the dimension x based on the tucker factors from Tensor Envelope PLS.

    Inputs:
    - output: The output from TPR_fit.
    - x: The input tensor predictors.
    - y: The response tensor. (The size must be equal to the sample size (last) dimension of x)

    IMPORTANT:
    - The last dimension of the input tensor should be the sample size dimension.

    Return:
    - x_reduced: The reduced x with the dimension specified by the envelope subspace by applying the Tucker tensor from TEPLS
    (output["Gamma"]).

    Reference: 
    1. Zeng J, Wang W, Zhang X (2021). “TRES: An R Package for Tensor Regression and Envelope Algorithms.” 
    Journal of Statistical Software, 99(12), 1–31. doi:10.18637/jss.v099.i12.

    2. Zhang, X. and Li, L., 2017. Tensor envelope partial least-squares regression. 
    Technometrics, 59(4), pp.426-436.
    """
    factors=[]
    for i in range(len(output["Gamma"])):
        factors.append(output["Gamma"][i].T)

    x_reduced = tucker_to_tensor((x.T.values,factors))

    return x_reduced

