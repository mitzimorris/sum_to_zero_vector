### BYM2 model helper functions - compute scaling factor tau

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from libpysal.weights import Queen

def q_inv_dense(Q, A=None):
    # Compute the inverse of the sparse precision matrix Q
    Sigma = splinalg.inv(Q).todense()
    if A is None:
        return Sigma
    else:
        A = np.ones((1, Sigma.shape[0]))
        W = Sigma @ A.T
        Sigma_const = Sigma - W @ np.linalg.inv(A @ W) @ W.T
        return Sigma_const

def get_scaling_factor(nbs: Queen) -> np.float64:
    """
    Compute the geometric mean of the spatial covariance matrix
    """
    adj_matrix = nbs.full()[0]
    adj_matrix = sp.csr_matrix(adj_matrix)  # Convert to sparse matrix

    # Create ICAR precision matrix (diagonal minus adjacency)
    Q = sp.diags(np.ravel(adj_matrix.sum(axis=1))) - adj_matrix
    # Add jitter to the diagonal for numerical stability
    Q_pert = Q + sp.eye(nbs.n) * np.max(Q.diagonal()) * np.sqrt(np.finfo(float).eps)
    # Compute the diagonal elements of the covariance matrix
    Q_inv = q_inv_dense(Q_pert, adj_matrix)

    # Compute the geometric mean of the variances (diagonal elements of Q_inv)
    return np.exp(np.mean(np.log(np.diag(Q_inv))))

