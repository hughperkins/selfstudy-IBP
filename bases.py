"""
Methods to facilitate working with bases, projections etc

Note: it looks like this is not needed at all:
- QR decomposition gives us the basis, as Q
- matrix multiplication with Q.T gives us the projection into that basis
"""
import numpy as np


print('loading bases')


def get_orthonormal_basis(X):
    """
    we are going to calculate an orthonormal basis
    for the columnspace of X
    """

    # first basis vector is simply normalized form of
    # first column vector of X
    # note: we are going to assume X columns are
    # linearly independent for now
    # B is basis

    # check X is 2-dimensional matrix
    assert len(X.shape) == 2

    K = X.shape[1]
    D = X.shape[0]
    B = np.zeros((D, K), dtype=np.float32)
    for k in range(K):
        b_unnorm = np.copy(X[:, k])
        for k2 in range(k):
            b_unnorm -= X[:, k].dot(B[:, k2]) * B[:, k2]
        b_norm = b_unnorm / np.linalg.norm(b_unnorm)
        B[:, k] = b_norm
    return B


def proj_orthonorm(v, B):
    """
    project vector v into orthonormal basis B
    """
    # K = B.shape[1]
    # proj = np.zeros((K,), dtype=np.float32)
    # for k in range(K):
    #     print('proj.shape', proj.shape)
    #     print('v.shape', v.shape)
    #     print('B[:, k].shape', B[:, k].shape)
    #     proj[k] = v.dot(B[:, k])
    # print('proj', proj)
    # print('B.dot(v)', B.dot(v))
    # print('v.dot(B)', v.dot(B))
    # return proj
    return B.T.dot(v)
