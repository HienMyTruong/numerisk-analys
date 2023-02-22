import numpy as np

def modifiedGramSchmidt(A):
    """
    Gives a orthonormal matrix, using modified Gram Schmidt Procedure
    :param A: a matrix of column vectors
    :return: a matrix of orthonormal column vectors
    """
    # assuming A is a square matrix
    dim = A.shape[0]
    Q = np.zeros(A.shape, dtype=A.dtype)
    for j in range(0, dim):
        q = A[:,j]
        for i in range(0, j):
            rij = np.vdot(q, Q[:,i])
            q = q - rij*Q[:,i]
        rjj = np.linalg.norm(q, ord=2)
        if np.isclose(rjj,0.0):
            raise ValueError("invalid input matrix")
        else:
            Q[:,j] = q/rjj
    return Q


def gram_schmidt(X):
    O = np.zeros(X.shape)
    for i in range(X.shape[1]):
        # orthogonalization
        vector = X[:, i]
        space = O[:, :i]
        projection = vector @ space
        vector = vector - np.sum(projection * space, axis=1)  # normalization
        norm = np.sqrt(vector @ vector)
        vector /= abs(norm) < 1e-8 and 1 or norm

        O[:, i] = vector
    return O

"""
vectors = np.array([
    [1, 1, 2, 0, 1, 1],
    [0, 0, 0, 1, 2, 1],
    [1, 2, 3, 1, 3, 2],
    [1, 0, 1, 0, 1, 1]
], dtype=float)
"""


vectors = np.array([[4, 0], [3, 1]], dtype=float)
orthonormal = gram_schmidt(vectors)

print(orthonormal)

