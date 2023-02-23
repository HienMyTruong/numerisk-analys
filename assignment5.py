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
    vec = np.zeros(X.shape)
    for i in range(X.shape[1]):
        # orthogonalization
        vector = X[:, i]
        space = vec[:, :i]
        projection = vector @ space
        vector = vector - np.sum(projection * space, axis=1)  # normalization
        norm = np.sqrt(vector @ vector)
        vector /= abs(norm) < 1e-8 and 1 or norm

        vec[:, i] = vector
    return vec



def exercise1():
    v1_ort = np.array([4 / 5, 3 / 5])
    v1 = np.array([4, 3])
    v2 = np.array([0, 1])

    print(np.dot(v1_ort, v2))
    v2_ort = (v2 - np.dot(v1_ort, v2) * v1_ort)
    v2_ort = v2_ort/np.linalg.norm(v2_ort)
    print("v2_ort = " + str(v2_ort) + "\n")

    vectors = np.array([[4, 0], [3, 1]], dtype=float)
    orthonormal = gram_schmidt(vectors)

    q_actual = np.concatenate((v1_ort, v2_ort), axis=0)
    print("expected Q = " + str(orthonormal))
    print("actual Q: " + "q1= "+ str(v1_ort) + "   "+ "q2= "+ str(v2_ort) + "\n")


    print("\n")
    r_12 = np.dot(v1_ort, v2)

    print("r_12 = " + str(r_12) + "\n")

    r_matrix = np.array([[5, 3 / 5], [0, 4 / 5]], dtype=float)

    print("R matrix = "+ str(np.matmul(orthonormal, r_matrix)) + "\n")

def exercise2():


    q_matrix = np.array([
        [-0.4743, 0.2927, -0.5741, -0.2224, 0.5377, -0.1455],
        [-0.3162, 0.0197, -0.4939, 0.0623, -0.8056, -0.0527],
        [-0.7906, -0.2138, 0.5404, -0.0801, -0.0390, -0.0857],
        [-0.1581, 0.1414, -0.0225, 0.9645, 0.1548, -0.0133],
        [0, -0.9209, -0.3273, 0.0972, 0.1869, -0.0190],
        [-0.1581, 0.0099, -0.0692, -0.0216, 0.0387, 0.9839]
    ],dtype=float)

    r_matrix = np.array([
        [6.3246, -6.7989],
        [0, 7.6010],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0]
    ],dtype=float)

    print(f"{q_matrix.shape=}, {r_matrix.shape=}")
    a_matrix = np.matmul(q_matrix, r_matrix)

    return a_matrix



exercise1()
print(exercise2())