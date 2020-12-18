import numpy as np
import numerics2_manalu as np2

def qr(matrix):
  m,n = matrix.shape
  R = matrix.copy()
  Q = np.identity(m)

  for j in range(0, n):
      v, t = householder(R[j:, j])
      H = np.identity(m)
      H[j:, j:] -= t * v.reshape(-1, 1) @ v
      R = H @ R
      Q = H @ Q

  return Q[:n].T, R[:n]

def vector_norm(n , vec):
    result = 0
    for k in range (n ):
       result = result + vec[k] * vec[k]
    return math.sqrt(result)

def householder(vec):
    vec = np.asarray(vec, dtype=float)
    if vec.ndim != 1:
        raise ValueError("vec.ndim = %s, expected 1" % vec.ndim)

    n = len(vec)
    I = np.eye(n)
    e1 = np.zeros_like(vec).astype(float)
    e1[0] = 1.0
    V1 = e1 * np.linalg.norm(vec)
    print("V1:", V1)
    u = vec
    u[0] = -(np.sum(np.square(u[1:]))) / (vec[0] + np.linalg.norm(vec))
    u = u / np.linalg.norm(u)
    H = I - 2 * (np.outer(u, u))
    return V1, H

def leastSquares_lu(matrix, vector):
  matrixT = np.transpose(matrix)
  matrixTmatrix = np.dot(matrixT,matrix)
  matrixTvector = np.dot(matrixT, vector)
  x = np2.lusolve(matrixTmatrix, matrixTvector)
  r = vector - np.dot(matrix,x)
  return x, r 

def qrsolve(matrix, vector):
  (m,n) = matrix.shape
  if m == n:
    x = np2.lusolve(matrix,vector)
    r = 0
  else:
    x,r = leastSquares_lu(matrix, vector)

  resid = 0
  for i in range(len(r)):
    resid += r[i]
  return matrix, resid
