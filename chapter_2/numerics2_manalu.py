import numpy as np
import math

def rowswap(matrix, row_1, row_2):
  matrix[[row_1,row_2],:] = matrix[[row_2,row_1],:] 
  return matrix

def rowdiff(matrix, row_1, row_2, scale=1.0):
  matrix[row_1,:] = matrix[row_1,:]-np.float(scale)*matrix[row_2,:]
  return matrix

def rowscale(matrix, row_1, scale=1.0):
  matrix[row_1,:] = scale * matrix[row_1,:]
  return matrix

def noswapLU(matrix):
  A = np.asarray(matrix)
  (m,n) = A.shape

  if (m != n):
    raise ValueError("The input matrix is not square. Your matrix has size %d x %d", (m,n))

  L = np.zeros((m,n))
  for col in range(n):
    L[col,col] = 1.0
    pivot = np.float(A[col,col])
    for row in range(col+1, n):
      scalefactor = A[row,col]/pivot
      L[row,col] = scalefactor
      A = rowdiff(A, row, col, scalefactor)

  return L, A

def backsub(U,b):
  U = np.asarray(U)
  b = np.asarray(b)

  n = b.size
  (mU, nU) = U.shape

  if (mU != nU) or (n != nU):
    raise ValueError("The dimensions are not correct. Matrix U must be square and the length of b must be the number of columns of U")

  x = np.zeros(b.shape)

  x[n-1] = b[n-1]/np.float(U[n-1,n-1])

  for row in range(n-2, -1, -1):
    x[row] = (b[row]-np.dot(U[row, row+1:n], x[row+1:n]))/U[row,row]

  return x

def forwardsub(L,b):
  L = np.asarray(L)
  b = np.asarray(b)

  n = b.size
  (mL, nL) = L.shape

  if (mL != nL) or (n != nL):
    raise ValueError("The dimensions are not correct. Matrix L must be square and the length of b must be the number of columns in L")
  
  x = np.zeros(b.shape)
  x[0] = b[0]/np.float(L[0,0])

  for row in range(1,n):
    x[row] = (b[row] - np.dot(L[row,0:row], x[0:row]))/L[row,row]

  return x

def fbsolve(L,U,b):
  L = np.asarray(L)
  U = np.asarray(U)
  b = np.asarray(b)
  n = b.size
  (mU, nU) = U.shape
  (mL, nL) = L.shape

  if (mL != nL) or (n != nL) or (mU != nU) or (nU != n):
    raise ValueError("The dimensions are not correct. U and L must be square matrices. b must be equal to the number of columns of U and L")

  y = forwardsub(L,b)
  x = backsub(U,y)

  return x

def lusolve(A,b):
  L,U = noswapLU(A)
  x = fbsolve(L,U,b)
  return x

def lu(matrix):
  A = np.asarray(matrix)
  (m,n) = A.shape

  if (m != n):
    raise ValueError("The input matrix must be square.")
  
  L = np.zeros(A.shape)
  P = np.identity(n)

  for col in range(n):
    pivot_index = np.argmax(np.absolute(A[col:n, col]))
    pivot_index += col
    pivot = np.float(A[pivot_index,col])

    if (pivot):
      if (pivot_index != col):
        A = rowswap(A, pivot_index, col)
        L = rowswap(L, pivot_index,col)
        P = rowswap(P, pivot_index, col)
    else:
      raise ValueError("The input matrix is singular, so the decomposition fails.")

    for row in range(col+1, n):
      scalefactor = A[row,col]/pivot
      L[row,col] = scalefactor
      A = rowdiff(A, row, col, scalefactor)

  L += np.identity(n)
  return L, A, P

def jacobi(matrix, vector, initial_answer, tolerance, maxIter = 1000):
  D = np.diag(matrix)
  R = matrix - np.diagflat(D)
  current_answer = np.asarray(initial_answer)
  counter = 0
  error = np.max(current_answer)
  
  while (counter <= maxIter) and (error >= tolerance):
    current_answer = (vector - np.dot(R,current_answer)) / D
    counter += 1
    error = abs(np.max(current_answer) - error)

  return current_answer
  
def gausssiedel(matrix, vector, initial_answer, tolerance, maxIter = 1000):
  n = len(matrix)
  current_answer = np.asarray(initial_answer)
  counter = 0
  error = np.max(current_answer)
  
  while (counter <= maxIter) and (error >= tolerance):
    for j in range(0, n):
        d = vector[j]                   
        for i in range(0, n):      
            if(j != i): 
                d-=matrix[j][i] * current_answer[i]    
        current_answer[j] = d / matrix[j][j]
        error = np.max(current_answer)
        counter+=1 
  return current_answer
