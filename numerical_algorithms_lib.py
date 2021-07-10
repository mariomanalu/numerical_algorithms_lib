########################################
# This file contains the implementation of various numerical algorithms.
# Each algorithm has its own purpose. 
# 1. They can be used to solve a simple linear equation whose exact solution is hard, if not impossible, to compute.
#    For instance, the square root of two. 
#    That said, we can use Fixed Point Iteration, Bisection Method, Newton Method to compute the approximation of the square root of two.
# 2. They can be used to solve a multivariable linear equations. 
#    Multivariable linear equations can be written in matrix forms. 
#    The solutions boils down to  manipulating the matrices in such a way that the matrix is in reduced, echelon form.
#    Once the matrix is in reduced, echelon form, we can find a solution to one of the variables and substitute back to find the solutions to the rest of the variables
# 3. They can be used to approximate the result of some definite integrals.
#    Integrating a function can be computationally very expensive.
#    Plus, we are normally happy with approximating integrals to some decimal digits -- we rarely need the exact solution.
#    Thus, we can use Simpson's Method or Gaussian Quadrature to approximate some definite integrals at a lower cost.

# Credits:
# I learned about the algorithms in this file from reading Numerical Analysis by T.Sauer.
# I did not develop any of these algorithms. I am only implementing what has been conjectured and proven to be true.

# Disclaimer:
# Please read and study the code before using it. Some of these algorithms have hard-coded degree of error to stop while loops. 
# That is, once the solution is correct up to say the 6th decimal digit, the algorithm may stop. 
# The stopping criterion may need to be adjusted based on your needs.
##########################################  
import numpy as np
import math

# A function that computes the result of nested polynomials
def polynest(val, coeffs, bspoints = []):

    #val is the point where the function is evaluated
    #coeffs is a vector containing coefficients of the polynomial
    #bspoints is a vector containing basepoints. Default is empty.
    #this function evaluates the polynomial at val and returns current_val

    #5x^3 + 3x^2 + x + 4 in nested polynomial is 4 + x(1+x(5x+3))
    current_val = coeffs[len(coeffs) - 1]


    coeffs = np.asarray(coeffs)
    bspoints = np.asarray(bspoints)

    if coeffs.ndim != 1:
        raise ValueError("Give a vector!")

    if bspoints.ndim != 1:
        raise ValueError("Give a vector!")

    if bspoints != []:
        for index in range(len(coeffs)-2, -1, -1):
            current_val = current_val * (val - bspoints[index]) + coeffs[index]
    else:
        for index in range(len(coeffs)-2, -1, -1):
            current_val = current_val * val + coeffs[index]

    return current_val

#A function that bisects interval to find the root/roots of a function 
def bisection(fxn, ainput, binput, tol = .000001, maxIter = 250):
    #fxn is a function
    #ainput is the lower bound
    #binput is the upper bound
    #tol is tolerance. That is, any midpoint in a range of size equal or smaller than the tolerance is considered a root
    #maxIter is maximum iteration. That is, the midpoint just before the maxIter-th iteration is considered a root


    #Compute the sign of f(ainput)
    fa = np.sign(fxn(ainput))
    #Compute the sign of f(binput)
    fb = np.sign(fxn(binput))
    #Initialize iteration
    iteration = 0
    #Initialize roots
    roots = np.zeros((maxIter,1))
    #Compute initial range
    range = binput - ainput

    #If f(ainput) and f(binput) have the same signs, then there is no root in between ainput and binput
    if fa * fb > 0:
        raise ValueError("There is no root in this interval")
    else:
        #If fxn(ainput) == 0, then ainput is a root
        if (fxn(ainput) == 0):
            return 0, roots
        #If fxn(binput) == 0, then binput is a root
        elif (fxn(binput) == 0):
            return 0, roots

    #While the range is still greater than the tolerance and iteration is less than maxIter, keep bisecting the interval
    while (range > tol) and (iteration < maxIter):
        #Current range is 0.5 * previous range 
        range = 0.5 * range
        #Compute c as the new midpoint
        c = ainput + range
        #Store c in the array
        roots[iteration] = c
        #Compute the sign function of fxn(c)
        fc = np.sign(fxn(c))

        #If the signs of fxn(a) and fxn(c) are both positive or both negative, then the root is on the right half
        if fa * fc > 0:
            ainput = c
        #If the signs of fxn(a) and fxn(c) are different, then the root is on the left half
        else:
            binput = c
        
        #Increment iteration
        iteration += 1


    return c, roots

#A function that finds a root of a function using the fixed point iteration method
def fixedpt(fxn, xinit, tol = .000001, maxIter = 250):
    #fxn is a function
    #xinit is the initial approximation
    #tol is tolerance. That is, any midpoint in a range of size equal or smaller than the tolerance is considered a root
    #maxIter is maximum iteration. That is, the midpoint just before the maxIter-th iteration is considered a root

    #Initialize iter
    iter = 0
    #Initialize the first root
    r_prev = xinit
    #Compute initial error
    err = abs(r_prev)
    #Initialize roots
    roots = np.zeros((maxIter,1))

    #While the range is still greater than the tolerance and iteration is less than maxIter, keep computing new roots
    while (err > tol) and (iter < maxIter):
        #Computes current root
        r = fxn(r_prev)
        #Store current root
        roots[iter] = r
        #Computes error
        err = abs(r - r_prev)
        #Set the current root to be the previous root for the next interation
        r_prev = r
        #Increment iter
        iter += 1

    return r, roots

#A function that finds a root of a function using the Newton method
def newton(fxn, dfxn, xinit, tol = .000001, maxIter = 250):
    #fxn is a function
    #dfxn is the first 
    #tol is tolerance. That is, any r in a range of size equal or smaller than the tolerance is considered a root
    #maxIter is maximum iteration. That is, the midpoint just before the maxIter-th iteration is considered a root   

    #Initialize iter
    iter = 0
    #Initialize r_prev
    r_prev = xinit
    #Compute initial error
    err = abs(r_prev)
    #Initialize roots
    roots = np.zeros((maxIter,1))

    #While the error is still greater than the tolerance and iteration is less than maxIter, keep computing new roots
    while (err > tol) and (iter < maxIter):
        #Computes current root
        r = r_prev - (fxn(r_prev) / dfxn(r_prev))
        #Store the root
        roots[iter] = r
        #Compute new error
        err = abs(r - r_prev)
        #Set the current root to be the previous root for the next iteration
        r_prev = r
        iter += 1

    return r, roots

# A function that swaps two rows in a matrix
def rowswap(matrix, row_1, row_2):
  matrix[[row_1,row_2],:] = matrix[[row_2,row_1],:] 
  return matrix

# A function that computes the difference between two rows in a matrix
def rowdiff(matrix, row_1, row_2, scale=1.0):
  matrix[row_1,:] = matrix[row_1,:]-np.float(scale)*matrix[row_2,:]
  return matrix

# A function that scales a row in a matrix
def rowscale(matrix, row_1, scale=1.0):
  matrix[row_1,:] = scale * matrix[row_1,:]
  return matrix

# A helper function used to solve matrix equations
def noswapLU(matrix):
  A = np.asarray(matrix)
  (m,n) = A.shape

  # Check to see if the matrix is square
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

# A helper function used to perform back substitutions
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

# A helper function used to perform forward substitutions
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

# A helper function used to call both forward and back substitutions
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

# A function that solves matrix equations by decomposing the matrix into lower and upper triangle and using backward and forward substitutions
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

def newtondd(data_1, data_2):
  n = np.shape(data_2)[0]
  result = np.zeros([n, n])
  result[::,0] = data_2 # first column is y
  for j in range(1,n):
    for i in range(n-j):
            # create pyramid by updating other columns
      result[i][j] = (result[i+1][j-1] - result[i][j-1]) / (data_1[i+j] - data_1[i])
  return result[0]

def newtonInterp(xdata,ydata):

    coeff = newtondd(xdata,ydata)
    poly = lambda x: polynest(x,coeff,xdata)
    return poly

def chebyshevRoots(n):
  roots = [(math.cos((2 * i + 1) * math.pi / (2 * n + 2))) for i in range(n)]
  return roots

def chebyshevInterp(func, numroots, interval, x):
  lb = 0.5 * (interval[1] - interval[0])
  ub = 0.5 * (interval[1] + interval[0])
  new_f = [func(math.cos(math.pi * (k + 0.5) / numroots) * lb + ub) for k in range(numroots)]
  fac = 2.0 / numroots
  c = [fac * sum([new_f[k] * math.cos(math.pi * j * (k + 0.5) / numroots) for k in range(numroots)]) for j in range(numroots)]
  
  bottom = (2.0 * x - interval[0] - interval[1]) * (1.0 / (interval[1] - interval[0]))
  top = 2.0 * bottom
  (f, fx) = (c[-1], 0)             
  for i in c[-2:0:-1]:           
    (f, fx) = (top * f - fx + i, f)
  return bottom * f - fx + 0.5 * c[0]

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
  x = lusolve(matrixTmatrix, matrixTvector)
  r = vector - np.dot(matrix,x)
  return x, r 

def qrsolve(matrix, vector):
  (m,n) = matrix.shape
  if m == n:
    x = lusolve(matrix,vector)
    r = 0
  else:
    x,r = leastSquares_lu(matrix, vector)

  resid = 0
  for i in range(len(r)):
    resid += r[i]
  return matrix, resid

# A function that approximates integrals using the Simpson methods.
# I personally recommend using this method whenever one needs to compute a definite integral.
# The algorithm is fairly straightforward, and the proof is amusing. 
# Please contact me if you would like to know the proof.
# I will be happy to walk you through.
def simpson(function, a, b):
  subintervals = 2 #divide the interval a,b into two equal subintervals of length h
  h = ( b - a )/subintervals
  x = []
  fx = [] 
  i = 0
  while i<= subintervals: 
      x.append(a + i * h) 
      fx.append(function(x[i])) 
      i += 1

  answer = 0
  i = 0
  while i<= subintervals: 
      if i == 0 or i == subintervals: 
          answer+= fx[i] 
      elif i % 2 != 0: 
          answer+= 4 * fx[i] 
      else: 
          answer+= 2 * fx[i] 
      i+= 1
  answer = answer * (h / 3) 
  return answer

# A function that approximates integrals using the composite Simpson methods.
def compositeSimpson(function, subintervals, a, b):
  h = ( b - a )/subintervals
  x = []
  fx = [] 
  i = 0
  while i<= subintervals: 
      x.append(a + i * h) 
      fx.append(function(x[i])) 
      i += 1

  answer = 0
  i = 0
  while i<= subintervals: 
      if i == 0 or i == subintervals: 
          answer+= fx[i] 
      elif i % 2 != 0: 
          answer+= 4 * fx[i] 
      else: 
          answer+= 2 * fx[i] 
      i+= 1
  answer = answer * (h / 3) 
  return answer

# A function that approximates integrals using the adaptive Simpson methods.
def adaptiveSimpson(function, a, b, tol = 0.00001):
  h = 0.5 * ( b - a )

  x0 = a
  x1 = a + 0.5 * h
  x2 = a + h
  x3 = a + 1.5 * h
  x4 = b

  f0 = function( x0 )
  f1 = function( x1 )
  f2 = function( x2 )
  f3 = function( x3 )
  f4 = function( x4 )

  #Compute S(a,c)
  s0 = h * ( f0 + 4.0 * f2 + f4 ) / 3.0
  #Compute S(c,b)
  s1 = h * ( f0 + 4.0 * f1 + 2.0 * f2 + 4.0 * f3 + f4 ) / 6.0

  if abs( s0 - s1 ) >=  tol:
      s = adaptiveSimpson( function, x0, x2, 0.5 * tol ) + adaptiveSimpson( function, x2, x4, 0.5 * tol )        
  else:
      s = s1 + ( s1 - s0 ) / 15.0
  return s

# A function that approximates integrals using the Gaussian quadrature methods.     
def gaussQuad(fxn, n, lb = -1, ub = 1):
  nonstandardinterval = 0
  if ((lb != -1)) or ((ub != 1)):
    nonstandardinterval = 1

  nodes,weight = np.polynomial.legendre.leggauss(n)

  if (nonstandardinterval):
    mdpt = (lb+ub)/2.0
    halflength = (ub-lb)/2.0
    nodes = halflength*nodes+mdpt

  y = fxn(nodes)
  value = np.dot(weight,y)

  if (nonstandardinterval):
    value = halflength*value

  return value
