import numpy as np

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
