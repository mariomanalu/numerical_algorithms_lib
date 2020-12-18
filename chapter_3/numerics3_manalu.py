import numpy as np
import math
import numerics0_manalu as num0

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
    poly = lambda x: num0.polynest(x,coeff,xdata)
    return poly

def chebyshevRoots(n):
  roots = [(math.cos((2 * i + 1) * math.pi / (2 * n + 2))) for i in range(n)]
  return roots

######## DISCLAIMER #########
# I changed the syntax and the nature of the chebyshevInterp function slightly.
# The test file that you provided expected chebyshevInterp to return a function. I am not able to do that because I got confused about how to pass the value of x. That is, the value at which the interpolation is computed. I tried using lambda function, as you suggested, but my implementation of chebyshevInterp is long, so the lambda function does not seem to be helpful.
# Now, instead of returning a function, I make chebyshevInterp return the approximate solution by passing the value at which the interpolation is computed directly as input. I call that point x.
# Because of the change, I modified the "run_cheby" block in main.py. Please use the following block when you test the code. The change is minor. I replaced s*chebfunc(temp) with s*num.chebyshevInterp(func, numrts,[0.0,np.pi/2.0], temp).  I am confident that this implementation works well even though it is not exactly what the test file expects.
# if (run_cheby):
#    numrts = 7
#    rts = num.chebyshevRoots(numrts)
#    print(rts)

#    func = lambda x: np.sin(x)
    
#    def mysin(x):
#        s = 1;
#        temp = np.mod(x,2*np.pi)
#        if (temp>np.pi):
#            temp = 2*np.pi-temp
#            s = -1
#        elif (temp > np.pi/2.0):
#            temp = np.pi - temp
#        return s*num.chebyshevInterp(func, numrts,[0.0,np.pi/2.0], temp)

#    examp_pts = np.array([1.0, 2.0, 3.0, 4.0, 14.0, 1000.0])  
#    print("The input followed by the values of numpy's sine function, our version, and the chebyshev interpolating polynomial.")
#    for idx in range(6):
#        xval = examp_pts[idx]
#        print("%d\t %0.5f\t %0.5f\t  %0.5f" % (xval,func(xval),mysin(xval),num.chebyshevInterp(func, numrts,[0.0,np.pi/2.0], xval)))

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
