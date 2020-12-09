import numpy as np
import matplotlib.pyplot as plt
import numerics1_manalu as num

def fxn(x):
    return np.exp(-0.5*x) + np.sin(x) - 2*x

ainput = 0.0  # left endpoint
binput = 1.0  # and right endpoint of the interval for bisection.
xinit  = 1.0  # this is the inital estimate for fixed point iteration and newton

# You can define your own functions and intervals.

# change these to 1 if you want to test that section
# set them to zero if you don't.
run_bisection = 0
run_plotting  = 0
run_fixedpt   = 0
run_newton    = 1

if (run_bisection):
  print("\n\n ****** BISECTION ***** \n")

  rt, roots = num.bisection(fxn,ainput,binput)
  print("With default tolerance, bisection gets the approximate value %0.12f." % (rt))
  print("This took %d iterations." % (roots.size))

#print(roots)


if (run_plotting):
  t = np.arange(ainput,binput,0.001)

  # plotted the function
  plt.plot(t, fxn(t), 'r')
  # plot the x-axis
  plt.plot(t, 0.0*t, 'k')
  # plot the endpoints
  plt.plot([ainput, binput],[0.0,0.0],'go')
  plt.ion()
  plt.pause(0.4)

  inter = np.array([ainput, binput])
  twozeros = [0.0,0.0]
  for idx in range(roots.size):
    # plotted the function
    plt.plot(t, fxn(t), 'r')
    # plot the x-axis
    plt.plot(t, 0.0*t, 'k')
    plt.plot(inter,twozeros,'go')
    md = roots[idx]
    plt.plot([md],[0.0],'bo')
    inter = np.array([roots[idx-1], md])
    if (idx > 5):
      plt.xlim(0.6,0.7)
    plt.ion()
    plt.pause(0.4)


if (run_fixedpt):
  print("\n\n ****** FIXED POINT ITERATION ***** \n")

  # we need to convert the function above to a fixed point iteration.  Do that here:
  def gxn(x):
    return x + fxn(x)

  tol = 1.0e-6  # default value not passed, just written for the print statment.
  rt, roots = num.fixedpt(gxn,xinit)
  print("With tolerance = %0.2e, FPI gets the approximate value %0.12f." % (tol,rt))
  rt_old = rt
  print(roots)


if (run_newton):
  print("\n\n ****** NEWTON ***** \n")
  # we need to define the derivative of the function for Newton's Method
  def dfxn(x):
    return -0.5*np.exp(-0.5*x) + np.cos(x) - 2

  # now we run Newton's method, starting at the same point as FPI
  tol = 1.0e-6  # default value not passed, just written for the print statment.
  rt, roots = num.newton(fxn, dfxn,xinit)
  print("With tolerance = %0.2e, FPI gets the approximate value %0.12f." % (tol,rt))
  print("This took %d iterations." % (roots.size))

  # we can run it again and get all 12 decimal places
  tol = 1.0e-12  # default value not passed, just written for the print statment.
  rt, roots = num.newton(fxn, dfxn,xinit,tol)
  print("With tolerance = %0.2e, FPI gets the approximate value %0.12f." % (tol,rt))
  print("This took %d iterations." % (roots.size))
  print("\n Thought exercise: For this function with this initial estimate, what do you notice about the change in output and number of iterations and what does it mean about the result returned using the default tolerance?\n\n")
