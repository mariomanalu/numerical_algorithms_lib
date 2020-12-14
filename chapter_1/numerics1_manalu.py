import numpy as np
import math
#A function that bisects interval to find the root/roots of a function in Python
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
