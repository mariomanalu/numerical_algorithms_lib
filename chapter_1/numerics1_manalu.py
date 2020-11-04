#A script that performs bisection methods to find the root/roots of a function in Python

import numpy as np

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
    iteration = 0
    roots = np.zeros((maxIter,1))
    range = binput - ainput

    #If f(ainput) and f(binput) have the same signs, then there is no root in between ainput and binput
    if fa * fb >= 0:
        raise ValueError("There is no root in this interval")
    else:
        if (fxn(ainput) == 0):
            return 0, roots
        elif (fxn(binput) == 0):
            return 0, roots

    while (range > tol) and (iteration < maxIter):
        range = 0.5 * range
        c = ainput + range
        roots[iteration] = c
        fc = np.sign(fxn(c))

        if fa * fc > 0:
            ainput = c
        else:
            binput = c

        iteration += 1

    return c, roots
