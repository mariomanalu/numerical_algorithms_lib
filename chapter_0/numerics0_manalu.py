#A script that performs nested polynomial evaluation in Python

import numpy as np

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
