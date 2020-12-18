import numpy as np
import matplotlib.pyplot as plt
import numerics0_manalu as poly
import numerics4_manalu as num



linearmodel = 0
cubicmodel = 0
qrlinear = 1
qrcubic = 1



if (linearmodel):

    # Here's the example problem from the notes
    A = np.array([[1,1],[1,2],[1,3],[1,4],[1,5]])
    b = np.array([1.1, 1.9, 3.2, 3.8, 5])

    # You don't need to do any of this part if you write a function that does the leastSquares solution using LU.
    # I include it here as suggestions or ideas for numpy functions to do the things you want to do.
    # Store the transpose of A since it is used a lot
    At = np.transpose(A)  # Actually, we don't need to store this as we could instead just compute and store AtA and y
    # Apply the transpose to A and B
    AtA = np.dot(At,A)  # If we didn't want to store At we could just use np.dot(np.transpose(A),A)
    y = np.dot(At,b)    # And here too we could use np.dot(np.transpose(A),b)
    print("A^tA =")
    print(AtA)
    print("A^tb =")
    print(y)


    # Now we have the normal equations to solve.
    # Let's use our own implementations.  (If you want to check that your implementation is working well, compare with something from numpy like np.linalg.solve.
    x,r = num.leastSquares_lu(A,b)
    print("The least squares solution is x =")
    print(x)

    #r = b - np.dot(A,x)
    print("The residual is r = b-Ax =")
    print(r)
    # Define a function that will compute the values of the line with coefficients in coeffs
    def linfit(value,coeffs):
        return coeffs[0]+coeffs[1]*value

    # Let's create the set of data, plot the points, and then plot our least squares line.
    datapts_x = A[:,1]  # the x values of the data points are in column 1 of A
    datapts_y = b

    plt.show()

    t = np.arange(0.8,5.3,0.05)
    plt.plot(t,linfit(t,x),'w-')
    plt.plot(datapts_x,datapts_y,'bo')
    plt.show()


    plt.plot(datapts_x,datapts_y,'bo')
    t = np.arange(0.8,5.3,0.05)
    plt.plot(t,linfit(t,x),'r-')
    plt.show()

 

# ***************************************************
# *********** Models ********************************
# ***************************************************
# Let's make some random data from a model, add some errors, and then fit the data.

if (cubicmodel):

    # Make up a function
    # I'm writing a generic cubic
    def exactcube(x):
        return poly.polynest(x,np.array([2,-9,2,1]))
    # decid on a number of data points
    numpts = 14

    # Select some random x values.  We choose to randomly select values between -4 and 4
    x = 8*np.random.rand(numpts) - 4  # uniform values from (0,1), scaled to (0,8), shifted to (-4,4)
    # create the function values 
    y = np.zeros(x.shape)
    for idx in range(numpts):
        y[idx] = exactcube(x[idx])
    # add some offsets from the function values
    y = y+2*np.random.randn(y.size)  # added some error to the y-values drawn from a normal distribution with mean 0 and variance 2


    # So we just made some data that has 14 points coming from a particular cubic, but then the values are corrupted slightly.
    # We want to fit this data with a cubic.
    # To create A, we define a function generating the row for a particular data point
    def monomials(x):
       return np.array([1,x,x**2,x**3])

    # Create A
    A = np.zeros((numpts,4))
    for row in range(numpts):
        A[row,:] = monomials(x[row])

    # So we want to find the coefficients for our cubic model that have the smallest squared error: 
    # we want to solve the least squares problem for Ac=y where c is a vector of coefficients.
    # Then we will have as our least squares fit cubic the function p(x) = c[0] + c[1]x + c[2]x^2 + c[3]x^3
    coeff,resid = num.leastSquares_lu(A,y)

    def cubicfit(x):
        return poly.polynest(x,coeff)

    # Let's create the set of data, plot the points, and then plot our least squares line.
    datapts_x = A[:,1]  # the x values of the data points are in column 1 of A
    datapts_y = y

    plt.show()

    t = np.arange(-4.25,4.25,0.05)
    plt.plot(t,exactcube(t),'b-')
    plt.plot(datapts_x,datapts_y,'bo')
    plt.show()


    plt.plot(datapts_x,datapts_y,'bo')
    t = np.arange(-4.25,4.25,0.05)
    plt.plot(t,exactcube(t),'b-')
    plt.plot(t,cubicfit(t),'r-')
    plt.show()


if (qrlinear):

    A = np.array([[1,1],[1,2],[1,3],[1,4],[1,5]])
    b = np.array([1.1, 1.9, 3.2, 3.8, 5])
    Q,R = num.qr(A)
    Q = Q * (np.abs(Q)>1.0e-15)  # Here I'm removing things that should be zero so that it displays better.
    R = R * (np.abs(R)>1.0e-15)  # Same thing here.
    print("Q =")
    print(Q)
    print("R =")
    print(R)

    QtQ = np.transpose(Q) @ Q 
    # Here's an example of thresholding after your computations.  This must only be done when you are either confident that the values below the threshold should indeed be zero, or you have decided that values below the threshold are not relelvant and you knowingly get rid of them.  I did this above for printing reasons
    # Try turning this on and off and looking at the output.
    do_thresholding = 1
    if (do_thresholding):
        thresh = 1.0e-13
        print(np.greater(QtQ,thresh))
        QtQ = QtQ * (np.abs(QtQ)>thresh)
    print("Q^T Q should be the identity.  QtQ = ")
    print(QtQ)

    x,resid = num.qrsolve(A,b)
    print("The solution is x = ")
    print(x)
    print("The norm of the residual is %0.4e" % np.linalg.norm(resid))

if (qrcubic):
    # **********************IDENTICAL TO ABOVE cubicmodel other than using the qr with householder reflectors ********

    # Make up a function
    # I'm writing a generic cubic
    def exactcube(x):
        return poly.polynest(x,np.array([2,-9,2,1]))
    # decid on a number of data points
    numpts = 14

    # Select some random x values.  We choose to randomly select values between -4 and 4
    x = 8*np.random.rand(numpts) - 4  # uniform values from (0,1), scaled to (0,8), shifted to (-4,4)
    # create the function values 
    y = np.zeros(x.shape)
    for idx in range(numpts):
        y[idx] = exactcube(x[idx])
    # add some offsets from the function values
    y = y+2*np.random.randn(y.size)  # added some error to the y-values drawn from a normal distribution with mean 0 and variance 2


    # So we just made some data that has 14 points coming from a particular cubic, but then the values are corrupted slightly.
    # We want to fit this data with a cubic.
    # To create A, we define a function generating the row for a particular data point
    def monomials(x):
       return np.array([1,x,x**2,x**3])

    # Create A
    A = np.zeros((numpts,4))
    for row in range(numpts):
        A[row,:] = monomials(x[row])

    # So we want to find the coefficients for our cubic model that have the smallest squared error: 
    # we want to solve the least squares problem for Ac=y where c is a vector of coefficients.
    # Then we will have as our least squares fit cubic the function p(x) = c[0] + c[1]x + c[2]x^2 + c[3]x^3
    coeff,resid = num.qrsolve(A,y)

    print("The residual has norm %0.5e." % np.linalg.norm(resid))
    def cubicfit(x):
        return poly.polynest(x,coeff)

    # Let's create the set of data, plot the points, and then plot our least squares line.
    datapts_x = A[:,1]  # the x values of the data points are in column 1 of A
    datapts_y = y

    plt.show()

    t = np.arange(-4.25,4.25,0.05)
    plt.plot(t,exactcube(t),'b-')
    plt.plot(datapts_x,datapts_y,'bo')
    plt.show()


    plt.plot(datapts_x,datapts_y,'bo')
    t = np.arange(-4.25,4.25,0.05)
    plt.plot(t,exactcube(t),'b-')
    plt.plot(t,cubicfit(t),'r-')
    plt.show()

