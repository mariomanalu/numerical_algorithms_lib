import numpy as np
import numerics5_blanchard as num

run_simpson = 1
run_compositesimpson = 1
run_adaptivequad = 1
run_guassquad = 1


if (run_simpson):
    print("\n ******* Test Simpson's Method ******* \n")

    # Simpson's method on a single interval is a simple implementation.  
    # It is used in adaptive simpon's method even though we probalby wouldn't use it alone.
    # Implement simpsons method as      simpson(function,a,b) 
    # to compute the single simpson approximation of the definite integral from a to b of f(t).

    # This is example 5.6 from Sauer's first edition.
    simpApprox = num.simpson(np.log,1,2)
    print("Examp 5.6: Simpson Appx is %f." % simpApprox)


if(run_compositesimpson):
    print("\n ******* Test Composite Simpson's Method ******* \n")
  
    # Composite simpson is usefule and relatively easy to implement.
    # I recommend exploiting the slicing capabilities of numpy arrays.  Other hints: you might like np.linspace and np.sum.
    # Implement composite simpson's method as           compositeSimpson(function, n, a, b)
    # where n is the number of intervals, and [a,b] is the interval on which you are integrating.

    # This is example 5.8 from Sauer's first edition.
    compSimpApprox = num.compositeSimpson(np.log,4,1,2)
    print("Examp 5.8: Composite Simpson Appx is %f." % compSimpApprox)


if(run_adaptivequad):
    print("\n ******* Test Adaptive Quadrature ******* \n")

    # Implementing Adaptive Quadrature is fun.  It's recursive.  Recursion is always fun.  
    # However, you want to add a recursion safety net.  So, it is important to give the function 
    #  an input that counts the recursion level, similar to a max iteration.
    # My implementation has the form
    #
    #          adaptiveSimpson(fxn,a,b,tol=1e-12,Sab=0.0,recursion_counter=0)
    #
    # where Sab and recursion counter are passed only internally to the function. 
    # I used a maximum recursion depth of 20 since that has the potential to call the function 2^20 (about a million) times. 
    # Sab is the value of Simpson's Method on the interval you are currently working on [a,b].
    # you pass Sab since you already computed it in the last iteration.
    # 
    # In the lecture, I don't think I pointed out that you need to use the portion of the tolerance that belongs to that interval.
    # So, when you have a tolerance of epsilon, and you pass a tolerance to the two half intervals, 
    # they each need a tolerance of epsilon/2.  Then, in the end, the total integral will meet the tolerance epsilson.
    #
    # I would write your own version but then read the version in the book.  
    # It's an interesting take on recursion though it has not safety stopping criterion for the recursion.
    
    # This is Example 5.12 in Sauer's first edition.  You should get something like 2.5008.
    def fff(x):
        return 1.0+np.sin(np.exp(3*x))
    adaptApprox = num.adaptiveSimpson(fff,-1,1,0.00005)
    print("Adaptive Simpson Approximation is %f" % adaptApprox)
    adaptApprox = num.adaptiveSimpson(fff,-1,1)
    print("Adaptive Simpson Approximation is %f" % adaptApprox)

if (run_guassquad):
    print("\n ******* Test Gaussian Quadrature *******  \n")

    # Implement Gaussian Quadrature as       gaussQuad(function, n, a, b)
    # where n is the number of nodes (roots of Legendre polynomial p_n(x)), 
    # and [a,b] is the interval on which you are integrating.
    # As mentioned in the video lecture, use the numpy.polynomial.legendre.leggauss(n) to get the roots and weights.

    # This is Example 20 in the notes with some extra computation.
    actualvalue = 0.536610608871

    def fxn(x):
        return x * np.exp(-x)

    a=1
    b=3
    n=3

    myGq = num.gaussQuad(fxn,n,a,b)
    print("\t n = %d" % n)
    print("Guassian Quadrature approximation = %0.12f" % myGq)
    print("The error is %0.12f" % np.abs(myGq-actualvalue))



    n=5
    myGq = num.gaussQuad(fxn,n,a,b)
    print("\n\t n = %d" % n)
    print("Guassian Quadrature approximation = %0.12f" % myGq)
    print("The error is %0.12f" % np.abs(myGq-actualvalue))    


    n=6
    myGq = num.gaussQuad(fxn,n,a,b)
    print("\n\t n = %d" % n)
    print("Guassian Quadrature approximation = %0.12f" % myGq)
    print("The error is %0.12f" % np.abs(myGq-actualvalue)) 

    n=7
    myGq = num.gaussQuad(fxn,n,a,b)
    print("\n\t n = %d" % n)
    print("Guassian Quadrature approximation = %0.12f" % myGq)
    print("The error is %0.12f" % np.abs(myGq-actualvalue)) 
