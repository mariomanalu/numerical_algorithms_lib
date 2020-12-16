import numpy as np
import numerics2_manalu as num

#A = np.array([[4, 3, 1], [2, 2, 3], [1, 1, 4]])
A = np.array([[4.0, 3, 1], [2, 2, 3], [1, 1, 4]])
k = 500
#A = np.random.rand(k,k)
#print(A)

rowops = 0
luonly = 0
forwardsub = 0
backsub = 0
fbsub = 0
lucomplete = 0
jac = 0
gs = 1

if (rowops):
    A = num.rowswap(A,2,0)
    print("Row 0 and row 2 should have swapped.")
    print(A)

    A = num.rowswap(A,0,2)
    print("They should have swapped back")
    print(A)

    A = num.rowdiff(A,1,0,1/2.0)
    A = num.rowdiff(A,2,0,1/4.0)
    print("I should have done GaussElim to the first column.")
    print(A)


    A = num.rowdiff(A,2,1,1/2.0)
    print("I should have completed GaussElim to form an upper triangular matrix.")
    print(A)

    scale = 1/2.0
    A = num.rowscale(A,0,1/2.0)
    print("I should have scaled the first row by %0.1f" % scale)
    print(A)

if (luonly):
    #print("starting over *************************")
    A = np.array([[4.0, 3, 1], [2, 2, 3], [1, 1, 4]])
    print(A)

    #L,U = num.noswapLU(A)
    L,U,P = num.lu(A)
    print("Just did LU:")
    print("L is the matrix")
    print(L)
    print("U is the matrix")
    print(U)
    print("P is the matrix")
    print(P)

#print("Let's create a vector b")
#b = np.array([1.0, 0.0, 1.0])
#b = np.random.randn(k,1)
#print(b)

if (forwardsub):

    print("Conduct forward sub with L")
    y = num.forwardsub(L,b)
    print(y)

if (backsub):
    print("Now use backward sub to complete the solve")
    x = num.backsub(U,y)
    print(x)

if (fbsub):
    print("Now just solve it directly with L and U in one command")
    x = num.fbsolve(L,U,b)
    print(x)

if (lucomplete):
    #print("starting over *************************")
    A = np.array([[4.0, 3, 1], [2, 2, 3], [1, 1, 4]])
    print(A)
    print("And finally, reset A and use the code to do a complete solve from A")
    x = num.lusolve(A,b)
    print(x)

A=[[3, 1, -1],[2, -1, -1],[1, 3, 5]]
b = np.array([1,2,3])
xinit = np.zeros(b.shape)
if (jac):
    xinit = [1.0, 1.0, 1.0]
    x = num.jacobi(A,b,xinit,1e-10,1000)
    print(x)
     
xinit = np.zeros(b.shape)
if (gs):
    xinit = [1.0, 1.0, 1.0]
    x = num.gausssiedel(A,b,xinit,1e-10,1000)
    print(x)
