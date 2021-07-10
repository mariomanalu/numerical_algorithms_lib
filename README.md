### Numerical Algorithms Lib

# Introduction

This file contains the implementation of various numerical algorithms.

Each algorithm has its own purpose.

1. They can be used to solve a simple linear equation whose exact solution is hard, if not impossible, to compute.

For instance, the square root of two. That said, we can use Fixed Point Iteration, Bisection Method, Newton Method to compute the approximation of the square root of two.

2. They can be used to solve a multivariable linear equations.

Multivariable linear equations can be written in matrix forms. The solutions boils down to manipulating the matrices in such a way that the matrix is in reduced, echelon form. Once the matrix is in reduced, echelon form, we can find a solution to one of the variables and substitute back to find the solutions to the rest of the variables

3. They can be used to approximate the result of some definite integrals.

Integrating a function can be computationally very expensive. Plus, we are normally happy with approximating integrals to some decimal digits -- we rarely need the exact solution. Thus, we can use Simpson's Method or Gaussian Quadrature to approximate some definite integrals at a lower cost.

  

This is an open source library. Thus, if you would like to implement and add some more algorithms, feel free to create a pull request.

  

# How to use:

1. Download / clone the repo.

2. The heart of this repo is numerical_algorithms_lib.py.

3. Import the file to your script. Or you can just add your script to the cloned repo.

4. If you need a reference, there are test files included in the folders.

  

# Tips:

1. If you are to need the root of a function, use Bisection Method. I like it because of it is O(logn) time complexity.

2. If you need to integrate a function, use the Simpson's Method. The algorithm is fairly straightforward: it just keeps integrating quadratic functions that covers the area under the input function's curve. The proof of this algorithm is super amusing. Please contact me if you would like to know the proof. I will be happy to walk you through the proof.

  

# Credits:

I learned about the algorithms in this file from reading [Numerical Analysis by T.Sauer](https://www.pearson.com/us/higher-education/program/Sauer-Numerical-Analysis-3rd-Edition/PGM1735484.html). I did not develop any of these algorithms. I am only implementing what has been conjectured and proven to be true. If you are interested in knowing more about these algorithms, definitely read the book. If the book is too expensive for you to buy, I would like to let you know that there is a free pdf version somewhere on the internet. I do not recommend you to do that, but I share the believe that educational books should be free or at least affordable.
  
# Disclaimer:

Please read and study the code before using it. Some of these algorithms have hard-coded degree of error to stop while loops. That is, once the solution is correct up to say the 6th decimal digit, the algorithm may stop. The stopping criterion may need to be adjusted based on your needs.

  

Keep in mind that these algorithms are not designed to compute exact solutions. There is a trade-off here: the algorithms can only approximate up to a certain degree, but they are guaranteed to be faster than algorithms that can compute the exact solutions, if such algorithms exist.
