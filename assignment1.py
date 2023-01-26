# This is a sample Python script.
import math
import matplotlib.pyplot as plt
from numpy import array,linspace,sqrt,sin
from numpy.linalg import norm


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

def f(x):
    return x**4 - x**3 -10


def h(x):
    # print(x)
    #return math.log(x-x**3)
    return math.cos(x)
    #return math.cos(x)**2
    #return (2*(x**3)-math.exp(x))/(3*(x**2)-1)
    #return (x**3 + math.exp(x))


#Given initial interval [a,b] such that f(a)f(b) < 0, n is number of steps
def bisection_method(a, b, n):
    condition = True
    i = 1
    while condition:
        x = (a + b) / 2
        if f(x) < 0:
            a = x
        else:
            b = x
        if i == n:
            condition = False
        else:
            condition = True
            i = i + 1
    print("Required root is: ", x)

def fixed_point_iteration(start_value, number_of_iterations):
    i = 0

    x = start_value
    saved_iterations = list()

    for i in range(number_of_iterations):
        x = float(x)
        saved_iterations.append(h(x))
        x = h(x)

    return saved_iterations

def fixedp(f,x0,tol=10e-5,maxiter=100):
 """ Fixed point algorithm """
 e = 1
 itr = 0
 xp = []
 while(e > tol and itr < maxiter):
  x = f(x0)      # fixed point equation
  e = norm(x0-x) # error at the current step
  x0 = x
  xp.append(x0)  # save the solution of the current step
  itr = itr + 1
 return x,xp


def g(x):
 x[0] = 1/4*(x[0]*x[0] + x[1]*x[1])
 x[1] = sin(x[0]+1)
 return array(x)

#hej

def exercise1a():
    a = input("First approximation root: ")
    b = input("Second approximation root: ")
    n = input("Input number of bisection steps: ")
    a = float(a)
    b = float(b)
    n = int(n)  # number of steps

    if f(a) * f(b) > 0:
        print("Try again")
    else:
        bisection_method(a, b, n)


def exercise1b():
    a = input("First approximation root: ")
    b = input("Second approximation root: ")
    error_pow = input("Set error pow: ")
    error_pow = float(error_pow)
    error = 10**(error_pow)

    n = math.log( 1/(2*error),2)
    a = float(a)
    b = float(b)
    print(n)
    n = int(n)  # number of steps



    if f(a) * f(b) > 0:
        print("Try again")
    else:
        bisection_method(a, b, n)




def trying_fixed_point_iteration():
    x, xf = fixedp(g, [0, 1])
    print('   x =', x)
    print('f(x) =', g(xf[len(xf) - 1]))

    f = lambda x: sqrt(x)

    x_start = .5
    xf, xp = fixedp(f, x_start)

    x = linspace(0, 2, 100)
    y = f(x)
    plt.plot(x, y, xp, f(xp), 'bo',
             x_start, f(x_start), 'ro', xf, f(xf), 'go', x, x, 'k')

    plt.show()







iterations = list()
def fixed_point_iteration_recursive(start_value, n):
    iterations.append(float(h(start_value)))

    if( n == 0 ):
        return iterations
    else:
        fixed_point_iteration_recursive(h(start_value), n-1)




def exercise2a():
    print(fixed_point_iteration(0.7390, 100))


exercise2a()