# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.



def f(x):
    return x**4 - x**3 -10
#Given initial interval [a,b] such that f(a)f(b) < 0, n is number of iterations
def bisection_method(a, b, n):
    condition = True
    i = 1
    while condition:
        x = (a+b)/2
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


a = input("First approximation root: ")
b = input("Second approximation root: ")
n = input("Input number of iterations")
a = float(a)
b = float(b)
n = int(n)  #number of iterations

if f(a)*f(b) > 0:
    print("Try again")
else:
    bisection_method(a,b, n)