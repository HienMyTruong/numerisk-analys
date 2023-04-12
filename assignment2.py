import math

import matplotlib.pyplot as plt
def f(x):
    return x**3 - x
    #return (math.exp(math.sin(x)**3) + x**6 - 2 * x**4 - x**3 - 1)

def df_dx(x):
    return 3 * x**2 - 1
    #return (6 * x**3 - 8*x - 3) * x**2 + math.exp(math.sin(x)**3)*(math.sin(x)**2)*math.cos(x)

def next_iteration_of_x(x):
    if df_dx(x) == 0:
        raise ValueError("Division by 0")
    return x - ((f(x)) / (df_dx(x)))


def newton_raph(x_init, nbr_of_iterations):


    i = 0
    x = x_init
    arr = []
    for i in range(nbr_of_iterations):
        x = float(x)
        arr.append(x)
        x = next_iteration_of_x(x)

    return arr



print(newton_raph(0.447213595499959, 50))
