from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom


x_points = [1, 2, 3]
y_points = [1, 1, 0]



def plot_function_a(a, b):
    def f(x):
       return 1 + a*(x-1) - b*((x-1)**3)

    x = np.linspace(1,2, 100)

    plt.plot(x, f(x), color='red')

def calculated_function_x(t):

    return t**4 + 4*t + 1



def calculated_function_y(t):

    return (-1)*(t**4) + 8*(t**3) - 12*(t**2) + 8*t + 1



def plot_function_b(c,d):


    def f(x):
       return 1 + c*(x-2) - (3/4)*((x-2)**2) + d*((x-2)**3)

    x = np.linspace(2, 3, 100)

    plt.plot(x, f(x), color='orange')


def plot_cubic_spline(x_points, y_points):
    if np.any(np.diff(x_points) < 0):
        indexes = np.argsort(x_points).astype(int)
        x_points = np.array(x_points)[indexes]
        y_points = np.array(y_points)[indexes]

    f = CubicSpline(x_points, y_points, bc_type= 'natural')
    x_new = np.linspace(min(x_points), max(x_points), 100)
    y_new = f(x_new)

    plt.plot(x_new, y_new)
    plt.scatter(x_points, y_points)
    plt.title('Cubic Spline interpolation')


def exercise1():
    plot_cubic_spline(x_points, y_points)
    plot_function_b(-1/2, 1/4)
    plot_function_a(1/4, 1/4)
    plt.show()


def Beinstein(n, k):
    coeff = binom(n, k)

    def _bpoly(x):
        return coeff * x ** k * (1 - x) ** (n - k)

    return _bpoly


def Bezier(points, num=200):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for ii in range(N):
        curve += np.outer(Beinstein(N - 1, ii)(t), points[ii])
    return curve


def exercise2():
    xp = np.array([1, 2, 3, 4, 6])
    yp = np.array([1, 3, 3, 3, 4])

    x, y = Bezier(list(zip(xp, yp))).T

    plt.plot(x,y)
    plt.plot(xp, yp, "ro")
    plt.plot(xp, yp, "b--")

    t = np.linspace(1, 6, 10_000)

    x_t = calculated_function_x(t)
    y_t = calculated_function_y(t)

    plt.plot(x_t, y_t)


    plt.show()