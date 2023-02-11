import numpy as np
import matplotlib.pyplot as plt



def divided_diff(x, y):
    '''
    function to calculate the divided
    differences table
    '''
    n = len(y)
    coef = np.zeros([n, n])
    # the first column is y
    coef[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = \
                (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])



    return coef


def newton_poly(coef, x_data, x):
    n = len(x_data) - 1
    p = coef[n]
    for k in range(1, n + 1):
        p = coef[n - k] + (x - x_data[n - k]) * p
    return p


#x = np.array([-1, 1, 2, 3])
#y = np.array([-3, 1, 3, 7])
# get the divided difference coef
def plot_results(x, y):
    a_s = divided_diff(x, y)[0, :]

    # evaluate on new data points
    x_new = np.arange(-5, 2.1, .1)
    y_new = newton_poly(a_s, x, x_new)

    plt.figure(figsize = (12, 8))
    plt.plot(x, y, 'bo')
    plt.plot(x_new, y_new)
    plt.show()


def proterm(i, value, x):
    pro = 1;
    for j in range(i):
        pro = pro * (value - x[j]);
    return pro;



def dividedDiffTable(x, y, n):
    for i in range(1, n):
        for j in range(n - i):
            y[j][i] = ((y[j][i - 1] - y[j + 1][i - 1]) /
                       (x[j] - x[i + j]));
    return y;


def applyFormula(value, x, y, n):
    sum = y[0][0];

    for i in range(1, n):
        sum = sum + (proterm(i, value, x) * y[0][i]);

    return sum;


def printDiffTable(y, n):
    for i in range(n):
        for j in range(n - i):
            print(round(y[i][j], 4), "\t",
                  end=" ");

        print("");


n = 4;
y = [[0 for i in range(10)]
     for j in range(10)];
x = [-1, 1, 2, 3];

y[0][0] = 3;
y[1][0] = 1;
y[2][0] = 3;
y[3][0] = 7;

y = dividedDiffTable(x, y, n);

printDiffTable(y, n);

value = 7;

print("\nValue at", value, "is",
      round(applyFormula(value, x, y, n), 2))