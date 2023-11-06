# вспомогательные функции

import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable


def func(x, index):
    return index[0] * x[0] ** 2 + index[1] * x[1] ** 2 + index[2] * x[0] * x[1] + index[3] * x[0] + index[4] * x[1]


def derivative(x, index):
    fX = func(x, index)
    dfX = np.array([2 * index[0] * x[0] + index[2] * x[1] + index[3], 2 * index[1] * x[1] + index[2] * x[0] + index[4]])
    return fX, dfX


def plot_graph(x_1, x_2, w, x, y, text):
    plt.contour(x_1, x_2, w, levels=50, alpha=.5)
    plt.plot(x, y, label=text, color='black')
    plt.legend()
    plt.show()


def print_table(iterations, x1, x2, f, text):
    table = PrettyTable()
    table.add_column("i", iterations)
    table.add_column("x1", x1)
    table.add_column("x2", x2)
    table.add_column("f(X)", f)
    print(text)
    print(table)


def hesse(index):
    return np.array([[index[0] * 2, index[2]], [index[2], index[1] * 2]])
