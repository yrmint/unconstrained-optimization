# метод Ньютона

import numpy as np
import util


def newton(initialX, index, e):
    H = util.hesse(index)

    # Область построения
    x1scale = np.arange(0, 16, 0.1)
    x2scale = np.arange(0, 16, 0.1)
    x1scale, x2scale = np.meshgrid(x1scale, x2scale)
    w = util.func([x1scale, x2scale], index)

    X = initialX
    fX, dfX = util.derivative(X, index)
    i = 0
    iterations = [0]
    x1 = [X[0]]
    x2 = [X[1]]
    f = [fX]

    while np.linalg.norm(dfX) > e:
        X = X - np.linalg.inv(H) @ dfX
        fX, dfX = util.derivative(X, index)
        f.append(fX)
        x1.append(X[0])
        x2.append(X[1])
        i += 1
        iterations.append(i)

    util.plot_graph(x1scale, x2scale, w, x1, x2, 'Метод Ньютона')
    util.print_table(iterations, x1, x2, f, "Метод Ньютона:")
