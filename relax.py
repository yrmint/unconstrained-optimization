# метод релаксации

import util
import numpy as np


def relax(initialX, index, e):
    H = util.hesse(index)

    # Область построения
    x1scale = np.arange(0, 16, 0.1)
    x2scale = np.arange(0, 16, 0.1)
    x1scale, x2scale = np.meshgrid(x1scale, x2scale)
    w = util.func([x1scale, x2scale], index)

    # метод релаксационный
    X = initialX
    fX, dfX = util.derivative(X, index)
    i = 0
    iterations = [0]
    x1 = [X[0]]
    x2 = [X[1]]
    f = [fX]
    K = np.array([0, dfX[1]])
    t = -(dfX.dot(K)) / (K.dot(H).dot(K))

    while np.linalg.norm(dfX) > e:
        X = X + t * K
        fX, dfX = util.derivative(X, index)
        f.append(fX)
        x1.append(X[0])
        x2.append(X[1])
        K = dfX
        t = -(dfX.dot(K)) / (K.dot(H).dot(K))
        i += 1
        iterations.append(i)

    util.plot_graph(x1scale, x2scale, w, x1, x2, 'Релаксационный метод')
    util.print_table(iterations, x1, x2, f, "Релаксационный метод:")
