# метод наискорейшего подъёма (спуска)

import numpy as np
import util


def st_descent(initialX, index, e):
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

    t = -(dfX.dot(dfX)) / (dfX.dot(H).dot(dfX))

    while np.linalg.norm(dfX) > e:
        X = X + t * dfX
        fX, dfX = util.derivative(X, index)
        f.append(fX)
        x1.append(X[0])
        x2.append(X[1])
        t = -(dfX.dot(dfX)) / (dfX.dot(H).dot(dfX))
        i += 1
        iterations.append(i)

    util.plot_graph(x1scale, x2scale, w, x1, x2, 'Метод наискорейшего подъёма')
    util.print_table(iterations, x1, x2, f, "Метод наискорейшего подъёма:")
