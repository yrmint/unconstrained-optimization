# метод Бройдена

import util
import numpy as np


def matr(matrix):
    return np.array([[matrix[0]], [matrix[1]]])


def broyden(initialX, index, e):
    H = util.hesse(index)

    # Область построения
    x1scale = np.arange(0, 16, 0.1)
    x2scale = np.arange(0, 16, 0.1)
    x1scale, x2scale = np.meshgrid(x1scale, x2scale)
    w = util.func([x1scale, x2scale], index)

    # метод бройдена
    X = initialX
    fX, dfX = util.derivative(X, index)
    i = 0
    iterations = [0]
    x1 = [X[0]]
    x2 = [X[1]]
    f = [util.func(X, index)]

    n = -np.eye(2)
    K = dfX     # 0-вой шаг
    t = - (dfX.dot(K)) / (K.dot(H).dot(K))
    bX = X
    [fbX, dfbX] = util.derivative(bX, index)
    X = X + t * K

    [fX, dfX] = util.derivative(X, index)   # последующие шаги
    i = 1
    iterations.append(i)
    x1.append(X[0])
    x2.append(X[1])
    f.append(fX)
    dg = dfX - dfbX
    dx = X - bX
    z = dx - n @ dg
    z = matr(z)
    dg = matr(dg)
    dn = (z @ z.T) / (z.T @ dg)
    n = n + dn
    K = -(n.dot(dfX))
    t = -(dfX.dot(K)) / (K.dot(H).dot(K))

    while np.linalg.norm(dfX) > e:
        bX = X
        X = X + t * K
        i += 1
        iterations.append(i)
        x1.append(X[0])
        x2.append(X[1])
        [fbX, dfbX] = util.derivative(bX, index)
        [fX, dfX] = util.derivative(X, index)
        f.append(fX)
        dg = dfX - dfbX
        dx = X - bX
        z = dx - n @ dg
        dn = (z @ z.T) / (z.T @ dg)
        n = n + dn
        K = -n @ dfX
        t = -(dfX.dot(K)) / (K.dot(H).dot(K))

    util.plot_graph(x1scale, x2scale, w, x1, x2, 'Метод Бройдена')
    util.print_table(iterations, x1, x2, f, "Метод Бройдена:")
