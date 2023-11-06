# метод сопряжённых градиентов

import util
import numpy as np


def conj_gradients(initialX, index, e):
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
    K = dfX
    t = -np.dot(dfX.T, K) / np.dot(np.dot(H, K).T, K)

    while np.linalg.norm(dfX) > e:
        X = X + t * K
        fX, dfX = util.derivative(X, index)
        f.append(fX)
        x1.append(X[0])
        x2.append(X[1])
        K = dfX + (np.linalg.norm(dfX)**2) / (np.linalg.norm(K)**2) * K
        t = -np.dot(dfX.T, K) / np.dot(np.dot(H, K).T, K)
        i += 1
        iterations.append(i)

    util.plot_graph(x1scale, x2scale, w, x1, x2, 'Метод сопряжённых градиентов')
    util.print_table(iterations, x1, x2, f, "Метод сопряжённых градиентов:")
