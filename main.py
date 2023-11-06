import numpy as np
import relax
import newton
import st_descent
import conj_gradients
import broyden

if __name__ == '__main__':
    index = np.array([-39, -51, 16, 294, 532])  # Значения всех аргументов
    initialX = np.array([15, 15])  # Начальная точка
    e = 0.1

    relax.relax(initialX, index, e)
    st_descent.st_descent(initialX, index, e)
    newton.newton(initialX, index, e)
    conj_gradients.conj_gradients(initialX, index, e)
    broyden.broyden(initialX, index, e)
