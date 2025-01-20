import math
import os

import numpy as np
from matplotlib import pyplot as plt

from Result import Result


def lipschitz_estimate(points):
    r = 1.1
    eps = 10E-6
    lamb_max = max([math.fabs(points[i][1]-points[i-1][1])/(points[i][0]-points[i-1][0]) for i in range(1, len(points))])
    x_max = max([points[i][0]-points[i-1][0] for i in range(1, len(points))])

    def build_list(i, n):
        if n == 1: return [i]
        else:
            if i == 1: return [i, i+1]
            elif i == n: return [i, i-1]
            else: return [i-1, i, i+1]

    n = len(points)

    H = list()
    for i in range(1, n):
        lamb = max([math.fabs(points[j][1]-points[j-1][1])/(points[j][0]-points[j-1][0]) for j in build_list(i, n-1)])
        gamma = (lamb_max/x_max)*(points[i][0]-points[i-1][0])
        H.append(max(eps, lamb, gamma))

    mu = [r*h for h in H]
    return mu

def build_F(points, mu):
    def F(t):
        conditions = [(points[i-1][0] <= t) & (t < points[i][0]) for i in range(1, len(points))]
        funcs = [np.vectorize(lambda x, i=i: np.array([points[i-1][1]-mu[i-1]*(x-points[i-1][0]), points[i][1]+mu[i-1]*(x-points[i][0])]).max()) for i in range(1, len(points))]
        return np.piecewise(t, conditions, funcs)
    return F

def min_F(points, mu):
    x = [(points[i][0]+points[i-1][0])/2 - (points[i][1]-points[i-1][1])/(2*mu[i-1]) for i in range(1, len(points))]
    r = [(points[i][1]+points[i-1][1])/2 - mu[i-1]*(points[i][0]-points[i-1][0])/2 for i in range(1, len(points))]
    p = list(zip(x,r))
    t = min([i for i in range(1, len(points))], key=lambda i: (points[i][1]+points[i-1][1])/2 - mu[i-1]*(points[i][0]-points[i-1][0])/2)
    arg = min(p, key=lambda p: p[1])[0]
    return arg, t

def minimize(f, bounds, count_limit=None, save_file=None):
    eps = 10E-4 * (bounds[1] - bounds[0])  # Точность
    points = [(bounds[0], f(bounds[0])), (bounds[1], f(bounds[1]))]  # Точки на которых происходят вычисления
    counter = 2  # кол-во вычислений функции f
    diff = bounds[1] - bounds[0]

    # Пока разность между сгенер. точками x не меньше эпсилона
    while True:
        mu = lipschitz_estimate(points)  # аппроксимируем константу липшица кусочно-линейно
        arg, t = min_F(points, mu)

        x0 = min(points, key=lambda p: p[1])[0]
        y0 = min(points, key=lambda p: p[1])[1]

        diff = points[t][0]-points[t-1][0]  # находим точность

        if diff < eps:
            break
        if count_limit != None:
            if counter == count_limit:
                break

        points.append((arg, f(arg)))  # добавляем новую точку
        points.sort(key=lambda x: x[0])  # сортируем точки
        counter += 1  # увеличиваем счетчик

    if save_file is not None:
        x, y = zip(*points)
        F = build_F(points, mu)
        xs = np.arange(bounds[0], bounds[1], 0.0001)
        fig, ax = plt.subplots()
        ax.plot(x, y, 'o')
        ax.plot(x0, y0, 'ro')
        ax.plot(xs, F(xs), label='Миноранта (F)')
        ax.plot(xs, f(xs), label='Целевая функция (f)')
        ax.legend(loc='best', ncol=2)

        dir = os.path.dirname(save_file)
        if not os.path.exists(dir):
            os.mkdir(dir)

        fig.savefig(save_file)
        plt.close(fig)

    return Result(points, counter, x0, y0)
