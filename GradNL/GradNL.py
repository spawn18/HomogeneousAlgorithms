import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline, Akima1DInterpolator

import statistics
from result import Result

ALGO_NAME = "GradNL"
KSI = 10E-6
R = 1.1

def lipschitz_estimate(points):
    lamb_max = max([math.fabs(points[i][1]-points[i-1][1])/(points[i][0]-points[i-1][0]) for i in range(1, len(points))])
    x_max = max([points[i][0]-points[i-1][0] for i in range(1, len(points))])

    def build_list(i, n):
        if n == 1: return [i]
        else:
            if i == 1: return [i, i+1]
            elif i == n: return [i-1, i]
            else: return [i-1, i, i+1]

    n = len(points)

    H = list()
    for i in range(1, n):
        lamb = max([math.fabs(points[j][1]-points[j-1][1])/(points[j][0]-points[j-1][0]) for j in build_list(i, n-1)])
        gamma = (lamb_max/x_max)*(points[i][0]-points[i-1][0])
        H.append(max([KSI, lamb, gamma]))

    mu = np.repeat([R * h for h in H], 2)

    return mu
def grad_boost(points, mu, grad_smoother):
    x,y = zip(*points)
    spline = CubicSpline(x, y, bc_type='clamped')  # вычисляем сплайн по точкам

    n = len(points)

    D = spline.derivative()
    vel = np.array([D(x) for x in spline.x])
    vel = np.repeat(vel, 2)[1:-1]
    vel = np.array([grad_smoother(-1 * v if i % 2 == 0 else v) for i, v in enumerate(vel)])

    for i in range(1, n):
        k = (points[i][1] - points[i - 1][1]) / (points[i][0] - points[i - 1][0])

        mu1 = mu[2 * (i - 1)] * vel[2 * (i - 1)]
        mu2 = mu[2 * (i - 1) + 1] * vel[2 * (i - 1) + 1]

        if k >= 0:
            mu[2 * (i - 1)] = max([mu1, KSI])
            mu[2 * (i - 1) + 1] = max([mu2, k + KSI])
        else:
            mu[2 * (i - 1)] = max([mu1, -k + KSI])
            mu[2 * (i - 1) + 1] = max([mu2, KSI])

    return mu

def build_F(points, mu):
    def F(t):
        conditions = [(points[i-1][0] <= t) & (t < points[i][0]) for i in range(1, len(points))]
        funcs = [lambda x, i=j: np.max([points[i-1][1]-mu[2*(i-1)]*(x-points[i-1][0]), points[i][1]-mu[2*(i-1)+1]*(points[i][0]-x)], axis=0) for j in range(1, len(points))]
        return np.piecewise(t, conditions, funcs)
    return F

def min_F(pts, mu):
    x = [(mu[2*(i-1)+1]*pts[i][0]+mu[2*(i-1)]*pts[i-1][0])/(mu[2*(i-1)] + mu[2*(i-1)+1]) - (pts[i][1]-pts[i-1][1])/(mu[2*(i-1)] + mu[2*(i-1)+1]) for i in range(1, len(pts))]
    r = [max([pts[i-1][1]-mu[2*(i-1)]*(x[i-1]-pts[i-1][0]), pts[i][1]-mu[2*(i-1)+1]*(pts[i][0]-x[i-1])]) for i in range(1, len(pts))]
    p = list(zip(x,r))
    arg = min(p, key=lambda p: p[1])[0]
    return arg

def minimize(funcs, grad_smoother):
    results = list()

    for i, f in enumerate(funcs):
        eps = 10E-4 * (f.bounds[1] - f.bounds[0])  # Точность
        points = [(f.bounds[0], f.eval(f.bounds[0])), (f.bounds[1], f.eval(f.bounds[1]))]  # Точки на которых происходят вычисления
        counter = 2  # кол-во вычислений функции f

        # Пока разность между сгенер. точками x не меньше эпсилона
        while True:
            x, y = zip(*points)  # разбиваем на 2 массива, x и y

            mu = lipschitz_estimate(points)  # аппроксимируем константу липшица кусочно-линейно
            mu = grad_boost(points, mu, grad_smoother)
            arg = min_F(points, mu)

            x0 = min(points, key=lambda p: p[1])[0]
            y0 = min(points, key=lambda p: p[1])[1]

            diff = min([math.fabs(arg-p[0]) for p in points]) # находим точность

            if diff < eps:
                break

            points.append((arg, f.eval(arg)))  # добавляем новую точку
            points.sort(key=lambda x: x[0])  # сортируем точки
            counter += 1  # увеличиваем счетчик

        if statistics.SAVE:
            x, y = zip(*points)
            F = build_F(points, mu)
            xs = np.arange(f.bounds[0], f.bounds[1], 0.0001)
            plt.plot(x, y, 'o', label='Точки испытаний')
            plt.plot(x0, y0, 'ro', label='Минимум')
            plt.plot(xs, F(xs), 'red', label='Миноранта')
            plt.plot(xs, f.eval(xs), 'black', label='Целевая функция')
            plt.legend(loc='best', ncol=2)
            plt.grid()
            plt.savefig(statistics.algo_path(ALGO_NAME, i+1), dpi=300)
            plt.close()


        success = statistics.check_convergence(f.min_x, x, 2*eps)
        results.append(Result(points, counter, x0, y0, f.min_y, success))

    return results
