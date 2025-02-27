import math
import statistics

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from result import Result

ALGO_NAME = "mishin_local_parab_grad_best"


def lipschitz_estimate(points):
    r = 1.1
    eps = 10E-6
    lamb_max = max([math.fabs(points[i][1] - points[i - 1][1]) / (points[i][0] - points[i - 1][0]) for i in range(1, len(points))])
    x_max = max([points[i][0] - points[i - 1][0] for i in range(1, len(points))])

    def build_list(i, n):
        if n == 1:
            return [i]
        else:
            if i == 1:
                return [i, i+1]
            elif i == n:
                return [i-1, i]
            else:
                return [i - 1, i, i + 1]

    n = len(points)

    H = list()
    for i in range(1, n):
        lamb = max([math.fabs(points[j][1] - points[j - 1][1]) / (points[j][0] - points[j - 1][0]) for j in
                    build_list(i, n - 1)])
        gamma = (lamb_max / x_max) * (points[i][0] - points[i - 1][0])
        H.append(max(eps, lamb, gamma))

    mu = np.repeat([r*h for h in H], 2)
    return mu

def grad_accel_boost(spline, points, mu, grad_smoother, accel_smoother):
    D = spline.derivative()
    DD = D.derivative()

    vel = np.array([D(x) for x in spline.x])
    vel = np.repeat(vel, 2)[1:-1]
    vel = np.array([grad_smoother(-1 * v if i % 2 == 0 else v) for i, v in enumerate(vel)])

    accel = np.array([DD(x) for x in spline.x])
    accel = np.repeat(accel, 2)[1:-1]
    accel = np.array([accel_smoother(-1 * a if i % 2 == 0 else a) for i, a in enumerate(accel)])

    n = len(points)

    for i in range(1, n):
        k = math.fabs(points[i][1] - points[i - 1][1]) / (points[i][0] - points[i - 1][0])
        if mu[2*(i-1)] * vel[2*(i-1)] > max([0, k]):
            mu[2*(i-1)] *= vel[2*(i-1)]
        if mu[2*(i-1)+1] * vel[2*(i-1)+1] > max([0, k]):
            mu[2*(i-1)+1] *= vel[2*(i-1)+1]

    return mu, accel

def build_P(spline, points, mu):
    def F(t):
        conditions = [(points[i-1][0] <= t) & (t <= points[i][0]) for i in range(1, len(points))]
        funcs = [lambda x, i=j, s=spline: np.max([s(x)-mu[2*(i-1)]*(x-points[i-1][0]), s(x)+mu[2*(i-1)+1]*(x-points[i][0])], axis=0) for j in range(1, len(points))]
        return np.piecewise(t, conditions, funcs)
    return F

def convert_coefs(c, off1, off2):
    r = c
    off = off2 - off1
    c[1] = 3*r[0]*off + r[1]
    c[2] = 3*r[0]*off**2 + 2*r[1]*off + r[2]
    c[3] = r[0]*off**3 + r[1]*off**2 + r[2]*off + r[3]
    return c

def minimize_cubic_piece(c, offset, bounds):
    roots = np.roots(np.polyder(c))
    roots = roots[np.isreal(roots)] + offset
    roots = roots[np.logical_and(bounds[0] <= roots, roots <= bounds[1])].tolist()

    def eval(x):
        return np.polyval(c, x - offset)

    mins_x = list(bounds) + roots
    x0 = min(mins_x, key=eval)
    y0 = eval(x0)
    return (x0, y0)

def minimize_P(spline, points, mu, acc):
    mins = list()
    for i in range(1, len(spline.x)):
        k1 = mu[2*(i-1)]
        k2 = mu[2*(i-1)+1]

        delta = points[i][0]-points[i-1][0]
        q1 = acc[2*(i-1)]/delta
        q2 = acc[2*(i-1)+1]/delta

        a = q1-q2
        b = -2*q1*points[i-1][0]+2*q2*points[i][0]+k1+k2
        c = q1*points[i-1][0]**2-q2*points[i][0]**2-k2*points[i][0]-k1*points[i-1][0]
        roots = np.roots([a,b,c])
        x_intersect = list(filter(lambda x: points[i-1][0] <= x <= points[i][0], roots))[0]

        int1 = (points[i - 1][0], x_intersect)
        int2 = (x_intersect, points[i][0])

        c1 = convert_coefs(np.array([0, -q1, -mu[2 * (i - 1)], 0]), points[i - 1][0], points[i - 1][0])
        c2 = convert_coefs(np.array([0, -q2, mu[2 * (i - 1) + 1], 0]), points[i][0], points[i - 1][0])

        c1 = spline.c[:, i - 1] + c1
        c2 = spline.c[:, i - 1] + c2

        c1_min = minimize_cubic_piece(c1, points[i - 1][0], int1)
        c2_min = minimize_cubic_piece(c2, points[i - 1][0], int2)

        mins.append(min([c1_min, c2_min], key=lambda x: x[1]))

    arg = min(mins, key=lambda x: x[1])[0]
    return arg


def minimize(funcs, grad_smoother, accel_smoother):
    results = list()

    for i, f in enumerate(funcs):
        eps = 10E-4 * (f.bounds[1] - f.bounds[0])  # Точность
        points = [(f.bounds[0], f.eval(f.bounds[0])), (f.bounds[1], f.eval(f.bounds[1]))]  # Точки на которых происходят вычисления
        counter = 2  # кол-во вычислений функции f

        # Пока разность между сгенер. точками x не меньше эпсилона
        while True:
            x, y = zip(*points)  # разбиваем на 2 массива, x и y
            spline = CubicSpline(x, y, bc_type='clamped')  # вычисляем сплайн по точкам

            mu = lipschitz_estimate(points)
            mu, accel = grad_accel_boost(spline, points, mu, grad_smoother, accel_smoother)
            arg = minimize_P(spline, points, mu, accel)

            x0 = arg
            y0 = f.eval(arg)
            counter += 1

            diff = min([math.fabs(arg-p[0]) for p in points]) # находим точность

            if diff < eps:
                break

            points.append((arg, f.eval(arg)))  # добавляем новую точку
            points.sort(key=lambda x: x[0])  # сортируем точки

        """
        P = build_P(spline, points, mu)
        xs = np.arange(f.bounds[0], f.bounds[1], 0.0001)
        plt.plot(x, y, 'o', label='Точки испытаний')
        plt.plot(x0, y0, 'or', label='Точка следующего испытания')
        plt.plot(xs, spline(xs), 'blue', label='Интерполянт (m)')
        plt.plot(xs, P(xs), 'red', label='Критерий (P)')
        plt.plot(xs, f.eval(xs), 'green', label='Целевая функция (f)')
        plt.legend(loc='best', ncol=2)
        plt.savefig(os.path.join(statistics.algo_path(ALGO_NAME, i + 1), 'final'))
        plt.close()
        """

        success = statistics.check_convergence(f.min_x, x, 2*eps)
        results.append(Result(points, counter, x0, y0, f.min_y, success))
    return results