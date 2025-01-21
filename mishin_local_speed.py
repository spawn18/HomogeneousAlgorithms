import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
import os
from Result import Result

ALGO_NAME = "mishin_local_speed"

def lipschitz_estimate(spline, points):
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

    mu = np.repeat([r*h for h in H], 2)
    vel = gen_velocities(spline)
    mu *= vel

    return mu

def gen_velocities(spline):
    D = spline.derivative()
    vel = np.array([D(x) for x in spline.x])
    vel = np.repeat(vel, 2)[1:-1]
    vel = np.array([vel_map(-1*v if i % 2 == 0 else v) for i, v in enumerate(vel)])
    return vel

def vel_map(t):
    return (2/math.pi)*math.atan(t)+1

def build_P(spline, points, mu):
    def F(t):
        conditions = [(points[i-1][0] <= t) & (t <= points[i][0]) for i in range(1, len(points))]
        funcs = [np.vectorize(lambda x, i=i, s=spline: np.array([s(t)-mu[2*(i-1)]*(x-points[i-1][0]), s(t)+mu[2*(i-1)+1]*(x-points[i][0])]).max()) for i in range(1, len(points))]
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
    roots = roots[np.isreal(roots)]
    roots = roots[np.logical_and(bounds[0] <= roots+offset, roots+offset <= bounds[1])]

    mins = [bounds[0], bounds[1]]

    if roots.size != 0:
        m = min(roots, key=lambda x: np.polyval(c, x-offset)) + offset
        mins.append(m)
    return mins

def minimize_P(spline, points, mu):
    P = build_P(spline, points, mu)

    mins = list()
    for i in range(1, len(spline.x)):

        if mu[2*(i-1)] != 0 and mu[2*(i-1)+1] != 0:
            x_intersect = (mu[2*(i-1)]*points[i-1][0]+mu[2*(i-1)+1]*points[i][0])/(mu[2*(i-1)] + mu[2*(i-1)+1])
        else:
            x_intersect = (points[i-1][0] + points[i][0])/2

        int1 = (points[i-1][0], x_intersect)
        int2 = (x_intersect, points[i][0])

        c1 = convert_coefs(np.array([0, 0, -mu[2*(i-1)], 0]), points[i-1][0], points[i-1][0])
        c2 = convert_coefs(np.array([0, 0, mu[2*(i-1)+1], 0]), points[i][0], points[i-1][0])

        c1 = spline.c[:,i-1] + c1
        c2 = spline.c[:,i-1] + c2

        s1_mins = minimize_cubic_piece(c1, points[i-1][0], int1)
        s2_mins = minimize_cubic_piece(c2, points[i-1][0], int2)

        mins.extend(s1_mins + s2_mins)

    arg = min(mins, key=lambda x: P(x))
    return arg


def minimize(f, bounds, count_limit=None, save_file=None):
    eps = 10E-4 * (bounds[1] - bounds[0])  # Точность
    points = [(bounds[0], f(bounds[0])), (bounds[1], f(bounds[1]))]  # Точки на которых происходят вычисления
    diff = bounds[1] - bounds[0]  # длина отрезка
    counter = 2  # кол-во вычислений функции f

    # Пока разность между сгенер. точками x не меньше эпсилона
    while True:
        x, y = zip(*points)  # разбиваем на 2 массива, x и y
        spline = CubicSpline(x, y, bc_type='clamped')  # вычисляем сплайн по точкам

        mu = lipschitz_estimate(spline, points)
        arg = minimize_P(spline, points, mu)

        x0 = arg
        y0 = f(arg)
        counter += 1

        diff = min([math.fabs(arg-p[0]) for p in points]) # находим точность

        if diff < eps:
            break
        if count_limit != None:
            if counter == count_limit:
                break

        points.append((arg, f(arg)))  # добавляем новую точку
        points.sort(key=lambda x: x[0])  # сортируем точки

    if save_file is not None:
        P = build_P(spline, points, mu)
        xs = np.arange(bounds[0], bounds[1], 0.0001)
        fig, ax = plt.subplots()
        ax.plot(x, y, 'o')
        ax.plot(x0, y0, 'or')
        ax.plot(xs, spline(xs), 'blue', label='Интерполянт (m)')
        ax.plot(xs, np.vectorize(P)(xs), 'red', label='Критерий (P)')
        ax.plot(xs, f(xs), 'green', label='Целевая функция (f)')
        ax.legend(loc='best', ncol=2)

        dir = os.path.dirname(save_file)
        if not os.path.exists(dir):
            os.mkdir(dir)

        fig.savefig(save_file)
        plt.close(fig)

    return Result(points, counter, x0, y0)
