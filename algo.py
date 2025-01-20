import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
import os
from Result import Result


def lipschitz_estimate_f(points):
    max = 0
    for i in range(0, len(points)):
        for j in range(0, len(points)):
            if i != j:
                L = math.fabs(points[i][1]-points[j][1])/math.fabs(points[i][0]-points[j][0])
                if L > max:
                    max = L
    return max

def build_P(spline, K):
    def P(t):
        return spline(t) - K*min([math.fabs((t-s)) for s in spline.x])
    return P

def convert_coefs(c, off1, off2):
    r = c
    off = off2 - off1
    c[1] = 3*r[0]*off + r[1]
    c[2] = 3*r[0]*off**2 + 2*r[1]*off + r[2]
    c[3] = r[0]*off**3 + r[1]*off**2 + r[2]*off + r[3]
    return c

def minimize_cubic_piece(c, bounds):
    roots = np.roots(np.polyder(c))
    roots = roots[np.isreal(roots)]

    mins = [bounds[0], bounds[1]]

    if roots.size != 0:
        m = min(roots, key=lambda x: np.polyval(c, x-bounds[0])) + bounds[0]
        if bounds[0] <= m <= bounds[1]:
            mins.append(m)
    return mins

def minimize_P(spline, points, K):
    P = build_P(spline, K)
    mins = list()
    for i in range(1, len(spline.x)):

        if K != 0:
            x_intersect = (K*points[i-1][0]+K*points[i][0])/(2*K)
        else:
            x_intersect = (points[i-1][0] + points[i][0])/2

        int1 = (points[i-1][0], x_intersect)
        int2 = (x_intersect, points[i][0])

        c1 = convert_coefs(np.array([0, 0, 2*K, 0]), points[i-1][0], points[i-1][0])
        c2 = convert_coefs(np.array([0, 0, -2*K, 0]), points[i][0], points[i-1][0])

        c1 = spline.c[:,i-1] + c1
        c2 = spline.c[:,i-1] + c2

        s1_mins = minimize_cubic_piece(c1, int1)
        s2_mins = minimize_cubic_piece(c2, int2)

        mins.extend(s1_mins + s2_mins)

    arg = min(mins, key=lambda x: P(x))
    return arg

# Оценка константы Липшица у интерполянта
def lipschitz_estimate_m(spline):
    D = spline.derivative()
    roots = D.derivative().roots(discontinuity=False)
    roots = roots[np.isfinite(roots)].tolist()
    xspline = spline.x.tolist()
    return max(map(lambda t: math.fabs(D(t)), roots+xspline))

def minimize(f, bounds, count_limit=None, save_file=None):
    eps = 10E-4 * (bounds[1] - bounds[0])  # Точность
    points = [(bounds[0], f(bounds[0])), (bounds[1], f(bounds[1]))]  # Точки на которых происходят вычисления
    diff = bounds[1] - bounds[0]  # длина отрезка
    counter = 2  # кол-во вычислений функции f

    # Пока разность между сгенер. точками x не меньше эпсилона
    while True:
        x, y = zip(*points)  # разбиваем на 2 массива, x и y
        spline = CubicSpline(x, y, bc_type='clamped')  # вычисляем сплайн по точкам

        L_f = lipschitz_estimate_f(points)  # аппроксимируем константу липшица кусочно-линейно
        L_m = lipschitz_estimate_m(spline)  # аппроксимируем константу липшица у сплайна
        K = L_m+L_f # Считаем К умнож. на множитель

        arg = minimize_P(spline, points, 2*K)

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
        P = build_P(spline, 2*K)
        xs = np.arange(bounds[0], bounds[1], 0.0001)
        fig, ax = plt.subplots()
        ax.plot(x, y, 'o')
        ax.plot(x0, y0, 'or')
        ax.plot(xs, np.vectorize(P)(xs), label='Критерий (P)')
        ax.plot(xs, f(xs), label='Целевая функция (f)')
        ax.legend(loc='best', ncol=2)

        dir = os.path.dirname(save_file)
        if not os.path.exists(dir):
            os.mkdir(dir)

        fig.savefig(save_file)
        plt.close(fig)

    return Result(points, counter, x0, y0)
