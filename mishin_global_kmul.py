import math
import statistics

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
import os
from result import Result

ALGO_NAME = "pochechueva"
delta = 10E-3
MUL = 1.4

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
    def P(t, s=spline):
        t = np.atleast_1d(t)
        return s(t) - K*np.min([np.fabs(t-x) for x in spline.x], axis=0)
    return P

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

        c1 = convert_coefs(np.array([0, 0, -2*K, 0]), points[i-1][0], points[i-1][0])
        c2 = convert_coefs(np.array([0, 0, 2*K, 0]), points[i][0], points[i-1][0])

        c1 = spline.c[:,i-1] + c1
        c2 = spline.c[:,i-1] + c2

        s1_mins = minimize_cubic_piece(c1, points[i - 1][0], int1)
        s2_mins = minimize_cubic_piece(c2, points[i - 1][0], int2)

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

def minimize(funcs, count_limit=None, save_final=True):
    results = list()

    for i, f in enumerate(funcs):
        q = 0
        eps = 10E-4 * (f.bounds[1] - f.bounds[0])  # Точность
        points = [(f.bounds[0], f.eval(f.bounds[0])), (f.bounds[1], f.eval(f.bounds[1]))]  # Точки на которых происходят вычисления
        diff = f.bounds[1] - f.bounds[0]  # длина отрезка
        counter = 2  # кол-во вычислений функции f

        # Пока разность между сгенер. точками x не меньше эпсилона
        while True:
            x, y = zip(*points)  # разбиваем на 2 массива, x и y
            spline = CubicSpline(x, y, bc_type='clamped')  # вычисляем сплайн по точкам

            L_f = lipschitz_estimate_f(points)  # аппроксимируем константу липшица кусочно-линейно
            L_m = lipschitz_estimate_m(spline)  # аппроксимируем константу липшица у сплайна
            K = math.pow(MUL, q)*(L_m+L_f+delta) # Считаем К умнож. на множитель

            arg = minimize_P(spline, points, 2*K)

            x0 = arg
            y0 = f.eval(arg)
            counter += 1

            diff = min([math.fabs(arg-p[0]) for p in points]) # находим точность

            if diff < eps:
                q += 1
                break
            if count_limit != None:
                if counter == count_limit:
                    break

            points.append((arg, f.eval(arg)))  # добавляем новую точку
            points.sort(key=lambda x: x[0])  # сортируем точки

        if save_final:
            P = build_P(spline, 2*K)
            xs = np.arange(f.bounds[0], f.bounds[1], 0.0001)
            plt.plot(x, y, 'o', label='Точки испытаний')
            plt.plot(x0, y0, 'or', label='Точка следующего испытания')
            plt.plot(xs, spline(xs), 'blue', label='Интерполянт (m)')
            plt.plot(xs, P(xs), 'red', label='Критерий (P)')
            plt.plot(xs, f.eval(xs), 'green', label='Целевая функция (f)')
            plt.legend(loc='best', ncol=2)
            plt.savefig(os.path.join(statistics.algo_path(ALGO_NAME, i + 1), 'final'))
            plt.close()

        results.append(Result(points, counter, x0, y0))
    return results
