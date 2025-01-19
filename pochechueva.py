import math
import numpy as np
from scipy.interpolate import CubicSpline, PPoly
from scipy.optimize import direct

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
        return spline(t) - 2*K*min([math.fabs((t-s)) for s in spline.x])
    return P

def convert_coefs(c, offset):
    return
def minimize_cubic_piece(c, bounds):
    roots = np.roots(np.polyder(c))
    return min(roots, key=lambda x: np.polyval(c, x))+bounds[0]

def minimize_P(spline, points, K):
    mins = points
    for i in range(1, len(spline.x)):
        x_intersect = (points[i-1][1]-points[i][1]+K*points[i-1][0]+K*points[i][0])/(2*K)
        y_intersect = spline(x_intersect)
        p_intersect = (x_intersect, y_intersect)

        int1 = (points[i-1][0], x_intersect)
        int2 = (x_intersect, points[i][0][0])

        c1 = convert_coefs(np.array([0, 0, 2*K, 0]), points[i-1][0], points[i-1][0])
        c2 = convert_coefs(np.array([0, 0, -2*K, 0]), points[i][0], points[i-1][0])

        c1 = spline.c[:,i-1] + c1
        c2 = spline.c[:,i-1] + c2

        s1_mins = minimize_cubic_piece(c1, int1)
        s2_mins = minimize_cubic_piece(c2, int2)

        mins.extend(s1_mins, s2_mins, p_intersect)

    arg = min(mins, key=lambda x: x[1])[0]
    return arg

# Оценка константы Липшица у интерполянта
def lipschitz_estimate_m(spline):
    D = spline.derivative()
    roots = D.derivative().roots(discontinuity=False)
    roots = roots[np.isfinite(roots)].tolist()
    xspline = spline.x.tolist()
    return max(map(lambda t: math.fabs(D(t)), roots+xspline))

def minimize(f, bounds, min_y, count_limit=None):
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
        diff = min([math.fabs(arg-p[0]) for p in points]) # находим точность

        if diff < eps:
            x0 = arg
            y0 = f(arg)
            break
        else:
            points.append((arg, f(arg)))  # добавляем новую точку
            points.sort(key=lambda x: x[0])  # сортируем точки
            counter += 1  # увеличиваем счетчик

            if count_limit != None:
                if counter == count_limit:
                    break

    return Result(points, counter, x0, y0)
