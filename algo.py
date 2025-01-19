import math
import numpy as np
from scipy.interpolate import CubicSpline, PPoly
from scipy.optimize import direct
from Result import Result

def holders_estimate_f(points, k):
    max = 0
    for i in range(0, len(points)):
        for j in range(0, len(points)):
            if i != j:
                L = math.fabs(points[i][1]-points[j][1])/math.fabs(points[i][0]-points[j][0])**k
                if L > max:
                    max = L
    return max

def build_P(spline, K):
    def P(t):
        D = spline.derivative()
        return spline(t) - 2*K*min([math.fabs((map_speed(D(s)) if D(s) < 0 else 1)*(t-s)) for s in spline.x])
    return P

def map_speed(t):
    return (2/math.pi)*math.atan(math.fabs(t))+1

# Оценка константы Липшица у интерполянта
def lipschitz_estimate_m(spline):
    D = spline.derivative()
    roots = D.derivative().roots(discontinuity=False)
    roots = roots[np.isfinite(roots)].tolist()
    xspline = spline.x.tolist()
    return max(map(lambda t: math.fabs(D(t)), roots+xspline))

def estimate_spline_velocity(spline):
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
    while diff >= eps:
        x, y = zip(*points)  # разбиваем на 2 массива, x и y

        L_f = holders_estimate_f(points, 1)  # аппроксимируем константу липшица кусочно-линейно
        spline = CubicSpline(x, y, bc_type='clamped')  # вычисляем сплайн по точкам
        L_m = lipschitz_estimate_m(spline)  # аппроксимируем константу липшица у сплайна
        K = L_m+L_f # Считаем К умнож. на множитель

        if K == 0:
            K = 1

        P = build_P(spline, K)
        res = direct(P, [bounds], locally_biased=False)
        arg = res.x[0] # находим минимум P

        diff = min([math.fabs(p - arg) for p in x])  # находим точность

        points.append((arg, f(arg)))  # добавляем новую точку
        points.sort(key=lambda x: x[0])  # сортируем точки
        counter += 1  # увеличиваем счетчик

        if count_limit != None:
            if counter == count_limit:
                break

    x0 = arg
    y0 = f(arg)

    return Result(name="Mine", x0=x0, y0=y0, bounds=bounds, points=points, count=counter, diff=diff, f=f, P=P, m=spline, min_y=min_y)
