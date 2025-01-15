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

def poly_convert_basis(c, a, b):
    off = b-a
    new = c.copy()
    new[1] = 3 * c[0] * off + c[1]
    new[2] = 3 * c[0] * off ** 2 + 2 * c[1] * off + c[2]
    new[3] = c[0] * off ** 3 + c[1] * off ** 2 + c[2] * off + c[3]
    return new

def poly_add(c, vals, basis):
    r = c
    r[0] = c[0] + vals[0]
    r[1] = c[1] + 3*basis*vals[0]+vals[1]
    r[2] = c[2] + 3*vals[0]*basis**2 + 2*vals[1]*basis + vals[2]
    r[3] = c[3] + vals[0]*basis**3 + vals[1]*basis**2 + vals[2]*basis + vals[3]
    return r

def build_P(spline, K):
    def P(t):
        return spline(t) - 5*K*min([math.fabs(t-s) for s in spline.x])
    return P


    c = np.repeat(spline.c, 2, axis=1)
    r = np.repeat(spline.c, 2, axis=1)

    x_mid = [(spline.x[i] + spline.x[i + 1]) / 2 for i in range(len(spline.x) - 1)]
    xp = x_mid + spline.x.tolist()
    xp.sort()

    for i in range(0, c.shape[1], 2):
        c[:, i] = poly_convert_basis(c[:,i], xp[i], xp[i+1])

    #for i in range(0, c.shape[1], 2):
    #    c[:,i] = poly_add(c[:,i], [0, 0, 2*K, 0], xp[i])
    #    c[:,i+1] = poly_add(c[:,i], [0, 0, -2*K, 0], xp[i+1])

    return PPoly(c, xp, extrapolate=False)

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

def minimize(f, bounds, min_y):
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

        P = build_P(spline, K)
        res = direct(P, [bounds], locally_biased=False, vol_tol=10E-6, maxfun=10000)
        if res.success:
            arg = res.x[0] # находим минимум P
        else:
            print(res.message)

        diff = min([math.fabs(p - arg) for p in x])  # находим точность

        points.append((arg, f(arg)))  # добавляем новую точку
        points.sort(key=lambda x: x[0])  # сортируем точки
        counter += 1  # увеличиваем счетчик

    x0 = arg
    y0 = f(arg)
    error = math.fabs((f(arg) - min_y)/min_y)

    return Result(x0=x0, y0=y0, bounds=bounds, points=points, count=counter, diff=diff, error=error, f=f, P=P, m=spline, min_y=min_y)
