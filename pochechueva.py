import math
import numpy as np
from scipy.interpolate import CubicSpline, PPoly
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
    c = np.repeat(spline.c, 2, axis=1)
    r = np.repeat(spline.c, 2, axis=1)
    for i in range(0, c.shape[1], 2):
        mid = (spline.x[i // 2] + spline.x[i // 2 + 1]) / 2
        off = mid-spline.x[i//2]
        c[1][i+1] = 3*r[0][i]*off + r[1][i]
        c[2][i+1] = 3*r[0][i]*off**2 + 2*r[1][i]*off + r[2][i]
        c[3][i+1] = r[0][i]*off**3 + r[1][i]*off**2 + r[2][i]*off + r[3][i]

    for i in range(0, c.shape[1], 2):
        mid = (spline.x[i // 2] + spline.x[i // 2 + 1]) / 2
        acc = mid-spline.x[i//2+1]

        c[2][i] -= 2 * K
        c[2][i + 1] += 2*K
        c[3][i + 1] += 2*K*acc


    x_mid = [(spline.x[i] + spline.x[i + 1]) / 2 for i in range(len(spline.x) - 1)]
    xp = x_mid + spline.x.tolist()
    xp.sort()

    return PPoly(c, xp, extrapolate=False)


def minimize_p(P):
    roots = P.derivative().roots(discontinuity=False)
    roots = roots[np.isfinite(roots)].tolist()

    """
    if roots.size > 0:
        if roots.size > 1:
            i = 0
            while i < len(roots):
                if i+1 != len(roots):
                    if math.isnan(roots[i+1]):
                        j, = np.where(P.x == roots[i])[0]
                        l.append((roots[i]+P.x[j+1])/2)
                        i += 1
                else:
                    l.append(roots[i])
                i += 1
        else:
            l.append(roots[0])
    """

    return min(roots+P.x.tolist(), key=lambda t: P(t))

# Оценка константы Липшица у интерполянта
def lipschitz_estimate_m(spline):
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

        L_f = lipschitz_estimate_f(points)  # аппроксимируем константу липшица кусочно-линейно
        spline = CubicSpline(x, y, bc_type='clamped')  # вычисляем сплайн по точкам
        L_m = lipschitz_estimate_m(spline)  # аппроксимируем константу липшица у сплайна
        K = L_m+L_f # Считаем К умнож. на множитель

        P = build_P(spline, K)
        arg = minimize_p(P)  # находим минимум P
        diff = min([math.fabs(p - arg) for p in x])  # находим точность

        points.append((arg, f(arg)))  # добавляем новую точку
        points.sort(key=lambda x: x[0])  # сортируем точки
        counter += 1  # увеличиваем счетчик

    x0 = arg
    y0 = f(arg)
    error = math.fabs(f(arg) - min_y)

    return Result(x0=x0, y0=y0, bounds=bounds, points=points, count=counter, diff=diff, error=error, f=f, P=P)
