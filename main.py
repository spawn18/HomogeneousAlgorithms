import math
import numpy as np
import functions
from scipy.interpolate import CubicSpline, PPoly
from scipy.optimize import direct, Bounds

# Оценить константу липшица методом Стронгина
# (найти все наклоны отрезков)
def lipschitz_estimate_f(points):
    max = 0
    for i in range(0, len(points)):
        for j in range(0, len(points)):
            if i != j:
                L = math.fabs(points[i][1]-points[j][1])/math.fabs(points[i][0]-points[j][0])
                if L > max:
                    max = L
    return max

def quad_powers(x):
    return [x**(2-i) for i in range(0, 3)]

def find_minimum_p(spline, K):
    mat = np.array(spline.c).T
    ref = mat
    mat = np.repeat(mat, 2, axis=0)

    for i in range(0, len(mat), 2):
        off = (points[i//2+1][0]+points[i//2][0])/2
        com = off-points[i//2][0]

        mat[i][2] -= 2*K
        mat[i+1][1] += 3*ref[i//2][0]*com
        mat[i+1][2] += 3*ref[i//2][0]*com**2+2*ref[i//2][1]*com+2*K
        mat[i+1][3] += ref[i//2][0]*com**3+ref[i//2][1]*com**2+ref[i//2][2]*com+2*K*(off-points[i//2+1][0])

    x, y = zip(*points)
    x_ = []
    for i in range(0, len(x)-1):
        x_.append(x[i])
        x_.append((x[i]+x[i+1])/2)
    x_.append(x[-1])

    poly = PPoly(-mat.T, x_, extrapolate=False)

    roots = poly.derivative().roots().flatten()
    x__ = np.unique(np.concatenate([np.array(x_), roots]))
    y_ = poly(x__)
    roots = zip(x__, y_)
    m = max(roots, key=lambda s: s[1])[0]
    return m

def lipschitz_estimate_m(spline):
    m = []
    mat = np.array(spline.derivative().c).T

    for i, coefs in enumerate(mat):
        xc = -coefs[1]/(2*coefs[0]) + x[i]
        v = []
        if x[i] <= xc <= x[i+1]:
            v.append(math.fabs(np.dot(mat[i], np.array(quad_powers(xc-x[i])))))

        v.append(math.fabs(np.dot(mat[i], np.array(quad_powers(0)))))
        v.append(math.fabs(np.dot(mat[i], np.array(quad_powers(x[i+1]-x[i])))))

        m.append(max(v))
    return max(m)

results = []

# Вычисление оптимума каждой из функций
for i, f in enumerate(functions.funcs):
    a, b, f = f.a, f.b, f.eval # Начальные точки и функция вычисления
    eps = 10E-4 * (b - a) # Эпсилон
    points = [(a, f(a)), (b, f(b))] # Точки на которых происходят вычисления

    diff = b-a # длина отрезка
    counter = 2 # кол-во вычислений функции f

    # Пока точность не меньше эпсилона
    while diff >= eps:
        points.sort(key=lambda x: x[0]) # сортируем точки
        x, y = zip(*points) # разбиваем на 2 массива, x и y

        L_f = lipschitz_estimate_f(points) # аппроксимируем константу липшица кусочно-линейно
        spline = CubicSpline(x, y, bc_type='clamped') # вычисляем сплайн по точкам
        L_m = lipschitz_estimate_m(spline) # аппроксимируем константу липшица у сплайна
        K = L_f + L_m + 100 # Считаем К
        arg = find_minimum_p(spline, K) # находим минимум P

        diff = min([math.fabs(p[0] - arg) for p in points])
        points.append((arg, f(arg))) # добавляем новую точку
        counter += 1 # увеличиваем счетчик

    results.append((i+1, counter, arg, f(arg), diff)) # запись о результате

print("Результаты: ")
for r in results:
    print("Функция: {} Кол-во: {} Минимум: {} Значение: {} Точность: {}".format(r[0],r[1],r[2],r[3],r[4]))



