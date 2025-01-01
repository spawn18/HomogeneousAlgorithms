import math
import numpy as np
import functions
from scipy.interpolate import CubicSpline, PPoly
import matplotlib.pyplot as plt

# Оценить константу липшица
def lipschitz_estimate_f(points):
    max = 0
    for i in range(0, len(points)):
        for j in range(0, len(points)):
            if i != j:
                L = math.fabs(points[i][1]-points[j][1])/math.fabs(points[i][0]-points[j][0])
                if L > max:
                    max = L
    return max

def build_p(spline, K):
    coefs = np.repeat(spline.c, 2, axis=1)

    for i in range(0, len(points) - 1, 2):
        pass
        #coefs[2][i] -= 2*K
        #coefs[2][i+1] += 2*K

    print(coefs)
    xmid = [(x[i] + x[i + 1]) / 2 for i in range(0, len(x) - 1)]
    xfunc = list(spline.x) + xmid
    xfunc.sort()

    return PPoly(coefs, xfunc, extrapolate=False)

# Поиск минимума P = m(x) - Ks(x)
def find_min_p(spline, K):
    P = build_p(spline, K)
    roots = P.derivative().roots()

    l = list()
    for i in range(len(roots)):
        if not math.isnan(roots[i]):
            l.append(roots[i])
        else:
            j = P.x.index(roots[i-1])
            xm = (roots[i-1]+P.x[j+1])/2
            l.append(xm)

    min_x = min(P.x.tolist()+l, key=lambda t: P(t))
    return min_x

# Оценка константы Липшица у интерполянта
def lipschitz_estimate_m(spline):
    D = spline.derivative()
    roots = D.roots()
    roots = roots[np.isfinite(roots)]
    l = list()
    for i in range(len(roots)):
        if not math.isnan(roots[i]):
            l.append(roots[i])

    return max(l+list(x), key=lambda t: math.fabs(D(t)))

results = []

# Вычисление минимума каждой из функций
for i, f in enumerate(functions.funcs[:1]):
    eps = 10E-4 * (f.b - f.a) # Точность
    points = [(f.a, f.eval(f.a)),  (f.b, f.eval(f.b))] # Точки на которых происходят вычисления
    diff = f.b-f.a # длина отрезка
    counter = 2 # кол-во вычислений функции f

    # Пока разность между сгенер. точками x не меньше эпсилона
    while diff >= eps:
        points.sort(key=lambda x: x[0]) # сортируем точки
        x, y = zip(*points) # разбиваем на 2 массива, x и y

        L_f = lipschitz_estimate_f(points) # аппроксимируем константу липшица кусочно-линейно
        spline = CubicSpline(x, y, bc_type='clamped') # вычисляем сплайн по точкам
        L_m = lipschitz_estimate_m(spline) # аппроксимируем константу липшица у сплайна
        K = (L_f + L_m) + 1 # Считаем К умнож. на множитель

        arg = find_min_p(spline, K) # находим минимум P

        diff = min([math.fabs(p[0] - arg) for p in points]) # находим точность
        points.append((arg, f.eval(arg))) # добавляем новую точку
        counter += 1 # увеличиваем счетчик


    results.append((i+1, counter, arg, f.eval(arg), math.fabs(f.eval(arg)-f.min_y))) # запись о результате

P = build_p(spline, K)
xs = np.arange(f.a, f.b, 0.1)
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(x,y,'o')
ax.plot(xs, spline(xs), label='m')
ax.plot(xs, f.eval(xs), label='f')
ax.plot(xs, P(xs), label='p')
ax.set_xlim(f.a, f.b)
ax.legend(loc='lower left', ncol=2)
fig.savefig('test.png')

print("Результаты: ")
for r in results:
    print("Функция: {} Кол-во: {} x: {} y: {} Абсолютная погрешность y: {:f}".format(r[0],r[1],r[2],r[3],r[4]))



