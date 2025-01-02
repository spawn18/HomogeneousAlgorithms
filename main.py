import math
import numpy as np
import functions
from scipy.interpolate import CubicSpline, PPoly
from scipy.optimize import minimize, Bounds, direct
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

def build_P(spline, K):
    c = np.repeat(spline.c, 2, axis=1)
    r = np.repeat(spline.c, 2, axis=1)
    for i in range(0, c.shape[1], 2):
        mid = (spline.x[i//2] + spline.x[i//2+1])/2
        off = mid-spline.x[i//2]
        acc = mid-spline.x[i//2+1]

        c[2][i] -= 2*K
        c[1][i+1] = 3*r[0][i]*off + r[1][i]
        c[2][i+1] = 3*r[0][i]*off**2 + 2*r[1][i]*off + r[2][i] + 2*K
        c[3][i+1] = r[0][i]*off**3 + r[1][i]*off**2 + r[2][i]*off + r[3][i] + 2*K*acc

    x_mid = [(spline.x[i] + spline.x[i + 1]) / 2 for i in range(len(spline.x) - 1)]
    xp = x_mid + spline.x.tolist()
    xp.sort()

    return PPoly(c, xp, extrapolate=False), x_mid


def find_min_p_2(spline, K):
    P, x_mid = build_P(spline, K)
    roots = P.derivative().roots(discontinuity=False)
    l = list()

    if roots.size > 0:
        if roots.size > 1:
            for i in range(len(roots)-1):
                if math.isnan(roots[i+1]):
                    j, = np.where(P.x == roots[i])[0]
                    l.append((roots[i]+P.x[j+1])/2)
                    i += 1
                elif not math.isnan(roots[i]):
                    l.append(roots[i])
        else:
            l.append(roots[0])

    return min(l+x_mid, key=lambda t: P(t))

# Поиск минимума P = m(x) - Ks(x)
def find_min_p(spline, K):

    def P(t):
        return spline(t) - 2*K*min([math.fabs(t - x_) for x_ in x])
    res = direct(P, bounds=Bounds(x[0], x[-1]), maxfun=100000, maxiter=100000, vol_tol=1e-5)
    if not res.success:
        print(res.message)
        return None
    else:
        return res.x[0]

# Оценка константы Липшица у интерполянта
def lipschitz_estimate_m(spline):
    D = spline.derivative()
    roots = D.derivative().roots(discontinuity=False)
    l = roots[np.isfinite(roots)].tolist()
    xspline = spline.x.tolist()
    return max(map(lambda t: math.fabs(D(t)), l+xspline))

results = []

# Вычисление минимума каждой из функций
for i, f in enumerate(functions.funcs):
    eps = 10E-4 * (f.b - f.a) # Точность

    points = [(f.a, f.eval(f.a)),  (f.b, f.eval(f.b))] # Точки на которых происходят вычисления
    diff = f.b-f.a # длина отрезка
    counter = 2 # кол-во вычислений функции f

    # Пока разность между сгенер. точками x не меньше эпсилона
    while diff >= eps:
        x, y = zip(*points) # разбиваем на 2 массива, x и y

        L_f = lipschitz_estimate_f(points) # аппроксимируем константу липшица кусочно-линейно
        spline = CubicSpline(x, y, bc_type='clamped') # вычисляем сплайн по точкам
        L_m = lipschitz_estimate_m(spline) # аппроксимируем константу липшица у сплайна
        K = 1 # Считаем К умнож. на множитель

        arg = find_min_p_2(spline, K) # находим минимум P
        diff = min([math.fabs(p[0] - arg) for p in points]) # находим точность

        points.append((arg, f.eval(arg))) # добавляем новую точку
        points.sort(key=lambda x: x[0]) # сортируем точки
        counter += 1 # увеличиваем счетчик

    xs = np.arange(f.a, f.b, 0.001)
    P, _ = build_P(spline, K)
    fig, ax = plt.subplots()
    ax.plot(x, y, 'o')
    ax.plot(xs, P(xs), label='Критерий (P)')
    ax.plot(xs, np.vectorize(f.eval)(xs), label='Целевая функция (f)')
    #ax.plot(xs, spline(xs), label='Кубический сплайн (m)')
    ax.set_xlim(f.a, f.b)
    ax.legend(loc='upper left', ncol=2)
    fig.savefig('f'+str(i+1))

    results.append((i+1, counter, arg, f.eval(arg), math.fabs(f.eval(arg)-f.min_y))) # запись о результате

print("Результаты: ")
for r in results:
    print("Функция: {} Кол-во: {} x: {} y: {} Абсолютная погрешность y: {:f}".format(r[0],r[1],r[2],r[3],r[4]))



