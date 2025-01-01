import math
import numpy as np
import functions
from scipy.interpolate import CubicSpline
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

def find_min_p_2(spline, K):
    coefs = spline.derivative().c
    
    for i,c in enumerate
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
    DD = D.derivative()
    roots = D.roots()
    l = roots[np.isfinite(roots)].tolist()
    xspline = spline.x.tolist()
    return max(l+xspline, key=lambda t: math.fabs(DD(t)))

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
        #print("Coefs: {}".format(spline.c))
        L_m = lipschitz_estimate_m(spline) # аппроксимируем константу липшица у сплайна
        K = (L_f + L_m) + 1 # Считаем К умнож. на множитель
        #print("K:{}".format(K))

        arg = find_min_p(spline, K) # находим минимум P
        diff = min([math.fabs(p[0] - arg) for p in points]) # находим точность
        points.append((arg, f.eval(arg))) # добавляем новую точку
        points.sort(key=lambda x: x[0]) # сортируем точки
        #print("Points: {}".format(points))
        counter += 1 # увеличиваем счетчик


    results.append((i+1, counter, arg, f.eval(arg), math.fabs(f.eval(arg)-f.min_y))) # запись о результате


print("Результаты: ")
for r in results:
    print("Функция: {} Кол-во: {} x: {} y: {} Абсолютная погрешность y: {:f}".format(r[0],r[1],r[2],r[3],r[4]))



