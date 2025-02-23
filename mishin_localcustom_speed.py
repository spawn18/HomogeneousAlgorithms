import math
import statistics

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from result import Result

ALGO_NAME = "mishin_localcustom_speed"

# arctan function
def grad(x):
    if x < 0:
        return - 0.5 / (1 + math.exp(0.5*x)) + 1.25
    return 0.5 / (1 + math.exp(-0.5*x))+0.75

def lipschitz_estimate(spline, points):
    eps = 10E-6
    lamb_max = max([math.fabs(points[i][1]-points[i-1][1])/(points[i][0]-points[i-1][0]) for i in range(1, len(points))])
    x_max = max([points[i][0]-points[i-1][0] for i in range(1, len(points))])

    def build_list(i, n):
        if n == 1: return [i]
        else:
            if i == 1: return [i, i+1]
            elif i == n: return [i, i-1]
            else: return [i-1, i, i+1]

    n = len(points)

    H = list()
    for i in range(1, n):
        lamb = max([math.fabs(points[j][1]-points[j-1][1])/(points[j][0]-points[j-1][0]) for j in build_list(i, n-1)])
        alpha = (points[i][0]-points[i-1][0])/x_max
        h = alpha*lamb_max + (1-alpha)*lamb + eps
        H.append(h)

    mu = np.repeat([h for h in H], 2)
    vel = gen_velocities(spline)
    mu *= vel

    return mu

def gen_velocities(spline):
    D = spline.derivative()
    vel = np.array([D(x) for x in spline.x])
    vel = np.repeat(vel, 2)[1:-1]
    vel = np.array([grad(-1*v if i % 2 == 0 else v) for i, v in enumerate(vel)])
    return vel

def build_P(spline, points, mu):
    def F(t):
        conditions = [(points[i-1][0] <= t) & (t <= points[i][0]) for i in range(1, len(points))]
        funcs = [lambda x, i=j, s=spline: np.max([s(x)-mu[2*(i-1)]*(x-points[i-1][0]), s(x)+mu[2*(i-1)+1]*(x-points[i][0])], axis=0) for j in range(1, len(points))]
        return np.piecewise(t, conditions, funcs)
    return F

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

def minimize_P(spline, points, mu):
    P = build_P(spline, points, mu)

    mins = list()
    for i in range(1, len(spline.x)):

        if mu[2*(i-1)] != 0 and mu[2*(i-1)+1] != 0:
            x_intersect = (mu[2*(i-1)]*points[i-1][0]+mu[2*(i-1)+1]*points[i][0])/(mu[2*(i-1)] + mu[2*(i-1)+1])
        else:
            x_intersect = (points[i-1][0] + points[i][0])/2

        int1 = (points[i-1][0], x_intersect)
        int2 = (x_intersect, points[i][0])

        c1 = convert_coefs(np.array([0, 0, -mu[2*(i-1)], 0]), points[i-1][0], points[i-1][0])
        c2 = convert_coefs(np.array([0, 0, mu[2*(i-1)+1], 0]), points[i][0], points[i-1][0])

        c1 = spline.c[:,i-1] + c1
        c2 = spline.c[:,i-1] + c2

        s1_mins = minimize_cubic_piece(c1, points[i-1][0], int1)
        s2_mins = minimize_cubic_piece(c2, points[i-1][0], int2)

        mins.extend(s1_mins + s2_mins)

    arg = min(mins, key=lambda x: P(x))
    return arg


def minimize(funcs, count_limit=None):
    results = list()

    for i, f in enumerate(funcs):
        eps = 10E-4 * (f.bounds[1] - f.bounds[0])  # Точность
        points = [(f.bounds[0], f.eval(f.bounds[0])), (f.bounds[1], f.eval(f.bounds[1]))]  # Точки на которых происходят вычисления
        diff = f.bounds[1] - f.bounds[0]  # длина отрезка
        counter = 2  # кол-во вычислений функции f

        # Пока разность между сгенер. точками x не меньше эпсилона
        while True:
            x, y = zip(*points)  # разбиваем на 2 массива, x и y
            spline = CubicSpline(x, y, bc_type='clamped')  # вычисляем сплайн по точкам

            mu = lipschitz_estimate(spline, points)
            arg = minimize_P(spline, points, mu)

            x0 = arg
            y0 = f.eval(arg)
            counter += 1

            diff = min([math.fabs(arg-p[0]) for p in points]) # находим точность

            if diff < eps:
                break
            if count_limit != None:
                if counter == count_limit:
                    break

            points.append((arg, f.eval(arg)))  # добавляем новую точку
            points.sort(key=lambda x: x[0])  # сортируем точки

        """
        P = build_P(spline, points, mu)
        xs = np.arange(f.bounds[0], f.bounds[1], 0.0001)
        plt.plot(x, y, 'o', label='Точки испытаний')
        plt.plot(x0, y0, 'or', label='Точка следующего испытания')
        plt.plot(xs, spline(xs), 'blue', label='Интерполянт (m)')
        plt.plot(xs, P(xs), 'red', label='Критерий (P)')
        plt.plot(xs, f.eval(xs), 'green', label='Целевая функция (f)')
        plt.legend(loc='best', ncol=2)
        plt.savefig(os.path.join(statistics.algo_path(ALGO_NAME, i + 1), 'final'))
        plt.close()
        """

        success = statistics.check_convergence(f.min_x, x, 2*eps)
        results.append(Result(points, counter, x0, y0, f.min_y, success))
    return results