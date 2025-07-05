import math
import numpy as np
from matplotlib import pyplot as plt
import statistics
from result import Result

ALGO_NAME = "NL"
R = 1.2

def lipschitz_estimate(points):
    ksi = 10E-6
    lamb_max = max([math.fabs(points[i][1]-points[i-1][1])/(points[i][0]-points[i-1][0]) for i in range(1, len(points))])
    x_max = max([points[i][0]-points[i-1][0] for i in range(1, len(points))])

    def build_list(i, n):
        if n == 1: return [i]
        else:
            if i == 1: return [i, i+1]
            elif i == n: return [i-1, i]
            else: return [i-1, i, i+1]

    n = len(points)

    H = list()
    for i in range(1, n):
        lamb = max([math.fabs(points[j][1]-points[j-1][1])/(points[j][0]-points[j-1][0]) for j in build_list(i, n-1)])
        gamma = lamb_max * ((points[i][0]-points[i-1][0]) / x_max)
        H.append(max([ksi, lamb, gamma]))

    mu = [R*h for h in H]
    return mu

def build_F(points, mu):
    def F(t):
        conditions = [(points[i-1][0] <= t) & (t < points[i][0]) for i in range(1, len(points))]
        funcs = [lambda x, i=j: np.max([points[i-1][1]-mu[i-1]*(x-points[i-1][0]), points[i][1]+mu[i-1]*(x-points[i][0])], axis=0) for j in range(1, len(points))]
        return np.piecewise(t, conditions, funcs)
    return F

def min_F(points, mu):
    x = [(points[i][0]+points[i-1][0])/2 - (points[i][1]-points[i-1][1])/(2*mu[i-1]) for i in range(1, len(points))]
    r = [(points[i][1]+points[i-1][1])/2 - mu[i-1]*(points[i][0]-points[i-1][0])/2 for i in range(1, len(points))]
    p = list(zip(x,r))
    m = min(p, key=lambda p: p[1])
    arg = m[0]
    t = p.index(m)+1
    return arg, t

def minimize(funcs):
    results = list()

    for i, f in enumerate(funcs):
        eps = 10E-4 * (f.bounds[1] - f.bounds[0])
        points = [(f.bounds[0], f.eval(f.bounds[0])), (f.bounds[1], f.eval(f.bounds[1]))]
        counter = 2

        while True:
            x, y = zip(*points)
            mu = lipschitz_estimate(points)
            arg, t = min_F(points, mu)

            x0 = min(points, key=lambda p: p[1])[0]
            y0 = min(points, key=lambda p: p[1])[1]

            diff = min([math.fabs(arg-p[0]) for p in points])
            if diff < eps:
                break

            points.append((arg, f.eval(arg)))
            points.sort(key=lambda x: x[0])
            counter += 1

        if statistics.SAVE:
            F = build_F(points, mu)
            xs = np.arange(f.bounds[0], f.bounds[1], 0.0001)
            plt.plot(x, y, 'o', label='Точки испытаний')
            plt.plot(xs, F(xs), 'limegreen', label='Критерий')
            plt.plot(xs, f.eval(xs), 'black', label='Целевая функция')
            plt.plot(x0, y0, 'xy', label='Минимум')
            plt.title("Кол-во испытаний: " + str(counter))
            plt.legend(loc='best', ncol=2)
            plt.grid()
            plt.savefig(statistics.algo_path(ALGO_NAME, i+1), dpi=300)
            plt.close()

        success = statistics.check_convergence(f.min_x, x, eps)
        results.append(Result(points, counter, x0, y0, f.min_y, success))

    return results
