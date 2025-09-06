import math
import statistics

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from result import Result

ALGO_NAME = "CubicSplineGrad"
KSI = 10E-6
R = 1.1

def lipschitz_estimate(points):
    lamb_max = max(
        [math.fabs(points[i][1] - points[i - 1][1]) / (points[i][0] - points[i - 1][0]) for i in range(1, len(points))])
    x_max = max([points[i][0] - points[i - 1][0] for i in range(1, len(points))])

    def build_list(i, n):
        if n == 1:
            return [i]
        else:
            if i == 1:
                return [i, i + 1]
            elif i == n:
                return [i, i - 1]
            else:
                return [i - 1, i, i + 1]

    n = len(points)

    H = list()
    for i in range(1, n):
        lamb = max([math.fabs(points[j][1] - points[j - 1][1]) / (points[j][0] - points[j - 1][0]) for j in
                    build_list(i, n - 1)])
        gamma = lamb_max * (points[i][0] - points[i - 1][0]) / x_max
        H.append(max([KSI, lamb, gamma]))

    mu = np.repeat([R * h for h in H], 2)
    return mu

def grad_boost(spline, points, mu, grad_smoother):
    D = spline.derivative()
    vel = np.array([D(x) for x in spline.x])
    vel = np.repeat(vel, 2)[1:-1]
    vel = np.array([grad_smoother(-1 * v if i % 2 == 0 else v) for i, v in enumerate(vel)])

    n = len(points)

    for i in range(1, n):
        k = (points[i][1] - points[i - 1][1]) / (points[i][0] - points[i - 1][0])

        mu1 = mu[2 * (i - 1)] * vel[2 * (i - 1)]
        mu2 = mu[2 * (i - 1) + 1] * vel[2 * (i - 1) + 1]

        if k >= 0:
            mu[2 * (i - 1)] = max([mu1, KSI])
            mu[2 * (i - 1) + 1] = max([mu2, k + KSI, KSI])
        else:
            mu[2 * (i - 1)] = max([mu1, -k + KSI, KSI])
            mu[2 * (i - 1) + 1] = max([mu2, KSI])

    return mu

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
    roots = roots[np.isreal(roots)] + offset
    roots = roots[np.logical_and(bounds[0] <= roots, roots <= bounds[1])].tolist()

    def eval(x):
        return np.polyval(c, x - offset)

    mins_x = list(bounds) + roots
    x0 = min(mins_x, key=eval)
    y0 = eval(x0)
    return (x0, y0)

def minimize_P(spline, points, mu):
    mins = list()
    for i in range(1, len(spline.x)):
        x_intersect = (mu[2*(i-1)]*points[i-1][0]+mu[2*(i-1)+1]*points[i][0])/(mu[2*(i-1)] + mu[2*(i-1)+1])
        int1 = (points[i-1][0], x_intersect)
        int2 = (x_intersect, points[i][0])

        c1 = convert_coefs(np.array([0, 0, -mu[2*(i-1)], 0]), points[i-1][0], points[i-1][0])
        c2 = convert_coefs(np.array([0, 0, mu[2*(i-1)+1], 0]), points[i][0], points[i-1][0])

        c1 = spline.c[:,i-1] + c1
        c2 = spline.c[:,i-1] + c2

        c1_min = minimize_cubic_piece(c1, points[i-1][0], int1)
        c2_min = minimize_cubic_piece(c2, points[i-1][0], int2)

        mins.append(min([c1_min, c2_min], key=lambda x: x[1]))

    arg = min(mins, key=lambda x: x[1])[0]
    return arg

def minimize(funcs, grad_smoother):
    results = list()

    for i, f in enumerate(funcs):
        eps = 10E-4 * (f.bounds[1] - f.bounds[0])
        points = [(f.bounds[0], f.eval(f.bounds[0])), (f.bounds[1], f.eval(f.bounds[1]))]
        counter = 2

        while True:
            x, y = zip(*points)
            spline = CubicSpline(x, y, bc_type='clamped')

            mu = lipschitz_estimate(points)
            mu = grad_boost(spline, points, mu, grad_smoother)
            arg = minimize_P(spline, points, mu)

            x0 = arg
            y0 = f.eval(arg)
            counter += 1

            diff = min([math.fabs(arg-p[0]) for p in points])

            if diff < eps:
                break

            points.append((arg, f.eval(arg)))
            points.sort(key=lambda x: x[0])

        if statistics.SAVE:
            P = build_P(spline, points, mu)
            xs = np.arange(f.bounds[0], f.bounds[1], 0.0001)
            plt.plot(x, y, 'o', label='Points')
            plt.plot(xs, spline(xs), 'blue', label='Interpolant')
            plt.plot(xs, P(xs), 'limegreen', label='Criteria')
            plt.plot(xs, f.eval(xs), 'black', label='Target function')
            plt.plot(x0, y0, 'xy', label='Minimum')
            plt.title("Evaluation count: " + str(counter))
            plt.legend(loc='best', ncol=2)
            plt.grid()
            plt.savefig(statistics.algo_path(ALGO_NAME, i + 1), dpi=300)
            plt.close()

        success = statistics.check_convergence(f.min_x, x, eps)
        results.append(Result(points, counter, x0, y0, f.min_y, success))

    return results