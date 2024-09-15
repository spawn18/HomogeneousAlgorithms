import math
import functions
from scipy.interpolate import CubicSpline

def f(x):
    math.log(2*x)*math.log(3*x)-1

a = functions.f1.a
b = functions.f1.b
points = [(a, functions.f1.eval(a)), (b,functions.f1.eval(b))]

def lipschitz_estimate_f(points):
    max = 0
    for i in range(0, len(points)):
        for j in range(0, len(points)):
            if i != j:
                L = math.fabs(points[i][1]-points[j][1])/math.fabs(points[i][0]-points[j][0])
                if L > max:
                    max = L
    return max

while True:
    points.sort(key=lambda x: x[0])
    L_f = lipschitz_estimate_f(points)
    x, y = zip(*points)

    spline = CubicSpline(x, y)



