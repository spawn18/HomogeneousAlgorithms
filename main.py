import math
import numpy as np
import functions
import matplotlib.pyplot as plt
import pochechueva
import algo
import sergeev
import statistics


# Вычисление минимума каждой из функций
def print_result(i, r, min_y):
    print("Функция: {} Кол-во: {} x0: {} y0: {} y: {}".format(i, r.count, r.x0, r.y0, min_y))

def save_result(i, r):
    x,y = zip(*r.points)
    x = list(x)
    y = list(y)

    xs = np.arange(f.a, f.b, 0.0001)
    fig, ax = plt.subplots()
    ax.plot(x, y, 'o')
    ax.plot(xs, np.vectorize(r.F)(xs), label='Миноранта (F)')
    ax.plot(xs, np.vectorize(r.f)(xs), label='Целевая функция (f)')
    ax.legend(loc='best', ncol=2)
    fig.savefig('f'+str(i))
    fig.clf()

algorithms = [
    {
        "name": "Pochechueva",
        "function": pochechueva.minimize,
        "count": []
    },
    {
        "name": "sergeev",
        "function": sergeev.minimize,
        "count": []
    }
]

for a in algorithms:
    for i, f in enumerate(functions.funcs):
        r = a["function"](f.eval, (f.a, f.b), None, False)
        a["count"].append(r.count)
        print_result(i+1, r, f.min_y)

statistics.plot_comparison(algorithms)







