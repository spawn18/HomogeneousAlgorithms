import math
import numpy as np
import functions
import matplotlib.pyplot as plt
import pochechueva
import algo

# Вычисление минимума каждой из функций
def print_result(i, r):
    print("Функция: {} Кол-во: {} x0: {} y0: {} y: {}".format(i, r.count, r.x0, r.y0, r.min_y))

def save_result(i, r):
    x,y = zip(*r.points)
    x = list(x)
    y = list(y)

    xs = np.arange(f.a, f.b, 0.0001)
    fig, ax = plt.subplots()
    ax.plot(x, y, 'o')
    ax.plot(xs, np.vectorize(r.P)(xs), label='Критерий (P)')
    ax.plot(xs, r.m(xs), label='Интерполянт (m)')
    ax.plot(xs, np.vectorize(r.f)(xs), label='Целевая функция (f)')
    ax.set_xlim(r.bounds[0], r.bounds[1])
    ax.legend(loc='best', ncol=2)
    fig.savefig('f'+str(i))

for i, f in enumerate(functions.funcs):
    r = pochechueva.minimize(f.eval, (f.a, f.b), f.min_y)
    print_result(i+1, r)
    save_result(i+1, r)




