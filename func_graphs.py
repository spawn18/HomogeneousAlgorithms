import functions
import matplotlib.pyplot as plt
import numpy as np

for i,f in enumerate(functions.funcs):
    x = np.linspace(f.bounds[0], f.bounds[1], 500)

    plt.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)
    plt.plot(x, f.eval(x), color=(0,0,0))
    plt.title("Функция " + str(i+1))
    plt.savefig('graphs/f' + str(i+1) + '.png', dpi=300)
    plt.clf()

for i,f in enumerate(functions.sfuncs):
    x = np.linspace(f.bounds[0], f.bounds[1], 1000)

    plt.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)
    plt.plot(x, f.eval(x), color=(0, 0, 0))
    plt.title("Функция " + str(i + 1))
    plt.savefig('graphs/sf' + str(i + 1) + '.png', dpi=300)
    plt.clf()