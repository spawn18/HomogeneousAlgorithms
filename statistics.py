import numpy as np
from matplotlib import pyplot as plt


def plot_comparison(algorithms):

    names = list()
    x_axis = np.arange(20)

    for i in range(0, len(algorithms)):
        names.append(algorithms[i]["name"])

        offset = 0.2*i - len(algorithms)//2
        plt.bar(x_axis+offset, algorithms[i]["count"], 0.4, label=algorithms[i]["name"])

    plt.xticks(x_axis, [str(i) for i in range(1, 21)])
    plt.xlabel("Функции")
    plt.ylabel("Кол-во вычислений")
    plt.title("Сравнение алгоритмов")
    plt.legend()
    plt.savefig("comparison")