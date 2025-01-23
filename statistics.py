import os

import numpy as np
from matplotlib import pyplot as plt

FUNCS_DIR = "func"
COMPARISON_DIR = "comp"
ITERS_DIR = "iters"

def print_current_func(i, total):
    print("{}/{}".format(i, total))

def iter_path(algo_name, i):
    return os.path.join(FUNCS_DIR+str(i), algo_name, ITERS_DIR)

def algo_path(algo_name, i):
    return os.path.join(FUNCS_DIR+str(i), algo_name)

# Вычисление минимума каждой из функций
def print_results(funcs, results):
    for i,r in enumerate(results):
        print("Функция: {} Кол-во: {} x0: {} y0: {} y: {}".format(i+1, r.count, r.x0, r.y0, funcs[i].min_y))

def plot_comparison(algorithms):
    names = list()
    x_axis = np.arange(20)

    for i in range(0, len(algorithms)):
        names.append(algorithms[i]["name"])
        offset = 0.2*(i-len(algorithms)//2)
        plt.bar(x_axis+offset, algorithms[i]["count"], 0.2, align='edge', label=algorithms[i]["name"])

    plt.xticks(x_axis, [str(i) for i in range(1, 21)])
    plt.gca().yaxis.get_major_locator().set_params(integer=True)
    plt.gca().yaxis.grid()
    plt.xlabel("Функции")
    plt.ylabel("Кол-во вычислений")
    plt.title("Сравнение алгоритмов")
    plt.legend()
    plt.savefig("comparison")

def create_dir_tree(funcs, algorithms):
    for i in range(0, len(funcs)):
        func_dir = FUNCS_DIR + str(i+1)
        if not os.path.exists(func_dir):
            os.mkdir(func_dir)

        for alg in algorithms:
            alg_dir = os.path.join(func_dir, alg["name"])
            if not os.path.exists(alg_dir):
                os.mkdir(alg_dir)

            iter_dir = os.path.join(alg_dir, ITERS_DIR)
            if not os.path.exists(iter_dir):
                os.mkdir(iter_dir)