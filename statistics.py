import csv
import math
import os

import numpy as np
from matplotlib import pyplot as plt

FUNCS_DIR = "func"
COMPARISON_DIR = "comp"
ITERS_DIR = "iters"

def iter_path(algo_name, i):
    return os.path.join(FUNCS_DIR+str(i), algo_name, ITERS_DIR)

def algo_path(algo_name, i):
    return os.path.join(FUNCS_DIR+str(i), algo_name)

# Вычисление минимума каждой из функций
def print_results(funcs, results):
    for i,r in enumerate(results):
        print("Функция: {} Кол-во: {} x0: {} y0: {} y: {}".format(i+1, r.count, r.x0, r.y0, funcs[i].min_y))


def check_convergence(x_mins, xks, eps):
    for x in x_mins:
        for xk in xks:
            if math.fabs(xk - x) < eps:
                return True
    return False

def write_comparison(names, results, functions):
    n = len(results)
    m = len(functions)

    with open('comparison.csv', 'w') as file:
        wr = csv.writer(file, quoting=csv.QUOTE_ALL)

        wr.writerow(['']+names)
        for i in range(0, m):
            wr.writerow([str(i+1)]+[str(res[i].count) if res[i].success else '-' for res in results])
        wr.writerow(['Среднее']+[str(np.average([r.count for r in res])) for res in results])


def create_dir_tree(funcs, algorithms):
    for i in range(0, len(funcs)):
        func_dir = FUNCS_DIR + str(i+1)
        if not os.path.exists(func_dir):
            os.mkdir(func_dir)

        for alg in algorithms:
            alg_dir = os.path.join(func_dir, alg["name"])
            if not os.path.exists(alg_dir):
                os.mkdir(alg_dir)