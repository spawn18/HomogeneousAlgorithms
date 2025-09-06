import csv
import math
import os

import numpy as np

FUNCS_DIR = "func"
COMPARISON_DIR = "comp"
ITERS_DIR = "iters"

SAVE = True

def iter_path(algo_name, i):
    return os.path.join(FUNCS_DIR+str(i), algo_name, ITERS_DIR)

def algo_path(algo_name, i):
    return os.path.join(FUNCS_DIR+str(i), algo_name)

def check_convergence(x_mins, xks, eps):
    for x in x_mins:
        for xk in xks:
            if math.fabs(xk - x) < eps:
                return True
    return False

def write_comparison(names, results, functions):
    m = len(functions)

    with open('comparison.csv', 'w') as file:
        wr = csv.writer(file, quoting=csv.QUOTE_ALL)

        wr.writerow(['']+names)
        for i in range(0, m):
            wr.writerow([str(i+1)]+[str(res[i].count) if res[i].success else str(res[i].count)+'*' for res in results])
        wr.writerow(['Среднее']+[str(np.average([r.count for r in res])) for res in results])

def create_dir_tree(algo_names):
    for i in range(0, 20):
        func_dir = FUNCS_DIR + str(i+1)
        if not os.path.exists(func_dir):
            os.mkdir(func_dir)