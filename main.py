import math

import NL
import CubicSpline
import CubicSplineGrad
import GradNL
import QradNL

import statistics
import functions

func_pick = functions.funcs

def smoother(x, a, b):
    if x < 0:
        return (2/math.pi) * math.atan((a-1)*x/b) + 1
    else:
        return (a-1)*(2/math.pi) * math.atan(x/b) + 1


smoother1=lambda x: smoother(x, 1.8, 7.75)
smoother2=lambda x: smoother(x, 1.3, 6.4)
smoother3=lambda x: smoother(x, 1.65, 3.7)

algo_names = ['NL', 'CubicSpline', 'CubicSplineGrad', 'GradNL', 'QradNL']
statistics.create_dir_tree(algo_names)

results = list()

results.append(NL.minimize(func_pick))
results.append(CubicSpline.minimize(func_pick))
results.append(CubicSplineGrad.minimize(func_pick, grad_smoother=smoother1))
results.append(GradNL.minimize(func_pick, grad_smoother=smoother2))
results.append(QradNL.minimize(func_pick, grad_smoother=smoother3))
statistics.write_comparison(algo_names, results, func_pick)






