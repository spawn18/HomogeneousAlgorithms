import sergeev, mishin_local, mishin_local_grad, mishin_localnconvex_grad
import statistics
import functions

func_pick = functions.funcs

def grad1(x, a, b):
    if a == 0 or b == 0:
        return 1

    if -b * a <= x <= b * a:
        return (1 / a) * x + 1
    elif x < -b * a:
        return -b + 1
    else:
        return b + 1

def accel1(x, a, b):
    if a == 0 or b == 0:
        return 1

    if -b * a <= x <= b * a:
        return (1 / a) * x + 1
    elif x < -b * a:
        return -b + 1
    else:
        return b + 1

#dir_names = ['sergeev', ]
#statistics.create_dir_tree(names)

results = list()
results.append(sergeev.minimize(func_pick))
results.append(mishin_local.minimize(func_pick))
results.append(mishin_local_grad.minimize(func_pick, grad_smoother=lambda x: grad1(x, 0.25, 0.4)))
results.append(mishin_localnconvex_grad.minimize(func_pick, grad_smoother=lambda x: grad1(x, 0.25, 0.4), exponent=6.5))

names = ['sergeev', 'mishin_local', 'mishin_local_grad', 'mishin_local_grad_qconvex']
statistics.write_comparison(names, results, func_pick)







