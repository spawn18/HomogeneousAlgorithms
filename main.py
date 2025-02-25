import sergeev, mishin_local, mishin_local_grad, mishin_local_accel_grad
import statistics
import functions

func_pick = functions.funcs


#dir_names = ['sergeev', ]
#statistics.create_dir_tree(names)

results = list()
results.append(sergeev.minimize(func_pick))
results.append(mishin_local.minimize(func_pick))
results.append(mishin_local_grad.minimize(func_pick))
results.append(mishin_local_accel_grad.minimize(func_pick))

names = ['sergeev', 'mishin_local', 'mishin_local_grad_best', 'mishin_local_accel_grad_best']
statistics.write_comparison(names, results, func_pick)







