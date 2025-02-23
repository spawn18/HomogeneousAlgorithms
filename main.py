import mishin_localcustom_speed
import mishin_speed
import pochechueva, sergeev, mishin_local, mishin_local_speed
import statistics
import functions

func_pick = functions.funcs

#names = ['l','s','ls1','ls2','ls3']
names = ['l','ls2','lcs']

results = list()
results.append(mishin_local.minimize(func_pick))
#results.append(mishin_speed.minimize(func_pick))
results.append(mishin_local_speed.minimize_grad1(func_pick))
#results.append(mishin_local_speed.minimize_grad2(func_pick))
#results.append(mishin_local_speed.minimize_grad3(func_pick))
results.append(mishin_localcustom_speed.minimize(func_pick))

statistics.write_comparison(names, results, func_pick)







