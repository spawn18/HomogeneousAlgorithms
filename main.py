import pochechueva, sergeev, mishin_local, mishin_local_speed
import statistics
import functions

algorithms = [
    {
        "name": "pochechueva",
        "function": pochechueva.minimize,
        "results": []
    },
    {
        "name": "sergeev",
        "function": sergeev.minimize,
        "results": []
    },
    {
        "name": "mishin_local",
        "function": mishin_local.minimize,
        "results": []
    },
    {
        "name": "mishin_local_speed",
        "function": mishin_local_speed.minimize,
        "results": []
    }
]

statistics.create_dir_tree(functions.funcs, algorithms)

for alg in algorithms[0:1]:
    minimize = alg["function"]
    results = minimize(functions.funcs[0:1], 5, True)
    statistics.print_results(functions.funcs, results)

#statistics.plot_comparison(algorithms)







