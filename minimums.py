from functions import *

def find_minimum_brute_force(f, k):
    xm = f.a
    ym = f.eval(xm)
    for i in range(0, k):
        x = f.a + (i/k)*(f.b-f.a)
        y = f.eval(x)
        if y < ym:
            xm = x
            ym = y
    return (xm, ym)

print("Минимумы функций")
for i, f in enumerate(funcs):
    t = find_minimum_brute_force(f, 10000000)
    print("Функция #{}: x = {}\t y = {}".format(i+1, t[0], t[1]))
