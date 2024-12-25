import functions

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

for f in functions.funcs:
    t = find_minimum_brute_force(f, 10000000)
    print("min_x = {} min_y = {}".format(t[0], t[1]))
