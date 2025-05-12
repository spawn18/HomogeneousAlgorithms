import math

from matplotlib import pyplot as plt

import CubicSplineGrad
import functions
import numpy as np

def arctan(x, a, b, q=1):
    if x < 0:
        return q*(2/math.pi) * math.atan((a-1)*x/b) + 1
    else:
        return q*(a-1)*(2/math.pi) * math.atan(x/b) + 1

def average(a, b):
    results = CubicSplineGrad.minimize(functions.funcs, grad_smoother=lambda t: arctan(t, a, b))
    count = np.array([r.count for r in results])
    total = sum([1 if r.success else 0 for r in results])
    avg = np.average(count)
    return avg, '*' if total != 20 else ''

average_vec = np.vectorize(average)

a = np.linspace(1, 2, 21)
b = np.linspace(1, 10, 21)

A,B = np.meshgrid(a, b)
data = average_vec(A,B)

fig, ax = plt.subplots()
im = ax.imshow(data[0])

ax.set_xlabel('α')
ax.set_ylabel('β')

ax.set_xticks(range(len(a)), labels=[round(x, 2) for x in a], fontsize=4)
ax.set_yticks(range(len(b)), labels=[round(x, 2) for x in b], fontsize=4)

for i in range(len(b)):
    for j in range(len(a)):
        text = ax.text(j, i, str(data[0][i,j])+str(data[1][i,j]), ha="center", va="center", color="w", fontsize=4)

ax.set_title("Поиск параметров θ")
fig.tight_layout()
plt.savefig('grad_optimal.png', dpi=400)