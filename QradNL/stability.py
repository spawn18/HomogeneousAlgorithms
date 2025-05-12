import math
import QradNL
import functions
import numpy as np

from matplotlib import pyplot as plt

def arctan(x, a, b):
    if x < 0:
        return (2/math.pi) * math.atan((a-1)*x/b) + 1
    else:
        return (a-1)*(2/math.pi) * math.atan(x/b) + 1

def average(a, b):
    results = QradNL.minimize(functions.funcs, grad_smoother=lambda t: arctan(t, a, b))
    count = np.array([r.count for r in results])
    total = sum([1 if r.success else 0 for r in results])
    avg = np.average(count)
    return avg, '*' if total != 20 else ''

average_vec = np.vectorize(average)

a = np.linspace(1.7*(1-0.005), 1.7*(1+0.005), 10, dtype=np.float64)
b = np.linspace(2.35*(1-0.005), 2.35*(1+0.005), 10, dtype=np.float64)

A,B = np.meshgrid(a, b)
data = average_vec(A,B)

print("mean: " + str(np.mean(data[0])))
print("std: " + str(np.std(data[0])))

fig, ax = plt.subplots()
im = ax.imshow(data[0])

ax.set_xlabel('a')
ax.set_ylabel('b')

ax.set_xticks(range(len(a)), labels=[round(x, 4) for x in a], fontsize=4)
ax.set_yticks(range(len(b)), labels=[round(x, 4) for x in b], fontsize=4)

for i in range(len(b)):
    for j in range(len(a)):
        text = ax.text(j, i, str(data[0][i,j])+str(data[1][i,j]), ha="center", va="center", color="w", fontsize=6)

ax.set_title("Анализ чувствительности решения")
fig.tight_layout()
plt.savefig('stability.png', dpi=400)