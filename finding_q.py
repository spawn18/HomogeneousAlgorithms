import math

from matplotlib import pyplot as plt

import QradNL
import functions
import numpy as np

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
    return avg, True if total != 20 else False

average_vec = np.vectorize(average)

q = np.linspace(1,10,10)

data = average_vec(q)

fig, ax = plt.subplots()

ax.set_xlabel('Показатель роста (q)')
ax.set_ylabel('Среднее число испытаний')
ax.scatter(q[data[1] == True], data[0][data[1] == True], marker='o', label="Точность достигнута")
ax.scatter(q[data[1] == False], data[0][data[1] == False], marker='x', label="Точность не достигнута")
plt.legend(loc='best', ncol=2)
ax.set_title("Поиск q")
fig.tight_layout()
plt.savefig('q_optimal.png', dpi=300)