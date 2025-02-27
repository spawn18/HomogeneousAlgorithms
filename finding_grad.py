from matplotlib import pyplot as plt

import mishin_local_grad
import functions
import numpy as np

def grad(x, a, b):
    if a == 0 or b == 0:
        return 1

    if -b*a <= x <= b*a:
        return (1/a)*x+1
    elif x < -b*a:
        return -b+1
    else:
        return b+1

def average(a, b):
    results = mishin_local_grad.minimize(functions.funcs, grad_smoother=lambda t: grad(t, a, b))
    count = np.array([r.count for r in results])
    total = sum([1 if r.success else 0 for r in results])
    avg = np.average(count)
    return avg, '' if total == 20 else '*'

average_vec = np.vectorize(average)

a = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 1, 5, 10, 25, 50])
b = np.array([0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

A,B = np.meshgrid(a, b)
data = average_vec(A,B)

fig, ax = plt.subplots()
im = ax.imshow(data[0])

ax.set_xlabel('Наклон')
ax.set_ylabel('Макс. приращение')

# Show all ticks and label them with the respective list entries
ax.set_xticks(range(len(a)), labels=[str(i) for i in a])
ax.set_yticks(range(len(b)), labels=[str(i) for i in b])

# Loop over data dimensions and create text annotations.
for i in range(len(b)):
    for j in range(len(a)):
        text = ax.text(j, i, str(data[0][i,j])+str(data[1][i,j]), ha="center", va="center", color="w", fontsize='x-small')

ax.set_title("grad")
fig.tight_layout()
plt.savefig('grad_optimal.png')