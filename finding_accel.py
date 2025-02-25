from matplotlib import pyplot as plt
import mishin_local_accel_grad
import functions
import numpy as np

def accel_smoother(x, a, b):
    if -b * a <= x <= b * a:
        return (1 / a) * x + 1
    elif x < -b * a:
        return -b + 1
    else:
        return b + 1

def average(a, b):
    results = mishin_local_accel_grad.minimize(functions.funcs, accel_smoother=lambda t: accel_smoother(t, a, b))
    avg = np.average([r.count if r.success else 0 for r in results])
    if sum([1 if r.success else 0 for r in results]) == 20:
        return avg
    else:
        return 100.0

average_vec = np.vectorize(average)

a = np.array([0.1, 0.25, 0.33, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 25, 50, 100])
b = np.array([0.01, 0.025, 0.05, 0.1, 0.25, 0.33, 0.4, 0.5, 0.66, 0.75, 0.85, 1])

A,B = np.meshgrid(a, b)
data = average_vec(A,B)

fig, ax = plt.subplots()
im = ax.imshow(data)

ax.set_xlabel('Наклон')
ax.set_ylabel('Макс. приращение')

# Show all ticks and label them with the respective list entries
ax.set_xticks(range(len(a)), labels=[str(i) for i in a])
ax.set_yticks(range(len(b)), labels=[str(i) for i in b])

# Loop over data dimensions and create text annotations.
for i in range(len(b)):
    for j in range(len(a)):
        text = ax.text(j, i, data[i,j], ha="center", va="center", color="w")

ax.set_title("acceleration")
fig.tight_layout()
plt.savefig('acc_optimal.png')