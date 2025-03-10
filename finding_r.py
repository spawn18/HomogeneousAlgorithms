from matplotlib import pyplot as plt
import mishin_localnconvex_grad
import functions
import numpy as np

def grad1(x, a=0.25, b=0.4):
    if a == 0 or b == 0:
        return 1

    if -b*a <= x <= b*a:
        return (1/a)*x+1
    elif x < -b*a:
        return -b+1
    else:
        return b+1


def average(r):
    results = mishin_localnconvex_grad.minimize(functions.funcs, grad_smoother=lambda x: grad1(x, 0.25, 0.4), exponent=6.5, r=r)
    count = np.array([r.count for r in results])
    total = sum([1 if r.success else 0 for r in results])
    avg = np.average(count)
    return avg, True if total == 20 else False

average_vec = np.vectorize(average)

q = np.linspace(1,1.5,100)

data = average_vec(q)

fig, ax = plt.subplots()

ax.set_xlabel('r')
ax.set_ylabel('Среднее число испытаний')
ax.scatter(q[data[1] == True], data[0][data[1] == True], marker='o', label="Точность достигнута")
ax.scatter(q[data[1] == False], data[0][data[1] == False], marker='x', label="Точность не достигнута")
plt.legend(loc='best', ncol=2)
ax.set_title("Поиск множителя константы Липшица (r)")
fig.tight_layout()
plt.savefig('r_optimal.png', dpi=300)