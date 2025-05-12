import matplotlib.pyplot as plt
import numpy as np
import math


def theta(x, a, b):
    if x < 0:
        return (2/math.pi) * math.atan((a-1)*x/b) + 1
    else:
        return (a-1)*(2/math.pi) * math.atan(x/b) + 1

xs = np.linspace(-5, 5, 500)

theta1 = np.vectorize(lambda x: theta(x, 1.25, 1))
theta2 = np.vectorize(lambda x: theta(x, 1.5, 1))
theta3 = np.vectorize(lambda x: theta(x, 2, 1))

plt.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)

plt.plot(xs, theta1(xs), color=(1, 0.1, 1), label='θ(x, 1.25, 1)')
plt.plot(xs, theta2(xs), color=(0.5, 0.1, 0.5), label='θ(x, 1.5, 1)')
plt.plot(xs, theta3(xs), color=(0.0, 0.1, 1), label='θ(x, 2, 1)')
plt.legend(loc='best', ncol=2)

plt.savefig('theta_graphs.png', dpi=300)

