import math
import time
import matplotlib.pyplot as plt
import numpy as np

# m = 10
# m = 32768
# a = 23
# c = 12345
a = 106
c = 1283
m = 6075


def lcm(seed, size):
    print(f'seed {seed}')
    if size == 1:
        return math.ceil(math.fmod(a * math.ceil(seed) + c, m))
    r = [0 for i in range(size)]
    r[0] = math.ceil(seed)
    for i in range(1, size):
        r[i] = math.ceil(math.fmod((a * r[i - 1] + c), m))
    return r[1:size]


data = np.array(lcm(46.30239009857178, 250))
print(data)

data = np.sort(data)

f_x = 1 / (max(data) - min(data))
F_x = []
for i in range(data.size - 1):
    F_x.append((data[i] - min(data)) / (max(data) - min(data)))

F_x = np.sort(F_x)

# оценка максимального правдободобия
R_A = data
theta_A = R_A / np.sum(R_A)
print(f'оценка максимального прадободобия {theta_A}')

plt.hist(data, density=True, alpha=0.6, label='Гистограмма случайной величины', edgecolor='black')
plt.hlines(f_x, xmin=min(data), xmax=max(data), color='red', label='Плотность вероятности')
plt.legend(fontsize=9, loc=3)
plt.grid(ls=':')
plt.show()
plt.close()
plt.plot(F_x, label='Функция распределения')
plt.legend(fontsize=9, loc=4)
plt.show()
plt.close()
plt.plot(theta_A, label='Оценка максимального правдободобия')
plt.legend(fontsize=9, loc=4)
plt.show()
