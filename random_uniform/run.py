import math
import time
import matplotlib.pyplot as plt
import numpy as np

# m = 32768
m = 1000
a = 23
c = 12345


def lcm(seed, size):
    print(f'seed {seed}')
    if size == 1:
        return math.ceil(math.fmod(a * math.ceil(seed) + c, m))
    r = [0 for i in range(size)]
    r[0] = math.ceil(seed)
    for i in range(1, size):
        r[i] = math.ceil(math.fmod((a * r[i - 1] + c), m))
    return r[1:size]


data = np.array(lcm(time.time(), 10))
data = np.sort(data)
print(data)

f_x = 1 / (max(data) - min(data))
F_x = []
for i in range(data.size - 1):
    F_x.append((data[i] - min(data)) / (max(data) - min(data)))

F_x = np.sort(F_x)
# print(F_x)

plt.hist(data, bins=30, density=True, alpha=0.6,
         label='Гистограмма случайной величины')
plt.axhline(f_x, color='red', label='Плотность случайной величины')
plt.legend(fontsize=14, loc=1)
plt.grid(ls=':')
plt.show()
plt.close()
plt.plot(F_x, label='Функция распределения')
plt.show()

# оценка максимального правдободобия
R_A = data
theta_A = R_A / np.sum(R_A)
print(f'оценка максимального прадободобия {theta_A}')




