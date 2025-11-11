import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from numpy import empty, linspace, tanh, savez

def u_init(x, y, eps):
    return 0.5 * tanh(1 / eps * ((x - 0.5)**2 + (y - 0.5)**2 - 0.35**2)) - 0.17

def u_left(y, t):
    return 0.33

def u_right(y, t):
    return 0.33

def u_top(x, t):
    return 0.33

def u_bottom(x, t):
    return 0.33

# Параметры
start_time = time.time()
a, b, c, d = -2, 2, -2, 2
t_0, T = 0, 4
eps = 10**(-1.5)
N_x, N_y, M = 50, 50, 500
x, h_x = linspace(a, b, N_x + 1, retstep=True)
y, h_y = linspace(c, d, N_y + 1, retstep=True)
t, tau = linspace(t_0, T, M + 1, retstep=True)

u = empty((M + 1, N_x + 1, N_y + 1))

# Инициализация начальных условий
for i in range(N_x + 1):
    for j in range(N_y + 1):
        u[0, i, j] = u_init(x[i], y[j], eps)

# Установка граничных условий для всех слоёв времени
for m in range(M + 1):
    for j in range(N_y + 1):
        u[m, 0, j] = u_left(y[j], t[m])
        u[m, N_x, j] = u_right(y[j], t[m])
    for i in range(N_x + 1):
        u[m, i, 0] = u_bottom(x[i], t[m])
        u[m, i, N_y] = u_top(x[i], t[m])

# Основной цикл вычислений (только внутренние точки)
for m in range(M):
    for i in range(1, N_x):
        for j in range(1, N_y):
            d2x = (u[m, i + 1, j] - 2 * u[m, i, j] + u[m, i - 1, j]) / h_x**2
            d2y = (u[m, i, j + 1] - 2 * u[m, i, j] + u[m, i, j - 1]) / h_y**2
            d1x = (u[m, i + 1, j] - u[m, i - 1, j]) / (2 * h_x)
            d1y = (u[m, i, j + 1] - u[m, i, j - 1]) / (2 * h_y)
            u[m + 1, i, j] = u[m, i, j] + tau * (eps * (d2x + d2y) + u[m, i, j] * (d1x + d1y) + u[m, i, j]**3)

print(f"Время выполнения: {time.time() - start_time:.4f} сек")

# Сохранение результатов
savez("results_seq.npz", x=x, y=y, t=t, u=u)

# Визуализация: анимация эволюции u
fig, ax = plt.subplots()
def animate(m):
    ax.clear()
    ax.imshow(u[m], extent=[a, b, c, d], origin='lower', cmap='viridis')
    ax.set_title(f'Время t = {t[m]:.2f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

anim = FuncAnimation(fig, animate, frames=M+1, interval=50, repeat=True)
plt.show()  # Или anim.save('animation.gif') для сохранения