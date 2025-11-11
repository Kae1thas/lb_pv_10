from mpi4py import MPI
import numpy as np
from numpy import empty, linspace, tanh, savez
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

# Параметры (глобальные)
start_time = time.time()
a, b, c, d = -2, 2, -2, 2
t_0, T = 0, 4
eps = 10**(-1.5)
N_x, N_y, M = 50, 50, 500
x, h_x = linspace(a, b, N_x + 1, retstep=True)
y, h_y = linspace(c, d, N_y + 1, retstep=True)
t, tau = linspace(t_0, T, M + 1, retstep=True)

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

if numprocs < 1:
    if rank == 0:
        print("Нужно хотя бы 1 процесс!")
    comm.Abort()

# Виртуальная топология: линейка
comm_cart = comm.Create_cart(dims=[numprocs], periods=[False], reorder=True)
rank_cart = comm_cart.Get_rank()

# Функция для rcounts и displs
def auxiliary_arrays_determination(MM, num):
    ave, res = divmod(MM, num)
    rcounts = np.empty(num, dtype=np.int32)
    displs = np.empty(num, dtype=np.int32)
    for k in range(num):
        rcounts[k] = ave + 1 if k < res else ave
        displs[k] = 0 if k == 0 else displs[k-1] + rcounts[k-1]
    return rcounts, displs

rcounts_N_x, displs_N_x = auxiliary_arrays_determination(N_x + 1, numprocs)
N_x_part = rcounts_N_x[rank_cart]

# Вспомогательные размеры (призрачные слои)
if rank_cart in [0, numprocs - 1]:
    N_x_part_aux = N_x_part + 1
else:
    N_x_part_aux = N_x_part + 2
u_part_aux = np.empty((M + 1, N_x_part_aux, N_y + 1), dtype=np.float64)

# Индексы для реальных точек
send_start = 0 if rank_cart == 0 else 1
send_end = N_x_part + send_start

# Локальные глобальные индексы для init
x_local_start = displs_N_x[rank_cart]

# Инициализация реальных точек для m=0
for i_local in range(N_x_part):
    i_aux = send_start + i_local
    i_global = x_local_start + i_local
    for j in range(N_y + 1):
        u_part_aux[0, i_aux, j] = u_init(x[i_global], y[j], eps)

# Установка границ (константы для всех m сразу)
u_part_aux[:, :, 0] = 0.33  # bottom
u_part_aux[:, :, N_y] = 0.33  # top
if rank_cart == 0:
    u_part_aux[:, 0, :] = 0.33  # left
if rank_cart == numprocs - 1:
    right_boundary_idx = send_start + N_x_part - 1
    u_part_aux[:, right_boundary_idx, :] = 0.33  # right

# Начальный обмен для призраков на m=0
if rank_cart > 0:
    comm_cart.Sendrecv(
        sendbuf=[u_part_aux[0, 1, :], N_y + 1, MPI.DOUBLE],
        dest=rank_cart - 1, sendtag=0,
        recvbuf=[u_part_aux[0, 0, :], N_y + 1, MPI.DOUBLE],
        source=rank_cart - 1, recvtag=MPI.ANY_TAG
    )
if rank_cart < numprocs - 1:
    comm_cart.Sendrecv(
        sendbuf=[u_part_aux[0, -2, :], N_y + 1, MPI.DOUBLE],
        dest=rank_cart + 1, sendtag=0,
        recvbuf=[u_part_aux[0, -1, :], N_y + 1, MPI.DOUBLE],
        source=rank_cart + 1, recvtag=MPI.ANY_TAG
    )

# Основной цикл
for m in range(M):
    # Вычисления для внутренних точек
    for i in range(1, N_x_part_aux - 1):
        for j in range(1, N_y):
            d2x = (u_part_aux[m, i + 1, j] - 2 * u_part_aux[m, i, j] + u_part_aux[m, i - 1, j]) / h_x**2
            d2y = (u_part_aux[m, i, j + 1] - 2 * u_part_aux[m, i, j] + u_part_aux[m, i, j - 1]) / h_y**2
            d1x = (u_part_aux[m, i + 1, j] - u_part_aux[m, i - 1, j]) / (2 * h_x)
            d1y = (u_part_aux[m, i, j + 1] - u_part_aux[m, i, j - 1]) / (2 * h_y)
            u_part_aux[m + 1, i, j] = u_part_aux[m, i, j] + tau * (eps * (d2x + d2y) + u_part_aux[m, i, j] * (d1x + d1y) + u_part_aux[m, i, j]**3)

    # Обмен граничными значениями для m+1
    if rank_cart > 0:
        comm_cart.Sendrecv(
            sendbuf=[u_part_aux[m + 1, 1, :], N_y + 1, MPI.DOUBLE],
            dest=rank_cart - 1, sendtag=0,
            recvbuf=[u_part_aux[m + 1, 0, :], N_y + 1, MPI.DOUBLE],
            source=rank_cart - 1, recvtag=MPI.ANY_TAG
        )
    if rank_cart < numprocs - 1:
        comm_cart.Sendrecv(
            sendbuf=[u_part_aux[m + 1, -2, :], N_y + 1, MPI.DOUBLE],
            dest=rank_cart + 1, sendtag=0,
            recvbuf=[u_part_aux[m + 1, -1, :], N_y + 1, MPI.DOUBLE],
            source=rank_cart + 1, recvtag=MPI.ANY_TAG
        )

# Синхронизация и сбор полного u на rank 0 с Gatherv (tuple для recvbuf)
comm.Barrier()
rcounts_layer = np.array([rcounts_N_x[k] * (N_y + 1) for k in range(numprocs)], dtype=np.int32)
rcounts_total = rcounts_layer * (M + 1)  # counts для каждого процесса (всего элементов)

if rank == 0:
    u_full = empty((M + 1, N_x + 1, N_y + 1), dtype=np.float64)
    recvbuf_tuple = (u_full.ravel(), rcounts_total)  # tuple: (buffer, counts)
else:
    recvbuf_tuple = None

real_part = u_part_aux[:, send_start:send_end, :].ravel()
comm.Gatherv(real_part, recvbuf_tuple, root=0)

# Вывод и сохранение только на root
if rank == 0:
    print(f"Время выполнения (1D, {numprocs} proc): {time.time() - start_time:.4f} сек")
    savez("results_1d.npz", x=x, y=y, t=t, u=u_full)

    # Анимация
    fig, ax = plt.subplots()
    def animate(m):
        ax.clear()
        ax.imshow(u_full[m], extent=[a, b, c, d], origin='lower', cmap='viridis')
        ax.set_title(f'Время t = {t[m]:.2f} (1D parallel)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    anim = FuncAnimation(fig, animate, frames=range(0, M+1, 10), interval=200, repeat=True)
    plt.show()