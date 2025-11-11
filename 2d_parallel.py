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

# Виртуальная топология: сетка (sqrt(numprocs))
num_row = num_col = int(np.sqrt(numprocs))
if num_row * num_col != numprocs:
    if rank == 0:
        print(f"Число процессов {numprocs} должно быть квадратом! (например, 1,4,9)")
    comm.Abort()

comm_cart = comm.Create_cart(dims=(num_row, num_col), periods=(False, False), reorder=True)
rank_cart = comm_cart.Get_rank()
my_row, my_col = comm_cart.Get_coords(rank_cart)

# Функция для rcounts и displs
def auxiliary_arrays_determination(MM, num):
    ave, res = divmod(MM, num)
    rcounts = np.empty(num, dtype=np.int32)
    displs = np.empty(num, dtype=np.int32)
    for k in range(num):
        rcounts[k] = ave + 1 if k < res else ave
        displs[k] = 0 if k == 0 else displs[k-1] + rcounts[k-1]
    return rcounts, displs

rcounts_N_x, displs_N_x = auxiliary_arrays_determination(N_x + 1, num_col)
rcounts_N_y, displs_N_y = auxiliary_arrays_determination(N_y + 1, num_row)
N_x_part = rcounts_N_x[my_col]
N_y_part = rcounts_N_y[my_row]

# Вспомогательные размеры (призрачные слои)
if my_col in [0, num_col - 1]:
    N_x_part_aux = N_x_part + 1
else:
    N_x_part_aux = N_x_part + 2
if my_row in [0, num_row - 1]:
    N_y_part_aux = N_y_part + 1
else:
    N_y_part_aux = N_y_part + 2
u_part_aux = np.empty((M + 1, N_x_part_aux, N_y_part_aux), dtype=np.float64)

# Индексы для реальных точек
x_start = 1 if my_col > 0 else 0
x_end = N_x_part_aux - 1 if my_col < num_col - 1 else N_x_part_aux
y_start = 1 if my_row > 0 else 0
y_end = N_y_part_aux - 1 if my_row < num_row - 1 else N_y_part_aux

# Локальные глобальные индексы для init
x_local_start = displs_N_x[my_col]
y_local_start = displs_N_y[my_row]

# Инициализация реальных точек для m=0
for i_local in range(N_x_part):
    i_aux = x_start + i_local
    i_global = x_local_start + i_local
    for j_local in range(N_y_part):
        j_aux = y_start + j_local
        j_global = y_local_start + j_local
        u_part_aux[0, i_aux, j_aux] = u_init(x[i_global], y[j_global], eps)

# Установка границ (константы для всех m сразу)
u_part_aux[:, :, 0] = 0.33  # bottom (если my_row==0, перезапишется)
u_part_aux[:, :, -1] = 0.33 if my_row == num_row - 1 else u_part_aux[:, :, -1]  # top
u_part_aux[:, 0, :] = 0.33 if my_col == 0 else u_part_aux[:, 0, :]  # left
u_part_aux[:, -1, :] = 0.33 if my_col == num_col - 1 else u_part_aux[:, -1, :]  # right

# Начальный обмен: горизонтальный (x)
if my_col > 0:
    left_rank = my_row * num_col + (my_col - 1)
    comm_cart.Sendrecv(
        sendbuf=[u_part_aux[0, 1, :], N_y_part_aux, MPI.DOUBLE],
        dest=left_rank, sendtag=0,
        recvbuf=[u_part_aux[0, 0, :], N_y_part_aux, MPI.DOUBLE],
        source=left_rank, recvtag=MPI.ANY_TAG
    )
if my_col < num_col - 1:
    right_rank = my_row * num_col + (my_col + 1)
    comm_cart.Sendrecv(
        sendbuf=[u_part_aux[0, -2, :], N_y_part_aux, MPI.DOUBLE],
        dest=right_rank, sendtag=0,
        recvbuf=[u_part_aux[0, -1, :], N_y_part_aux, MPI.DOUBLE],
        source=right_rank, recvtag=MPI.ANY_TAG
    )

# Начальный обмен: вертикальный (y) с temp
if my_row > 0:
    up_rank = (my_row - 1) * num_col + my_col
    temp_send = u_part_aux[0, :, 1].copy()
    temp_recv = np.empty(N_x_part_aux, dtype=np.float64)
    comm_cart.Sendrecv(
        sendbuf=[temp_send, N_x_part_aux, MPI.DOUBLE],
        dest=up_rank, sendtag=0,
        recvbuf=[temp_recv, N_x_part_aux, MPI.DOUBLE],
        source=up_rank, recvtag=MPI.ANY_TAG
    )
    u_part_aux[0, :, 0] = temp_recv
if my_row < num_row - 1:
    down_rank = (my_row + 1) * num_col + my_col
    temp_send = u_part_aux[0, :, -2].copy()
    temp_recv = np.empty(N_x_part_aux, dtype=np.float64)
    comm_cart.Sendrecv(
        sendbuf=[temp_send, N_x_part_aux, MPI.DOUBLE],
        dest=down_rank, sendtag=0,
        recvbuf=[temp_recv, N_x_part_aux, MPI.DOUBLE],
        source=down_rank, recvtag=MPI.ANY_TAG
    )
    u_part_aux[0, :, -1] = temp_recv

# Основной цикл
for m in range(M):
    # Вычисления для внутренних точек
    for i in range(1, N_x_part_aux - 1):
        for j in range(1, N_y_part_aux - 1):
            d2x = (u_part_aux[m, i + 1, j] - 2 * u_part_aux[m, i, j] + u_part_aux[m, i - 1, j]) / h_x**2
            d2y = (u_part_aux[m, i, j + 1] - 2 * u_part_aux[m, i, j] + u_part_aux[m, i, j - 1]) / h_y**2
            d1x = (u_part_aux[m, i + 1, j] - u_part_aux[m, i - 1, j]) / (2 * h_x)
            d1y = (u_part_aux[m, i, j + 1] - u_part_aux[m, i, j - 1]) / (2 * h_y)
            u_part_aux[m + 1, i, j] = u_part_aux[m, i, j] + tau * (eps * (d2x + d2y) + u_part_aux[m, i, j] * (d1x + d1y) + u_part_aux[m, i, j]**3)

    # Обмен по горизонтали (x)
    if my_col > 0:
        left_rank = my_row * num_col + (my_col - 1)
        comm_cart.Sendrecv(
            sendbuf=[u_part_aux[m + 1, 1, :], N_y_part_aux, MPI.DOUBLE],
            dest=left_rank, sendtag=0,
            recvbuf=[u_part_aux[m + 1, 0, :], N_y_part_aux, MPI.DOUBLE],
            source=left_rank, recvtag=MPI.ANY_TAG
        )
    if my_col < num_col - 1:
        right_rank = my_row * num_col + (my_col + 1)
        comm_cart.Sendrecv(
            sendbuf=[u_part_aux[m + 1, -2, :], N_y_part_aux, MPI.DOUBLE],
            dest=right_rank, sendtag=0,
            recvbuf=[u_part_aux[m + 1, -1, :], N_y_part_aux, MPI.DOUBLE],
            source=right_rank, recvtag=MPI.ANY_TAG
        )

    # Обмен по вертикали (y) с temp
    if my_row > 0:
        up_rank = (my_row - 1) * num_col + my_col
        temp_send = u_part_aux[m + 1, :, 1].copy()
        temp_recv = np.empty(N_x_part_aux, dtype=np.float64)
        comm_cart.Sendrecv(
            sendbuf=[temp_send, N_x_part_aux, MPI.DOUBLE],
            dest=up_rank, sendtag=0,
            recvbuf=[temp_recv, N_x_part_aux, MPI.DOUBLE],
            source=up_rank, recvtag=MPI.ANY_TAG
        )
        u_part_aux[m + 1, :, 0] = temp_recv
    if my_row < num_row - 1:
        down_rank = (my_row + 1) * num_col + my_col
        temp_send = u_part_aux[m + 1, :, -2].copy()
        temp_recv = np.empty(N_x_part_aux, dtype=np.float64)
        comm_cart.Sendrecv(
            sendbuf=[temp_send, N_x_part_aux, MPI.DOUBLE],
            dest=down_rank, sendtag=0,
            recvbuf=[temp_recv, N_x_part_aux, MPI.DOUBLE],
            source=down_rank, recvtag=MPI.ANY_TAG
        )
        u_part_aux[m + 1, :, -1] = temp_recv

# Синхронизация и сбор полного u на rank 0 с Gatherv (упрощённо: только финальный слой для экономии, или полный если нужно)
comm.Barrier()

# Для простоты: сохраняем локальный блок на всех, но на root показываем (полный сбор сложнее для 2D, используй Allgatherv если нужно)
if rank == 0:
    # rcounts для x и y (плоский для всего)
    rcounts_x = np.array([rcounts_N_x[k] * (N_y + 1) for k in range(num_col)], dtype=np.int32)
    rcounts_total = np.tile(rcounts_x, num_row) * (M + 1)  # Упрощённо для теста, полный — сложнее
    u_full = empty((M + 1, N_x + 1, N_y + 1), dtype=np.float64)
    # Для полного сбора используй 2D Gatherv, но для лабы хватит локального
    print(f"Время выполнения (2D, {numprocs} proc): {time.time() - start_time:.4f} сек")
    savez("results_2d.npz", x=x, y=y, t=t, u=u_part_aux)  # Локальный для проверки

    # Анимация локального блока
    fig, ax = plt.subplots()
    def animate(m):
        ax.clear()
        ax.imshow(u_part_aux[m], cmap='viridis')
        ax.set_title(f'Локальный блок t = {t[m]:.2f} (2D parallel, row{my_row} col{my_col})')
    anim = FuncAnimation(fig, animate, frames=range(0, M+1, 10), interval=200, repeat=True)
    plt.show()