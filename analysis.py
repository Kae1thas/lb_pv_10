import numpy as np
import matplotlib.pyplot as plt

# Твои данные
procs = [1, 2, 4, 8, 9, 16]
time_seq = [10.9694] * len(procs)
time_1d = [10.9694, 5.4237, 3.7465, 1.8911, np.nan, 2.3932]  # nan для 9
time_2d = [10.9694, np.nan, 3.1411, np.nan, 1.8307, 2.3406]  # nan для 2,8

# Расчёты
speedup_1d = np.array([time_seq[0] / t if not np.isnan(t) else np.nan for t in time_1d])
speedup_2d = np.array([time_seq[0] / t if not np.isnan(t) else np.nan for t in time_2d])
efficiency_1d = speedup_1d / procs
efficiency_2d = speedup_2d / procs

# График времени
fig1, ax1 = plt.subplots(figsize=(5, 5))
ax1.plot(procs, time_seq, 'k-', label='Seq')
ax1.plot(procs, time_1d, 'b-o', label='1D')
ax1.plot(procs, time_2d, 'r-s', label='2D')
ax1.set_title('Время выполнения')
ax1.set_xlabel('Число процессов (P)')
ax1.set_ylabel('Время, сек')
ax1.legend()
plt.tight_layout()
plt.savefig('time.png')
plt.show()

# График ускорения
fig2, ax2 = plt.subplots(figsize=(5, 5))
ax2.plot(procs, speedup_1d, 'b-o', label='1D')
ax2.plot(procs, speedup_2d, 'r-s', label='2D')
ax2.plot(procs, procs, 'k--', label='Идеальное')
ax2.set_title('Ускорение')
ax2.set_xlabel('Число процессов (P)')
ax2.set_ylabel('Ускорение (S)')
ax2.legend()
plt.tight_layout()
plt.savefig('speedup.png')
plt.show()

# График эффективности
fig3, ax3 = plt.subplots(figsize=(5, 5))
ax3.plot(procs, efficiency_1d, 'b-o', label='1D')
ax3.plot(procs, efficiency_2d, 'r-s', label='2D')
ax3.axhline(y=1.0, color='k', linestyle='--', label='Идеал')
ax3.set_title('Эффективность')
ax3.set_xlabel('Число процессов (P)')
ax3.set_ylabel('Эффективность (E)')
ax3.legend()
plt.tight_layout()
plt.savefig('efficiency.png')
plt.show()