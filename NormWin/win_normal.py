import numpy as np
from matplotlib import pyplot as plt

n_point = 1024

#
#   Make normal window
#

kaiser_win = np.kaiser(n_point, 10)
kaiser_spec = np.fft.fft(kaiser_win)
symmetric_win = np.kaiser(n_point + 1, 10)[0:n_point]
symmetric_spec = np.fft.fft(symmetric_win)
normal_win = symmetric_win / np.sum(symmetric_win)
normal_spec = np.fft.fft(normal_win)

print("Обычное окно кайзера", kaiser_spec[0:5])
print("Симметричное окно", symmetric_spec[0:5])
print("Нормализованное окно", normal_spec[0:5])

#
#   Plot figure
#

# n_win = 256

# win_rect = np.zeros(n_point)
# win_rect[0:n_win] = 1
# sp_win_rect = np.fft.fft(win_rect)

# win_kaiser = np.zeros(n_point)
# win_kaiser[0:n_win] = np.kaiser(n_win, 10)
# sp_win_kaiser = np.fft.fft(win_kaiser)

# plt.figure("Rectangle window")
# plt.plot(win_rect)
# plt.figure("Spectrum of rectangle window")
# plt.plot(abs(sp_win_rect))
# plt.xlim([0, 20])
# plt.figure("Kaiser window")
# plt.plot(win_kaiser)
# plt.figure("Spectrum of kaiser window")
# plt.plot(abs(sp_win_kaiser))
# plt.xlim([0, 20])
#
# plt.show()

#
#   Test window
#

n_test = 100
freq_s = np.random.rand(n_test) + 3
amp_s = np.random.rand(n_test)
phi_s = np.random.rand(n_test) * np.pi
eval = np.zeros(n_test, dtype=complex)
for i_test in range(n_test):
    signal = amp_s[i_test] * np.cos([2 * np.pi * freq_s[i_test] * i / n_point - phi_s[i_test] for i in range(n_point)])
    signal_windowed = signal * normal_win
    eval[i_test] = 2 * np.sum([signal_windowed[i] * np.exp(1j * freq_s[i_test] * i * 2 * np.pi / n_point) for i in range(n_point)])

print(max(np.abs(amp_s-np.abs(eval))))
print(max(np.abs(phi_s-np.angle(eval))))

