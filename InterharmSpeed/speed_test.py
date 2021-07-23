from time import process_time

import matplotlib.pyplot as plt
import numpy as np


#
#   Interharmonics calculation speed test
#

#   Funs for interharmonics calculation

def get_corr_matrix(signal_size, k, freq_s):
    return np.array([[np.exp(
        -1j * 2 * np.pi * (freq_s + j / k) * i / signal_size
    ) for i in range(signal_size)] for j in range(k)])


def corr(signal_windowed: np.ndarray, matrix: np.ndarray):
    return matrix.dot(signal_windowed)


def added_fft(signal: np.ndarray, n: int, k: int):
    signal_padding = np.zeros(n * k)
    signal_padding[0:n] = signal
    return np.fft.fft(signal_padding)


def get_pruned_matrix(n: int, k: int):
    return np.array([[np.exp(-2 * np.pi * i * j * 1j / (k * n)) for i in range(n)] for j in range(k)])


def pruned_fft_pre(signal: np.ndarray, n: int, k: int):
    return np.array([[signal[i] / k for i in range(n)] for j in range(k)])


def pruned_fft_post(result: np.ndarray, n: int, k: int):
    return np.array([result[j, i] * k for i in range(n) for j in range(k)])


def pruned_fft(signal_matrix: np.ndarray, pruned_matrix: np.ndarray):
    return np.fft.fftn(signal_matrix * pruned_matrix, axes=[1])


#   Correctness test

n_point = 16
inter = 4
n_test = 1

amp = 1
freq_start = 5
freq = 5.2

beta = 14
window_symmetric = np.kaiser(n_point + 1, beta)[0:n_point]
window_normal = window_symmetric / np.mean(window_symmetric)

corr_matrix = get_corr_matrix(n_point, inter, freq_start)
pruned_matrix = get_pruned_matrix(n_point, inter)
err_fft = 0
err_pruned = 0
fft_slice = slice(freq_start * inter,(freq_start + 1) * inter)
for i_test in range(n_test):
    signal = amp * np.sin([2 * np.pi * freq * i / n_point for i in range(n_point)]) * window_normal
    sp_corr = corr(signal, corr_matrix) * 2 / n_point
    sp_fft = 2 * added_fft(signal, n_point, inter)[fft_slice] / n_point
    signal_matrix = pruned_fft_pre(signal, n_point, inter)
    result_matrix = pruned_fft(signal_matrix, pruned_matrix)
    sp_pruned = 2 * pruned_fft_post(result_matrix, n_point, inter)[fft_slice] / n_point
    err_fft += max(abs(abs(sp_corr) - abs(sp_fft)))
    err_pruned += max(abs(abs(sp_corr) - abs(sp_pruned)))
print(err_fft)
print(err_pruned)


#   Speed test
n_point = 2048
n_test = 100
n_calc_max = 10000
n_inters = [10, 50, 100, 500, 1000] # , 5000, 10000]
corr_times = np.zeros(len(n_inters))
pruned_times = np.zeros(len(n_inters))
fft_times = np.zeros(len(n_inters))

for i_inter, inter in enumerate(n_inters):
    print('Inter ', inter)
    corr_matrix = get_corr_matrix(n_point, inter, 4)
    pruned_matrix = get_pruned_matrix(n_point, inter)
    signal = np.zeros(n_point)
    signal_matrix = pruned_fft_pre(signal, n_point, inter)
    n_calc = n_calc_max // inter
    for i_test in range(n_test):
        t = process_time()
        for i_calc in range(n_calc):
            corr(signal, corr_matrix)
        corr_times[i_inter] += (process_time() - t)

        t = process_time()
        for i_calc in range(n_calc):
            pruned_fft(signal_matrix, pruned_matrix)
        pruned_times[i_inter] += (process_time() - t)

        t = process_time()
        for i_calc in range(n_calc):
            added_fft(signal, n_point, inter)
        fft_times[i_inter] += (process_time() - t)
    corr_times[i_inter] /= (n_calc * n_test)
    pruned_times[i_inter] /= (n_calc * n_test)
    fft_times[i_inter] /= (n_calc * n_test)

print(pruned_times)
print(corr_times)
print(fft_times)
plt.plot(n_inters, pruned_times, marker='.', color='blue')  # pruned FFT
plt.plot(n_inters, fft_times, marker='^', color='red')    # fft
plt.plot(n_inters, corr_times*40, marker='*', color='green')  # correlator
plt.show()
