import numpy as np
import matplotlib.pyplot as plt

#
#   Error amplitude to error frequency transform test
#

#
#   Parameters
#
n_point = 1024
n_test = 1000
n_exp = 6
exp_s = [i + 2 for i in range(n_exp)]
variance = 1

amp = 1
freq_start = 4
freq_end = 5
beta = 14

window_symmetric = np.kaiser(n_point + 1, beta)[0:n_point]
window_normal = window_symmetric / np.mean(window_symmetric)


#
# Estimator
#
def estimate(est_signal: np.array, window, exp_ratio):
    signal_windowed = est_signal * window
    signal_padding = np.zeros(n_point * exp_ratio)
    signal_padding[0:n_point] = signal_windowed
    spectrum = 2 * np.fft.fft(signal_padding) / n_point
    search_range = slice(freq_start * exp_ratio, freq_end * exp_ratio)
    delta = np.argmax(abs(spectrum[search_range]))
    est_freq = delta + freq_start * exp_ratio
    est_amp = spectrum[est_freq]
    return abs(est_amp), est_freq / exp_ratio


#
#   Modeling
#
errors_amp = np.zeros([n_exp, n_test])
errors_freq = np.zeros([n_exp, n_test])
errors_win = np.zeros([n_exp])

for i_exp, exp_ratio in enumerate(exp_s):
    freq_s = freq_start + np.random.random(n_test)
    errors_win[i_exp] = (1 - abs(np.sum(
        [window_normal[i] * np.exp(-1j * 2 * np.pi * i / (n_point * exp_ratio))
         for i in range(n_point)])) / n_point) / 2
    for i_test in range(n_test):
        noise = np.random.normal(0, np.sqrt(variance), n_point) * 0
        signal = amp * np.sin([2 * np.pi * freq_s[i_test] * i / n_point for i in range(n_point)])
        amp_est, freq_est = estimate(signal + noise, window_normal, exp_ratio)
        errors_amp[i_exp, i_test] = amp_est - amp
        errors_freq[i_exp, i_test] = freq_est - freq_s[i_test]

#
#   Analyse
#

amp_err = np.sqrt(np.mean(np.power(errors_amp, 2), 1))

plt.figure()
plt.plot(exp_s, amp_err, marker='.', color='blue')  # experimental
plt.plot(exp_s, errors_win, marker='^', color='red')  # theoretical

plt.show()
