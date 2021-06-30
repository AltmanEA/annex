import numpy as np

#
#   Test of an equation for a window spectrum for non integer point
#

n_point = 1024
beta = 14
exp_ratio = 10

window = np.kaiser(n_point + 1, beta)[0:n_point]
window_normal = window / np.mean(window)
win_spec = abs(np.fft.fft(window_normal))
window_exp = np.zeros(n_point * exp_ratio)
window_exp[0:n_point] = window_normal
win_spec_exp = abs(np.fft.fft(window_exp) / n_point)

n_calc = 10
win_spec_calc = np.zeros(n_calc)
win_spec_exp_calc = np.zeros(n_calc)
for j in range(n_calc):
    win_spec_calc[j] = abs(np.sum(
        [window_normal[i] * np.exp(-1j * 2 * np.pi * j * i / n_point)
         for i in range(n_point)]))
    win_spec_exp_calc[j] = abs(np.sum(
        [window_normal[i] * np.exp(-1j * 2 * np.pi * j * i / (n_point * exp_ratio))
         for i in range(n_point)])) / n_point

print(win_spec_calc)
print(win_spec[0:n_calc])
print(win_spec_exp_calc)
print(win_spec_exp[0:n_calc])

