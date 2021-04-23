# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 13:05:09 2021

@author: justinoberle
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import fftpack

# create signal
time_vec = np.arange(1, 11, 1)
wavelength = 1/.3
phase = 0
amp = 10
created_signal = amp * np.sin((2 * np.pi / wavelength * time_vec) + phase)

# plot it
fig, axs = plt.subplots(2, 1, figsize=(10,6))
axs[0].plot(time_vec, created_signal, label='exact_data')

# get fft and freq array
sig_fft = fftpack.fft(created_signal)
sample_freq = fftpack.fftfreq(created_signal.size, d=1)

# do inverse fft and verify same curve as original signal. This is fine!
filtered_signal = fftpack.ifft(sig_fft)
filtered_signal += np.mean(created_signal)

def find_peak_freqs(sample_freq, power, n=2):
    pos_mask = np.where(sample_freq > 0)
    freqs = sample_freq[pos_mask]
    pos_power = power[pos_mask]
    pos_power_sorted = sorted(pos_power)
    dominent = pos_power_sorted[-n:]
    # need to reorganize dominent to have proper sorting now
    index_list = []
    for i in range(len(dominent)):
        index_list.append(np.where(pos_power == dominent[i]))
    power_index = []
    for i in range(len(index_list)):
        power_index.append(pos_power[index_list[i]])
    dominant_freq_indices = []
    peak_freqs = []
    for i in range(len(dominent)):
        dominant_freq_indices.append(np.where(pos_power == dominent[i]))
        peak_freqs.append(freqs[dominant_freq_indices[i]])
    # # remove 1/f noise
    # peak_freqs_temp = peak_freqs.copy()
    # peak_freqs = [i for i in peak_freqs if round(1/i[0]) < len(sample_freq)]
    # if len(peak_freqs) < 1:
    #     peak_freqs = peak_freqs_temp
    return peak_freqs, pos_power, freqs 

# create individual signals for each frequency
filtered_signals = []
for i in range(len(sample_freq)):
    high_freq_fft = sig_fft.copy()
    high_freq_fft[np.abs(sample_freq) < np.nanmin(sample_freq[i])] = 0
    high_freq_fft[np.abs(sample_freq) > np.nanmax(sample_freq[i])] = 0
    filtered_sig = fftpack.ifft(high_freq_fft)
    filtered_sig -= np.mean(filtered_sig)
    filtered_signals.append(filtered_sig)

power = np.abs(sig_fft)
peak_freqs, pos_power, freqs = find_peak_freqs(sample_freq, power, n=len(created_signal))

# get phase, amplitude, and wavelength for each individual frequency
sig_size = len(created_signal)
wavelength = []
ph = []
amp = []
indices = []
for j in range(len(peak_freqs)):
    wavelength.append(1 / peak_freqs[j])
    indices.append(int(sig_size * peak_freqs[j]))
    sample_index = np.where(sample_freq==peak_freqs[j])
    phase = np.angle(sig_fft[j])
    phase = np.arctan2(sig_fft[sample_index].imag, sig_fft[sample_index].real)
    formula = ((((wavelength[j]) / 2) - 2) * np.pi) / wavelength[j]
    ph.append([phase + formula])
    amp.append([np.sqrt((sig_fft[sample_index].real * sig_fft[sample_index].real) + (sig_fft[sample_index].imag * sig_fft[sample_index].imag)) / (sig_size / 2)])

# create an equation for each frequency based on each phase, amp, and wavelength found from above.
def eqn(filtered_si, wavelength, time_vec, phase, amp):
    return amp * np.sin((2 * np.pi / wavelength * time_vec) + phase)
def find_equations(filtered_signals_mean, high_freq_fft, wavelength, filtered_signals, time_vec, ph, amp):
    equations = []
    for i in range(len(wavelength)):
        temp = eqn(filtered_signals[i], wavelength[i], time_vec, ph[i], amp[i])
        equations.append(temp + filtered_signals_mean)
    return equations

filtered_signals_mean = np.abs(np.mean(filtered_signals))
equations = find_equations(filtered_signals_mean, sig_fft, wavelength, 
                                      filtered_signals, time_vec, ph, amp)

# at this point each equation, for each frequency should match identically each signal from each frequency,
# however, the phase seems wrong and they do not match!!??
axs[0].plot(time_vec, filtered_signal, '--', linewidth=3, label='filtered_sig_combined')
axs[1].plot(time_vec, filtered_signals[3], label='filtered_sig[-1]')
axs[1].plot(time_vec, equations[3][0], label='equations[-1]')
axs[0].legend()
axs[1].legend()
fig.tight_layout()
plt.show()  











