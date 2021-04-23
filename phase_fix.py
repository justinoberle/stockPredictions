# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 18:26:32 2021

@author: justinoberle
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import fftpack
from scipy import integrate


time_vec = np.arange(1, 11, 1)
wavelength = [1/.3, 1/.2]
phase = [0, 1]
amp = [10.1, 10]
equation0 = amp[0] * np.sin((2 * np.pi / wavelength[0] * time_vec) + phase[0])
equation1 = amp[1] * np.sin((2 * np.pi / wavelength[1] * time_vec) + phase[1])
created_equation = equation0 + equation1

fig, axs = plt.subplots(3, 1, figsize=(10,6))
axs[0].plot(time_vec, created_equation, label='exact_data')



''' Do FFT and to get power and freq '''
def fft(sig, dt = 1):
    sig_fft = fftpack.fft(sig)
    power = np.abs(sig_fft)
    sample_freq = fftpack.fftfreq(sig.size, d=dt)
    return sig_fft, power, sample_freq

# find n most dominent frequencies
def find_peak_freqs(sample_freq, power, n=2):
    # this will break if there are duplicate frequencies with the exact same amplitude but this is highly unlikely so I left it as is
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

def create_n_signals(sig_mean, sig, sig_fft, peak_freqs, sample_freq):
    # loop through n iteratons to break apart signal into each frequency by itself. filtered_signals
    filtered_signals = []
    for i in range(len(peak_freqs)):
        high_freq_fft = sig_fft.copy()
        high_freq_fft[np.abs(sample_freq) < np.nanmin(peak_freqs[i])] = 0
        high_freq_fft[np.abs(sample_freq) > np.nanmax(peak_freqs[i])] = 0
        filtered_sig = fftpack.ifft(high_freq_fft)
        filtered_sig += sig_mean
        filtered_signals.append(filtered_sig)
    return filtered_signals

def create_combined_filtered_sig(sig_mean, sig, sig_fft, peak_freqs, sample_freq):
    # zero out every frequency that is not in peak frequencies, then inv FFT and only those are left
    # This is only used in plotting things for verification. Don't use for predictions only as it will slow down performance
    high_freq_fft = np.empty(len(sig_fft), dtype='complex128')
    high_freq_fft.fill(0)
    peak_freqs = [i[0] for i in peak_freqs]
    for j in range(len(high_freq_fft)):
        if np.abs(sample_freq[j]) in peak_freqs:
            high_freq_fft[j] = sig_fft[j]
        else:
            high_freq_fft[j] = sig_fft[j]
            pass
    filtered_sig = fftpack.ifft(high_freq_fft)
    filtered_sig += sig_mean
    return filtered_sig, high_freq_fft

def find_phases_wavelengths_amps(filtered_signals, high_freq_fft, peak_freqs, sig_size):
    ''' need to find where filtered_sig crosses mean and is increasing then use this index as phase start '''
    # do this for each frequency to find array of phases and wavelengths
    wavelength = []
    ph = []
    amp = []
    for j in range(len(peak_freqs)):
        wavelength.append(1 / peak_freqs[j])
        sample_index = np.where(sample_freq==peak_freqs[j])
        phase = np.arctan2(high_freq_fft[sample_index].imag, high_freq_fft[sample_index].real)
        formula = ((((wavelength[j][0]) / 2) - 2) * np.pi) / wavelength[j][0]
        ph.append([phase + formula])
        amp.append([np.sqrt((high_freq_fft[sample_index].real * high_freq_fft[sample_index].real) + (high_freq_fft[sample_index].imag * high_freq_fft[sample_index].imag)) / (sig_size / 2)])
    return ph, wavelength, amp
    
def eqn(filtered_si, wavelength, time_vec, phase, amp):
    return amp * np.sin((2 * np.pi / wavelength * time_vec) + phase)# + np.mean(filtered_si)

def find_equations(filtered_signals_mean, high_freq_fft, wavelength, filtered_signals, time_vec, ph, amp):
    # find equation for each frequency and place into a list of equations
    equations = []
    power_new = np.abs(high_freq_fft)
#    temp_powers = [i for i in power_new if i != 0]
#    temp_power = [] 
#    [temp_power.append(x) for x in temp_powers if x not in temp_power]
    for i in range(len(wavelength)):
        temp = eqn(filtered_signals[i], wavelength[i][0], time_vec, ph[i][0], amp[i][0])
        equations.append(temp + filtered_signals_mean)
    return equations, power_new

def predict_future(new_time_vec, equations, filtered_signals, wavelength, ph, amp):
    # predict future for each equation
    pred = []
#    lines = []
    for i in range(len(equations)):
        pred.append(eqn(filtered_signals[i], wavelength[i][0], new_time_vec, ph[i][0], amp[i][0]))
#        line = np.empty(len(time_vec))
#        line.fill(np.mean(abs(filtered_signals[i].real)))
#        lines.append(line)
    return pred


pred_list = []
sig = created_equation
sig_mean = np.mean(sig)
sig_fft, power, sample_freq = fft(sig)
peak_freqs, pos_power, freqs = find_peak_freqs(sample_freq, power, n=len(sig))
filtered_signals = create_n_signals(sig_mean, sig, sig_fft, peak_freqs, sample_freq)
filtered_sig, high_freq_fft = create_combined_filtered_sig(sig_mean, sig, sig_fft, peak_freqs, 
                                                           sample_freq)

'''my equation generator is wrong. The phase and or wavelength might be wrong as well.'''
''' IMPORTANT... my sum of waves is fine, mostly. It is the equation generator for each individual
    signal that needs fixed. Once fixed, summing should work '''
    
ph, wavelength, amp = find_phases_wavelengths_amps(filtered_signals, high_freq_fft, peak_freqs, len(sig))
# ph = []
# for i in range(len(high_freq_fft)):
#     ph.append(np.angle(high_freq_fft[i]))
filtered_signals_mean = np.abs(np.mean(filtered_signals))
equations, power_new = find_equations(filtered_signals_mean, high_freq_fft, wavelength, 
                                      filtered_signals, time_vec, ph, amp)
total_equations = sum(equations)
pred_x = np.arange(time_vec[-1] + 1, time_vec[-1] + 2)
pred_x = np.insert(pred_x, 0, time_vec[-1])
predicted_curves = predict_future(pred_x, equations, filtered_signals, wavelength, ph, amp)
total_pred = sum(predicted_curves) + filtered_signals_mean
dif = sig[-1] - total_pred[0]
total_pred = total_pred + dif
total_pred = np.array([0 if j < 0 else j for j in total_pred])
total_pred = total_pred[1:]

axs[0].plot(time_vec, filtered_sig, label='filtered_sig_combined')
axs[1].plot(time_vec, total_equations, label='total_equations')
axs[1].plot(time_vec, created_equation, label='total Signal')
axs[2].plot(pred_x, predicted_curves[-1] + np.mean(filtered_signals[-1]), linewidth=3, label='pred[-1]')
axs[2].plot(time_vec, filtered_signals[-1], label='filtered_sig[-1]')
axs[2].plot(time_vec, equations[-1], label='equations[-1]')
# axs[2].plot(time_vec, filtered_signals[-2])
# axs[2].plot(time_vec, filtered_signals[-3])
# axs[2].plot(time_vec, filtered_signals[-4])

axs[0].legend()
axs[1].legend()
axs[2].legend()
fig.tight_layout()
plt.show()  


