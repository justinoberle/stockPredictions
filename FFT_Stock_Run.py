# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 16:10:11 2021

@author: justinoberle
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import fftpack
from scipy import integrate

'''---------------------------------------------------------------------------------------------'''
'''--------------------IMPORT DATA AND PRE-PROCESS INTO TEST/VAL SETS FOR X AND Y---------------'''
'''---------------------------------------------------------------------------------------------'''

# import data with subtracted mean option. create x and y arrays
df1 = pd.read_csv("C:/Python_Scripts/Stock_Predictions/processed_data/AMD_3_30_21_processed_df.csv")
predict_interval = 1
remove_trendline_bool = True

# creating x and y arrays
y_array = np.array(df1.Close)[:-20]
x_array = np.arange(1, len(y_array) + 1)
section_data_all_time = [x_array, y_array]

'''------------------------------------------------------------------------------------------------'''
'''------------------------------ PROCESS DATA, FFT, IFFT ETC. ------------------------------------'''
'''------------------------------------------------------------------------------------------------'''

def fft(sig, dt = 1):
    sig_fft = fftpack.fft(sig)
    sig_fft = sig_fft[:]
    power = np.abs(sig_fft)
    sample_freq = fftpack.fftfreq(sig.size, d=dt)
    sample_freq = sample_freq[:]
    return sig_fft, power, sample_freq

# find n most dominent frequencies
def find_peak_freqs(sample_freq, power, n=2):
    pos_mask = np.where(sample_freq > 0)
    freqs = sample_freq[pos_mask]
    pos_power = power[pos_mask]
    pos_power_sorted = sorted(pos_power)
    dominent = pos_power_sorted[-n:]
#    print(dominent)
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
    # loop through peak_freqs, do ifft and get each signal as filtered_individual_signals
    filtered_individual_signals = []
    individual_signals_removed = []
    for i in range(len(peak_freqs)):
        high_freq_fft = sig_fft.copy()
        high_freq_fft2 = sig_fft.copy()
        high_freq_fft[np.abs(sample_freq) > peak_freqs[i]] = 0
        high_freq_fft[np.abs(sample_freq) < peak_freqs[i]] = 0
        high_freq_fft2[np.abs(sample_freq) < peak_freqs[i]] = 0
        filtered_sig = fftpack.ifft(high_freq_fft)
        filtered_sig2 = fftpack.ifft(high_freq_fft2)
        filtered_sig += sig_mean
        filtered_sig2 += sig_mean
        filtered_individual_signals.append(filtered_sig)
        individual_signals_removed.append(filtered_sig2)
    return filtered_individual_signals, individual_signals_removed 

def create_combined_filtered_sig(sig_mean, sig, sig_fft, peak_freqs, sample_freq):
    # zero out every frequency that is not in peak frequencies, then inv FFT and only those are left
    # This is only used in plotting things for verification. Don't use for predictions only as it will slow down performance
    high_freq_fft = np.empty(len(sig_fft), dtype='complex128')
    high_freq_fft.fill(0)
    peak_freqs = [i for i in peak_freqs]
    for j in range(len(high_freq_fft)):
        if np.abs(sample_freq[j]) in peak_freqs:
            high_freq_fft[j] = sig_fft[j]
        else:
            pass
    filtered_sig = fftpack.ifft(high_freq_fft)
    filtered_sig += sig_mean
    return filtered_sig, high_freq_fft 

def find_phases_wavelengths_amps(filtered_signals, high_freq_fft, peak_freqs, sample_freq, sig_size):
    ''' need to find where filtered_sig crosses mean and is increasing then use this index as phase start '''
    # do this for each frequency to find array of phases and wavelengths
    wavelength = []
    ph = []
    amp = []
    indices = []
    for j in range(len(peak_freqs)):
        wavelength.append(1 / peak_freqs[j])
        indices.append(int(sig_size * peak_freqs[j]))
        sample_index = np.where(sample_freq==peak_freqs[j])
        phase = np.arctan2(high_freq_fft[sample_index].imag, high_freq_fft[sample_index].real)
        formula = ((((wavelength[j]) / 2) - 2) * np.pi) / wavelength[j]
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
        temp = eqn(filtered_signals[i], wavelength[i], time_vec, ph[i], amp[i])
        equations.append(temp + filtered_signals_mean)
    return equations, power_new

def predict_future(new_time_vec, equations, filtered_signals, wavelength, ph, amp):
    # predict future for each equation
    pred = []
    for i in range(len(equations)):
        pred.append(eqn(filtered_signals[i], wavelength[i], new_time_vec, ph[i], amp[i]))
    return pred

# Find a slope and intercept for linear trendline to be removed
def best_fit_slope_and_intercept(xs,ys):
    m = (((np.mean(xs)*np.mean(ys)) - np.mean(xs*ys)) /
         ((np.mean(xs)*np.mean(xs)) - np.mean(xs*xs)))
    b = np.mean(ys) - m*np.mean(xs)
    return m, b

def create_equation_from_sum_of_sins(freq_domain, sample_freq, time_vec):
    phases = []
    amplitudes = []
    wavelengths = []
    new_equations = []
    sum_of_freqs_real = sum(freq_domain[:50].real)
    sum_of_freqs_imag = sum(freq_domain[:50].imag)
    for i in range(len(freq_domain)):
        if sample_freq[i] > 0:
            phase = np.arctan(freq_domain[i].imag / freq_domain[i].real) + np.pi / 2
            amp = np.sqrt((freq_domain[i].real * freq_domain[i].real) + (freq_domain[i].imag * freq_domain[i].imag))
            wavelength = np.ptp(time_vec) / sample_freq[i]
            new_equation = amp * (np.sin(((2 * np.pi * time_vec) / wavelength) + phase))
            phases.append(phase)
            amplitudes.append(amp)
            wavelengths.append(wavelength)
            new_equations.append(new_equation)
    phase = np.arctan(sum_of_freqs_imag / sum_of_freqs_real)
    equ = (sum(new_equations)) / len(new_equations)  
    return equ

def run_program(test_x, test_y, val_x, freqs_to_ignore_limit = 0, remove_trendline=True):
    if number_of_frequencies <= freqs_to_ignore_limit:
        freqs_to_ignore_limit = number_of_frequencies - 1
    if remove_trendline == True:
        slope, intercept = best_fit_slope_and_intercept(test_x, test_y)
        y_subtract = slope * np.arange(1, len(test_x) + 1) + intercept
        mean = np.average(y_subtract)
        y_subtracted = test_y - y_subtract + mean
        sig_fft, power, sample_freq = fft(y_subtracted, dt=1)
    else:
        sig_fft, power, sample_freq = fft(test_y, dt=1)
        y_subtracted = test_y

    # # plot fft
    # fig, axs = plt.subplots(2, 1, figsize=(10,6))
    # fig.suptitle("Stock Market Time Series", fontsize=16)
    # axs[0].plot(test_x, test_y, label='test data')
    # # axs[0].plot(test_x, y_subtract, label='Slope to be removed')
    # # axs[0].plot(test_x, y_subtracted, label='Slope removed')
    # axs[0].set_title("Time Domain Data")
    # axs[1].plot(sample_freq, power, label='FFT Data')
    # axs[1].set_title("Frequency Domain Data")
    # axs[0].legend()
    # axs[1].legend()
    # fig.tight_layout()
    # plt.show()    
    
    peak_freqs, pos_power, freqs = find_peak_freqs(sample_freq, power, n=number_of_frequencies)
    peak_freqs = [i for i in peak_freqs if i > freqs_to_ignore_limit]
    sig_mean = np.mean(test_y)
    strongest_signals, strongest_signals_removed = create_n_signals(sig_mean, test_y, sig_fft, peak_freqs, sample_freq)
    combined_signals, fft_combined_signals = create_combined_filtered_sig(sig_mean, test_y, sig_fft, peak_freqs, sample_freq)
    phase, wavelength, amp = find_phases_wavelengths_amps(strongest_signals, sig_fft, peak_freqs, sample_freq, len(test_y))
    # print(phase)
    # print(wavelength)
    strongest_signals_mean = np.abs(np.mean(strongest_signals))
    # wavelength = [i * len(wavelength) for i in wavelength]
    # print(wavelength)
    equations, power_new = find_equations(strongest_signals_mean, sig_fft, wavelength, 
                                      strongest_signals, test_x, phase, amp)
    total_equations = sum(equations)
    dif = test_y[0] - total_equations[0][0]
    total_equations = total_equations + dif
    new_val_x = np.insert(val_x, 0, val_x[0] - 1)
    predicted_curves = predict_future(new_val_x, equations, strongest_signals, wavelength, phase, amp)
    total_pred = sum(predicted_curves)
    dif = test_y[-temp_int-1] - total_pred[0][0]
    total_pred = total_pred + dif
    # total_pred = np.array([0 if j < 0 else j for j in total_pred])
    total_pred = total_pred[0][1:]
    if remove_trendline == True:
        y_add = slope * np.arange(1, len(val_x) + 1)
        mean = np.average(y_add)
        total_pred = total_pred + y_add + mean

    # total_equations2 = create_equation_from_sum_of_sins(sig_fft, sample_freq, test_x)
    # print(total_equations)
    # print(test_y)
    val_x = np.insert(val_x, 0, x_array[-temp_int] - 1)
    # plot signals
    fig, axs = plt.subplots(3, 1, figsize=(12,8))
    # for i in range(len(strongest_signals)):
    #     axs[0].plot(test_x[:], strongest_signals[i], label="signal " + str(peak_freqs[i]) + ' Hz')
    #     axs[0].plot(val_x[:], predicted_curves[i] + np.mean(strongest_signals[i]), label="pred " + str(peak_freqs[i]) + ' Hz')
    plot_index = -1
    axs[0].plot(test_x, test_y, label="signal " + str(peak_freqs[plot_index]) + ' Hz')
    axs[0].plot(test_x, total_equations[0], label="tot_eq " + str(peak_freqs[plot_index]) + ' Hz')
    axs[1].plot(test_x, strongest_signals[plot_index].real, label="signal " + str(peak_freqs[plot_index]) + ' Hz')
    axs[1].plot(test_x, equations[plot_index][0], label="equations " + str(peak_freqs[plot_index]) + ' Hz')
    axs[1].plot(val_x, predicted_curves[plot_index][0] + np.mean(strongest_signals[plot_index].real), label="pred " + str(peak_freqs[plot_index]) + ' Hz')
    axs[2].plot(sample_freq, fft_combined_signals.real, label='Combined Signals to be Used, Frequency Domain')
    
    axs[0].set_title("Strongest Frequencies (Time Domain)")
    axs[1].set_title("Strongest Frequencies Combined (Time Domain)")
    axs[2].set_title("Strongest Frequencies (Frequency Domain)")
    # if len(peak_freqs) < 5:
    #     axs[0].legend()
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    fig.tight_layout()
    plt.show()   
    
    return total_pred

'''-----------------------------------------------------------------------------------------------'''
'''----------------------------------------SCRIPT RUN STUFF---------------------------------------'''
'''-----------------------------------------------------------------------------------------------'''
# trying to predict this last timestep and for some reason it is not right.
# prediction[1] is wrooooong
temp_int = 2
start = x_array[-temp_int:-temp_int+1]
end = start + predict_interval
val_x = np.arange(start, end)
val_y = [y_array[-temp_int-1], y_array[-temp_int]]
# print(val_x)
# print(val_y)
# number_of_frequencies = 200
number_of_frequencies = len(x_array)
prediction = run_program(x_array, y_array, val_x, remove_trendline=remove_trendline_bool)
val_x = np.insert(val_x, 0, x_array[-temp_int] - 1)
# val_x += 1
prediction = np.insert(prediction, 0, y_array[-temp_int-1])

fig, axs = plt.subplots(2, 1, figsize=(10,4))
axs[0].plot(x_array, y_array, label='test data')
axs[0].plot(val_x, prediction, label='Predicted')
axs[1].plot(val_x, prediction, label='Predicted')
axs[1].plot(val_x, val_y, label='val_y')
axs[0].set_title("Predicted vs Actual")
axs[0].legend()
axs[1].legend()
fig.tight_layout()
plt.show()   

# # actual prediction with NO val data
# number_of_frequencies = len(x_array)
# # number_of_frequencies = 100
# index_start = -10
# prediction_x = np.arange(x_array[index_start] + 1, x_array[index_start] + 2)
# prediction_y = run_program(x_array, y_array, prediction_x, remove_trendline=remove_trendline_bool)
# prediction_x = np.insert(prediction_x, 0, x_array[index_start])
# prediction_y = np.insert(prediction_y, 0, y_array[index_start])

# fig, axs = plt.subplots(2, 1, figsize=(10,4))
# axs[0].plot(x_array, y_array, label='test data')
# axs[0].plot(prediction_x, prediction_y, label='Predicted')
# axs[1].plot(prediction_x, prediction_y, label='Predicted')
# axs[0].set_title("Predicted vs Actual")
# axs[0].legend()
# axs[1].legend()
# fig.tight_layout()
# plt.show()  

# profit = prediction[1] - prediction[0]
# print(prediction[0])
# print(profit)