# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 21:18:10 2021

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
number_of_intervals = 1 # this is the number of random intervals to be created and tested
remove_trendline_bool = True

# creating x and y arrays
y_array = np.array(df1.Close)
x_array = np.arange(1, len(y_array) + 1)
section_data_all_time = [x_array, y_array]

# separate data into test and validate sets for x and y and for each interval
def create_test_validate_sets(data, interval_to_predict):
    data_x = data[0]
    data_y = data[1]
    if interval_to_predict * 3 >= len(data_x):
        interval_to_predict = 1
    test_x = data_x[:-interval_to_predict]
    val_x = data_x[-interval_to_predict:]
    test_y = data_y[:-interval_to_predict]
    val_y = data_y[-interval_to_predict:]
    return test_x, test_y, val_x, val_y

# create test and validation set
test_x, test_y, val_x, val_y = create_test_validate_sets(section_data_all_time, predict_interval)

'''------------------------------------------------------------------------------------------------'''
'''-------------------------------------- GET INTERVAL DATA ---------------------------------------'''
'''-----------THIS WILL SECTION DATA INTO RANDOM INTERVALS, quantity AMOUNT OF TIMES---------------'''
'''------------------------------------------------------------------------------------------------'''

quantity = number_of_intervals

def get_begin_index_randomly(quantity, y_test, predict_interval):
    # get a random starting position with the first 1/10th of data removed
    y_length = len(y_test)
    starting_index = round(y_length - (9*y_length / 10))
    array = np.arange(starting_index, y_length - predict_interval)
    random_starting_points = np.random.choice(array, size=quantity, replace=True)
    return random_starting_points
    
def get_interval_data(quantity, test_x, test_y, predict_interval):
    begin_indices = get_begin_index_randomly(quantity, test_y, predict_interval)
    all_processed_data = []
    list_of_ending_indices = []
    for i in range(len(begin_indices)):
        begin_index = begin_indices[i]
        end_index = begin_index + predict_interval
        list_of_ending_indices.append(end_index)
        new_test_y = test_y[:begin_index]
        new_test_x = test_x[:begin_index]
        new_val_y = test_y[begin_index:end_index]
        new_val_x = test_x[begin_index:end_index]
        new_processed_data = [new_test_x, new_test_y, new_val_x, new_val_y]
        all_processed_data.append(new_processed_data)
    return all_processed_data, list_of_ending_indices

processed_data, ending_indices = get_interval_data(quantity, test_x, test_y, predict_interval)

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
        peak_freqs.append(freqs[dominant_freq_indices[i][0]][0])
    # remove 1/f noise
    peak_freqs_temp = peak_freqs.copy()
    peak_freqs = [i for i in peak_freqs if round(1/i) < len(sample_freq)]
    if len(peak_freqs) < 1:
        peak_freqs = peak_freqs_temp
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
    for i in range(len(wavelength)):
        temp = eqn(filtered_signals[i], wavelength[i], time_vec, ph[i], amp[i])
        equations.append(temp + filtered_signals_mean)
    return equations, power_new

def predict_future(new_time_vec, equations, filtered_signals, wavelength, ph, amp):
    # predict future for each equation
    pred = []
    for i in range(len(equations)):
        pred.append(eqn(filtered_signals[i], wavelength[i], new_time_vec, ph[i][0], amp[i][0]))
    return pred

# Find a slope and intercept for linear trendline to be removed
def best_fit_slope_and_intercept(xs,ys):
    m = (((np.mean(xs)*np.mean(ys)) - np.mean(xs*ys)) /
         ((np.mean(xs)*np.mean(xs)) - np.mean(xs*xs)))
    b = np.mean(ys) - m*np.mean(xs)
    return m, b





# Section data into chunks to do analysis on. 
# 1. Create a much much better statistical measure for good vs bad. This is incredibly important.
# 2. Try doing long freq removal for entire series, then cut in half and do again, and again, and so on.
# 3. Try doing same as above except short freq removals.
# 4. Try 1 and 2 after removing trendlines for each step or just at the beginning.
# 5. Try doing long/short predictions for each interval and see if they can add up to a better solution.

def run_program(test_x, test_y, val_x, val_y, freqs_to_ignore_limit = 2, remove_trendline=True):
    if number_of_frequencies <= freqs_to_ignore_limit:
        freqs_to_ignore_limit = number_of_frequencies - 1
    if remove_trendline == True:
        slope, intercept = best_fit_slope_and_intercept(test_x, test_y)
        y_subtract = slope * np.arange(1, len(test_x) + 1) + intercept
        mean = np.average(y_subtract)
        y_subtracted = test_y - y_subtract + mean
        sig_fft, power, sample_freq = fft(y_subtracted, dt=1/len(y_subtracted))
    else:
        sig_fft, power, sample_freq = fft(test_y, dt=1/len(test_y))
        y_subtracted = test_y

    # # plot fft
    # fig, axs = plt.subplots(2, 1, figsize=(10,6))
    # fig.suptitle("Stock Market Time Series", fontsize=16)
    # axs[0].plot(test_x, test_y, label='test data')
    # axs[0].plot(val_x, val_y, label='val data')
    # axs[0].plot(test_x, y_subtract, label='Slope to be removed')
    # axs[0].plot(test_x, y_subtracted, label='Slope removed')
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
    # # plot signals
    # fig, axs = plt.subplots(3, 1, figsize=(12,8))
    # for i in range(len(strongest_signals)):
    #     axs[0].plot(test_x[:], strongest_signals[i], label="signal " + str(peak_freqs[i]) + ' Hz')
    # axs[1].plot(test_x, combined_signals, label='Combined Signals to be Used, Time Domain')
    # axs[2].plot(sample_freq, fft_combined_signals, label='Combined Signals to be Used, Frequency Domain')
    
    # axs[0].set_title("Strongest Frequencies (Time Domain)")
    # axs[1].set_title("Strongest Frequencies Combined (Time Domain)")
    # axs[2].set_title("Strongest Frequencies (Frequency Domain)")
    # if len(peak_freqs) < 5:
    #     axs[0].legend()
    # axs[1].legend()
    # axs[2].legend()
    # fig.tight_layout()
    # plt.show()   
    
    phase, wavelength, amp = find_phases_wavelengths_amps(strongest_signals, sig_fft, peak_freqs, sample_freq, len(test_y))
    strongest_signals_mean = np.abs(np.mean(strongest_signals))
    equations, power_new = find_equations(strongest_signals_mean, fft_combined_signals, wavelength, strongest_signals, test_x, phase, amp)
    new_val_x = np.insert(val_x, 0, val_x[0] - 1)
    predicted_curves = predict_future(new_val_x, equations, strongest_signals, wavelength, phase, amp)
    total_pred = sum(predicted_curves) + strongest_signals_mean
    dif = test_y[-1] - total_pred[0]
    total_pred = total_pred + dif
    total_pred = np.array([0 if j < 0 else j for j in total_pred])
    total_pred = total_pred[1:]
    if remove_trendline_bool == True:
        y_add = slope * np.arange(1, len(val_x) + 1)
        mean = np.average(y_add)
        total_pred = total_pred + y_add + mean
    # val_x = np.insert(val_x, 0, test_x[-1])
    # val_y = np.insert(val_y, 0, test_y[-1])
    # fig, axs = plt.subplots(1, 1, figsize=(10,4))
    # axs.plot(test_x, test_y, label='test data')
    # axs.plot(val_x, val_y, label='Actual')
    # axs.plot(val_x, total_pred, label='Predicted')
    # axs.set_title("Predicted vs Actual")
    # axs.legend()
    # fig.tight_layout()
    # plt.show()   
    
    return total_pred
    

# create 2 arrays, one for the short arrays and 1 for the long arrays. Then find average values for all predictions
def average_predictions_from_each_interval(predictions):
    short_arrays = []
    normal_arrays = []
    predictions_list = predictions.copy()
    for i in range(len(predictions_list) - 1):
        if len(predictions_list[i]) < len(predictions_list[-1]):
            short_arrays.append(predictions_list[i])
        elif len(predictions_list[i]) == len(predictions_list[-1]):
            normal_arrays.append(predictions_list[i])
    normal_arrays.append(predictions_list[-1])
    
    new_prediction = sum(normal_arrays) / 3
    short_arrays = short_arrays[0]
    for i in range(len(short_arrays)):
        new_prediction[i] = (new_prediction[i] + short_arrays[i]) / 2
    return new_prediction

def place_weights_on_each_prediction_and_sum(predictions):
    weights = [0, .1, .3, .6]
    predictions_list = predictions.copy()
    short_arrays = []
    normal_arrays = []
    for i in range(len(predictions_list) - 1):
        if len(predictions_list[i]) < len(predictions_list[-1]):
            short_arrays.append(predictions_list[i] * weights[i])
        elif len(predictions_list[i]) == len(predictions_list[-1]):
            normal_arrays.append(predictions_list[i] * weights[i])
    normal_arrays.append(predictions_list[-1] * weights[-1])
    new_prediction = sum(normal_arrays)
    short_arrays = short_arrays[0]
    for i in range(len(short_arrays)):
        new_prediction[i] = new_prediction[i] + (short_arrays[i] * weights[i])
    return new_prediction

# weighted_prediction = place_weights_on_each_prediction_and_sum(predictions)
# avg_prediction = average_predictions_from_each_interval(predictions)
# section_data_ten_years_val_x = np.insert(section_data_ten_years_val_x, 0, section_data_ten_years_test_x[-1])
# section_data_ten_years_val_y = np.insert(section_data_ten_years_val_y, 0, section_data_ten_years_test_y[-1])

# fig, axs = plt.subplots(1, 1, figsize=(10,4))
# axs.plot(section_data_ten_years_test_x, section_data_ten_years_test_y, label='test data')
# axs.plot(section_data_ten_years_val_x, section_data_ten_years_val_y, label='Actual')
# axs.plot(section_data_ten_years_val_x, avg_prediction, label='average predicted')
# axs.plot(section_data_ten_years_val_x, weighted_prediction, label='weighted average predicted')
# axs.set_title("Predicted vs Actual")
# axs.legend()
# fig.tight_layout()
# plt.show()   

'''-------------------------------------- GET ERRORS -----------------------------------'''

def getRMSE(actual, predicted):
    length = len(actual)
    result_list = []
    for i in range(length):
        difference = predicted[i] - actual[i]
        result = (difference * difference)
        result_list.append(result)
    summation = np.sum(result_list)
    final = np.sqrt(summation / length)
    return final
def correct_direction(slope_a, slope_b):
    global slopes_a
    # 0=none, 1=not bad but not good, 2=good
    dif = np.abs(slope_a - slope_b)
    direction = "error"
    slopes_a.append(slope_a)
    if slope_a >= .4:
        if slope_b <= 0:
            direction = 0
        elif slope_b > 0 and slope_b < .2:
            direction = 1
        else:
            direction = 2
    elif slope_a > -.4 and slope_a < .4:
        if slope_a < -.1:
            if slope_b < 0:
                direction = 2
            elif slope_b < .1 and slope_b > 0:
                direction = 1
            else:
                direction = 0
        elif slope_a > -.1 and slope_a < .1:
            if slope_b < -.2:
                direction = 1
            elif slope_b < .2 and slope_b > -.2:
                direction = 2
            else:
                direction = 1       
        else:
            if slope_b > .1:
                direction = 2
            elif slope_b < .1 and slope_b > -.1:
                direction = 1
            else:
                direction = 0
        if slope_a < 0 and slope_b > 0 and dif > .2:
            direction = 0
        elif slope_a > 0 and slope_b < 0 and dif > .2:
            direction = 0

    elif slope_a <= -.4:
        if slope_b <= -.2:
            direction = 2
        elif slope_b > -.2 and slope_b < 0:
            direction = 1
        else:
            direction = 0
    return direction

def get_area_enclosed_by_2_curves(x, y1, y2):
    #x, y1, y2 must all be same length
    area_total_list = []
    for i in range(len(x) - 1): 
        a = i
        b = a+1
        new_x = np.array([x[a], x[b]])
        new_y1 = np.array([y1[a], y1[b]])
        new_y2 = np.array([y2[a], y2[b]])
        y1_m, y1_b = best_fit_slope_and_intercept(new_x, new_y1)
        y2_m, y2_b = best_fit_slope_and_intercept(new_x, new_y2)
        # find intersection which is when y1 = y2 solve for x_intersection
        if y2_m - y1_m == 0:
            # this will be inf because it does not intersect. In this case use new_x[1] as x_intercept
            x_intersection = new_x[1]
        else:
            x_intersection = (y1_b - y2_b) / (y2_m - y1_m)
        if new_x[0] == x_intersection or (x_intersection <= new_x[0] or x_intersection >= new_x[1]):
            x_intersection = new_x[1]
        y1_integration_equation = lambda x: (y1_m*x + y1_b)
        y2_integration_equation = lambda x: (y2_m*x + y2_b)
        area_y1 = integrate.quad(y1_integration_equation, new_x[0], x_intersection)
        area_y2 = integrate.quad(y2_integration_equation, new_x[0], x_intersection)
        area = np.abs(area_y1[0] - area_y2[0])
        
        if x_intersection == new_x[0] or x_intersection == new_x[1]:
            # intersects but at edge or outside of range
            if new_x[0] == x_intersection:
                x_intersection = new_x[1]
        else:
            # intersection in middle somewhere
            area_y1_2 = integrate.quad(y1_integration_equation, x_intersection, new_x[1])
            area_y2_2 = integrate.quad(y2_integration_equation, x_intersection, new_x[1])
            area = area + np.abs(area_y1_2[0] - area_y2_2[0])
        area_total_list.append(area)
        
    area_total = sum(area_total_list)
    return area_total

def get_area_stats(area_total, actual):
    tot = sum(actual)
    range_ = max(actual) - min(actual)
    average_area = area_total / len(actual)
    better_percent = average_area / range_
    percent = area_total / tot
    return percent, better_percent

# calculate a metric for how good/bad the prediction 
def get_true_error(test_x, test_y, val_x, val_y, prediction):
    slope_actual, not_needed = best_fit_slope_and_intercept(val_x, val_y)
    slope_predicted, not_needed = best_fit_slope_and_intercept(val_x, prediction)
    error_my_metric = correct_direction(slope_actual, slope_predicted)
    error_rmse = getRMSE(val_y, prediction)
    error_all = [error_my_metric, np.round(error_rmse, decimals=3)]
    return error_all

'''-----------------------------------------------------------------------------------------------'''
'''----------------------------------------SCRIPT RUN STUFF---------------------------------------'''
'''-----------------------------------------------------------------------------------------------'''


slopes_a = []
# loop through array to run program for these time intervals
predictions = []
errors = []
for i in range(len(processed_data)):
    test_x, test_y, val_x, val_y = processed_data[i][0], processed_data[i][1], processed_data[i][2], processed_data[i][3]

    number_of_frequencies = len(test_x)
    # number_of_frequencies = 10
    prediction = run_program(test_x, test_y, val_x, val_y)
    val_x = np.insert(val_x, 0, test_x[-1])
    val_y = np.insert(val_y, 0, test_y[-1])
    prediction = np.insert(prediction, 0, test_y[-1])

    error = get_true_error(test_x, test_y, val_x, val_y, prediction)
    predictions.append(prediction)
    errors.append(error)

    fig, axs = plt.subplots(2, 1, figsize=(10,4))
    axs[0].plot(test_x, test_y, label='test data')
    axs[0].plot(val_x, val_y, label='Actual')
    axs[0].plot(val_x, prediction, label='Predicted')
    axs[1].plot(val_x, val_y, label='Actual')
    axs[1].plot(val_x, prediction, label='Predicted')
    axs[0].set_title("Predicted vs Actual")
    axs[1].set_title("Error is " + str(error))
    axs[0].legend()
    axs[1].legend()
    fig.tight_layout()
    plt.show()   

# get indexes with 2 for my error
indices_0 = []
indices_1 = []
indices_2 = []
best_intervals = []
worst_intervals = []
for i in range(len(errors)):
    if errors[i][0] == 2:
        best_intervals.append(len(processed_data[i][0]))
        indices_2.append(i)
    elif errors[i][0] == 1:
        indices_1.append(i)
    else:
        worst_intervals.append(len(processed_data[i][0]))
        indices_0.append(i)
        
        

# df_error = pd.DataFrame(errors)
# filename = "errors"
# df_error.to_csv("C:/Python_Scripts/Stock_Predictions/errors/" + filename + ".csv", index=True)

# print(np.mean(np.abs(slopes_a)))
# print(np.mean([i[0] for i in errors]))
# print(np.mean([i[1] for i in errors]))