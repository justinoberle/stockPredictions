# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 11:35:05 2018

@author: justinoberle
"""

import numpy as np
from scipy import fftpack
from matplotlib import pyplot as plt
import pandas as pd
import cProfile, pstats, io

''' a few things to fix, 
    1) equation lags filtered_signal by 1 timestep but only at higher frequencies!????
    This is possibly always 1 timestep behind but it isn't noticable at lower frequencies.
    PICK BACK UP HERE! Already fixed a major issue. Gonna rock kaggle's socks off tomorrow.
    2) do this in batches
    3) make this OOP
'''
load_data_bool = True

if load_data_bool == True:
    df1_new = pd.read_csv('C:/Python_Scripts/Stock_Predictions/processed_data/AAPL_Apple_processed_df.csv')
    df1_new = df1_new.T
#    best_n_freqs = pd.read_csv("C:/Python_Scripts/Kaggle/Processed_data/FFT_Results/best_n_freqs.csv", index_col='Unnamed: 0')
#    best_errors = pd.read_csv("C:/Python_Scripts/Kaggle/Processed_data/FFT_Results/best_errors_df.csv", index_col='Unnamed: 0')
#    best_n_freqs = best_n_freqs.to_numpy().flatten()
    
def profile(fnc):
    
    """A decorator that uses cProfile to profile a function"""
    
    def inner(*args, **kwargs):
        
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner

def get_row_x_y(df_row):    
    dt = 1    
    counter = 0
    # removing nans
    for i in df_row:
        if np.isnan(i) == True:
            counter += 1
            pass
        else:
            start_i = int(counter)
            break
    df_row = df_row[start_i:]
    time_vec = np.arange(start_i, len(df_row) + start_i, dt)
#    # check for large areas of 0's and use logic to decide whether to remove this area or not. Param are n_zeros=300, length of returned array must be >= 150.
#    count_zeros = 0
#    start_i_temp = start_i
#    trigger = False
#    for i in range(len(df_row)):
#        if df_row[i] == 0:
#            count_zeros += 1
#        else:
#            if count_zeros >= 600:
#                df_row_temp = df_row[i:]
#                time_vec_temp = time_vec[i:]
#                start_i_temp += i
#                trigger = True
#                break
#            else:
#                count_zeros = 0
#    if trigger == True:
#        if len(df_row_temp) <= 150 and len([i for i in df_row if i != 0]) >= 1:
#            pass
#        else:
#            df_row = df_row_temp
#            time_vec = time_vec_temp
#            start_i = start_i_temp
    sig = np.array([i if np.isnan(i) == False else 0 for i in df_row])
    return time_vec, sig, start_i

''' Do FFT and to get power and freq '''
def fft(sig, dt = 1):
    sig_fft = fftpack.fft(sig)
    power = np.abs(sig_fft)
    sample_freq = fftpack.fftfreq(sig.size, d=dt)
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
    # remove 1/f noise
    peak_freqs_temp = peak_freqs.copy()
    peak_freqs = [i for i in peak_freqs if round(1/i[0]) < len(sample_freq)]
    if len(peak_freqs) < 1:
        peak_freqs = peak_freqs_temp
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
            pass
    
    filtered_sig = fftpack.ifft(high_freq_fft)
    filtered_sig += sig_mean
    return filtered_sig, high_freq_fft

def find_phases_wavelengths(filtered_signals, start_i, peak_freqs):
    ''' need to find where filtered_sig crosses mean and is increasing then use this index as phase start '''
    # do this for each frequency to find array of phases and wavelengths
    phases = []
    wavelength = []
    ph = []
    for j in range(len(filtered_signals)):
        avg = np.mean(abs(filtered_signals[j].real))
        for i in range(len(filtered_signals[j])):
            if i == len(filtered_signals[j]) - 1:
                phase = i + start_i
                break
            if filtered_signals[j][i].real < avg and filtered_signals[j][i+1].real > avg:
                total_dif = filtered_signals[j][i+1].real - filtered_signals[j][i].real
                # make a percentage multiplier from 0-1 normalized and add it as remainder
                difference = avg - filtered_signals[j][i].real
                percentage = difference / total_dif                
                phase = i + start_i + percentage
                break
            else:
                pass
        phases.append(phase)
        wavelength.append(1 / peak_freqs[j])
        ph.append((phase * 2 * np.pi) / wavelength[j])
#    print(phases)
#    print(wavelength)
#    print(ph)
    return ph, wavelength
    
def eqn(filtered_si, wavelength, time_vec, phase):
    amp = np.max(np.abs(filtered_si)) - np.mean(np.abs(filtered_si))
    return amp * np.sin((2 * np.pi / wavelength * time_vec) - phase)# + np.mean(filtered_si)

def find_equations(filtered_signals_mean, high_freq_fft, wavelength, filtered_signals, time_vec, ph):
    # find equation for each frequency and place into a list of equations
    equations = []
    power_new = np.abs(high_freq_fft)
#    temp_powers = [i for i in power_new if i != 0]
#    temp_power = [] 
#    [temp_power.append(x) for x in temp_powers if x not in temp_power]
    for i in range(len(wavelength)):
        temp = eqn(filtered_signals[i], wavelength[i][0], time_vec, ph[i][0])
        equations.append(temp + filtered_signals_mean)
    return equations, power_new

def predict_future(new_time_vec, equations, filtered_signals, wavelength, ph):
    # predict future for each equation
    pred = []
#    lines = []
    for i in range(len(equations)):
        pred.append(eqn(filtered_signals[i], wavelength[i][0], new_time_vec, ph[i][0]))
#        line = np.empty(len(time_vec))
#        line.fill(np.mean(abs(filtered_signals[i].real)))
#        lines.append(line)
    return pred

#@profile
def run_program(df1_new, new_time_vec, trendline, *args, index=0, n=4, plot_bool=True):
    pred_list = []
    global end_index
    printer = 500
    data_size = df1_new.index.size
    for iterations in range(index, index + 1):
        df_row = df1_new.iloc[iterations]
        time_vec, sig, start_i = get_row_x_y(df_row)
        sig_mean = np.mean(sig)
        sig_fft, power, sample_freq = fft(sig)
        peak_freqs, pos_power, freqs = find_peak_freqs(sample_freq, power, n=n)
        filtered_signals = create_n_signals(sig_mean, sig, sig_fft, peak_freqs, sample_freq)
        filtered_signals_mean = np.abs(np.mean(filtered_signals))
        filtered_sig, high_freq_fft = create_combined_filtered_sig(sig_mean, sig, sig_fft, peak_freqs, 
                                                                   sample_freq)
        ph, wavelength = find_phases_wavelengths(filtered_signals, start_i, peak_freqs)
        equations, power_new = find_equations(filtered_signals_mean, high_freq_fft, wavelength, 
                                              filtered_signals, time_vec, ph)
        pred = predict_future(new_time_vec, equations, filtered_signals, wavelength, ph)

#        sum_of_waves = sum(equations) - (n * filtered_signals_mean) + np.abs(np.mean(filtered_signals[-1]))
#        total_signal = sum(filtered_signals) - np.mean(np.abs(sum(filtered_signals))) + filtered_signals_mean
#        sum_of_waves = np.array([0 if j < 0 else round(j) for j in sum_of_waves])
        total_pred = sum(pred) + filtered_signals_mean
        dif = sig[-1] - total_pred[0]
        total_pred = total_pred + dif
        total_pred = np.array([0 if j < 0 else round(j) for j in total_pred])
        new_time_vec_longer = np.insert(new_time_vec, 0, new_time_vec[0] - 1)
        total_pred = np.insert(total_pred, 0, sig[-1])
        if args:
            actual = np.array(args[0].iloc[index])
            actual = np.insert(actual, 0, sig[-1])

        pred_list.append(total_pred)
        if iterations == printer:
            print(f'Iteration {printer} of {data_size}')
            printer += 500
        
        sig_trendline = trendline[:len(sig)]
        total_pred_trendline = trendline[len(sig) - 1:end_index]
        if plot_bool == True:
            
            fig, axs = plt.subplots(3, 1, figsize=(12,8))
            fig.canvas.set_window_title(f'index{iterations}')
        
            axs[0].plot(new_time_vec_longer, total_pred, label='pred_sum')
            axs[0].plot(new_time_vec_longer, total_pred + total_pred_trendline, label='pred')
    #        axs[0].plot(time_vec, lines[0])
    #        axs[0].plot(time_vec, sum_of_waves, label='sum_of_waves')
            axs[0].plot(time_vec, sig, label='Original Signal')
            axs[0].plot(time_vec, sig + sig_trendline)
            if args:
                axs[0].plot(new_time_vec_longer, actual, label='Actual')
    #        axs[0].plot(time_vec, total_signal, linewidth=3, label='total Signal')
    #        axs[0].plot(time_vec, filtered_sig, linewidth=3, label='Filtered Signal')
            axs[0].set_title('High Frequency Removed')
            axs[0].set_xlabel('Time')
            axs[0].set_ylabel('Amplitude')
            axs[0].legend(loc='best')
            
            axs[1].plot(time_vec, filtered_signals[-1], label='filtered_signals[-1]')
            axs[1].plot(time_vec, filtered_signals[-2], label='filtered_signals[-2]')
#            axs[1].plot(time_vec, filtered_signals[-3], label='filtered_signals[-3]')
#            axs[1].plot(time_vec, filtered_signals[0], label='filtered_signals[0]')
#            axs[1].plot(time_vec, filtered_signals[1], label='filtered_signals[1]')
            # axs[1].plot(time_vec, equations[-1], label='equations[-1]')
            # axs[1].plot(time_vec, equations[-2], label='equations[-2]')
#            axs[1].plot(time_vec, equations[-3], label='equations[-3]')
            axs[1].plot(new_time_vec, pred[-1] + filtered_signals_mean, label='pred[-1]')
#            axs[1].plot(new_time_vec, pred[-2] + filtered_signals_mean, label='pred[-2]')
#            axs[1].plot(new_time_vec, pred[-3] + filtered_signals_mean, label='pred[-3]')
            axs[1].legend()
               
            axs[2].plot(sample_freq, power_new)
            axs[2].set_title('Power after High Frequency Removal')
            axs[2].set_xlabel('Frequency in Hz')
            axs[2].set_ylabel('Power')
            axs[2].set_xlim(-1, 1)
            
            fig.tight_layout()
            plt.show()    

    return pred_list

# Find a slope and intercept for linear trendline to be removed
def best_fit_slope_and_intercept(xs,ys):
    m = (((np.mean(xs)*np.mean(ys)) - np.mean(xs*ys)) /
         ((np.mean(xs)*np.mean(xs)) - np.mean(xs*xs)))
    
    b = np.mean(ys) - m*np.mean(xs)
    
    return m, b


df = pd.read_csv("C:/Python_Scripts/Stock_Predictions/processed_data/AAPL_Apple_processed_df.csv")
y_array = np.array(df.Close)
y_array = y_array[:]
x_array = np.arange(1, len(y_array) + 1)
sig_fft, power, sample_freq = fft(y_array, dt=1/len(y_array))


slope, intercept = best_fit_slope_and_intercept(x_array, y_array)
#slope = 0
# Remove the trendline, set slope to 0 if you do not want to remove the trendline. Things break when they go neg
mean = np.ceil(np.average(y_array))
y_subtract = (slope * x_array) - mean
y_subtracted = y_array - y_subtract
y_sub = pd.DataFrame(y_subtracted).T

def get_begin_index_randomly(quantity, y_sub, df1_new, interval):
    starting_index = round(y_sub.columns.size - y_sub.columns.size / 2)
    array = np.arange(starting_index, y_sub.columns.size - interval)
    random_starting_points = np.random.choice(array, size=quantity, replace=True)
    return random_starting_points
    
def get_interval_data(quantity, y_sub, df1_new):
    interval = 7
    begin_indices = get_begin_index_randomly(quantity, y_sub, df1_new, interval)
    list_of_df_x = []
    list_of_df_y = []
    list_of_ending_indices = []
    for i in range(len(begin_indices)):
        begin_index = begin_indices[i]
        end_index = begin_index + interval
        list_of_ending_indices.append(end_index)
        df1_new_test_x = y_sub[df1_new.columns[:begin_index]]
        df1_new_test_y = y_sub[df1_new.columns[begin_index:end_index]]
        list_of_df_x.append(df1_new_test_x)
        list_of_df_y.append(df1_new_test_y)
    
    return list_of_df_x, list_of_df_y, list_of_ending_indices
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
    # 0=none, 1=not bad but not good, 2=good
    dif = np.abs(slope_a - slope_b)
    if slope_a >= .4:
        if slope_b <= .2:
            direction = 0
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
        else:
            direction = 0
    return direction
        
    
def find_best_n_frequencies(): 
    global df1_new_test_x
    global df1_new_test_y
    global x_axis_for_plotting
    global y_subtract
    global actual_list
    global predicted
    global predicted_freqs_list
    errors = []
    ns = []
    actual = np.array(df1_new_test_y.iloc[0])
    actual_list.append(np.insert(actual, 0, np.array(df1_new_test_x.iloc[0])[-1]))
    predicted_freqs_list = []
    for i in range(2,3):      
        ns.append(i)
        predictions_list = run_program(df1_new_test_x, x_axis_for_plotting, y_subtract, df1_new_test_y, n=i, plot_bool=False)
        predicted = predictions_list[0]
        predicted_freqs_list.append(predicted)
        errors.append(getRMSE(actual, predicted))
    return errors, ns

def find_best_everything():
    global end_index
    global df1_new_test_x
    global df1_new_test_y
    global x_axis_for_plotting
    global y_subtract
    global predicted
    global actual_list
    
    number_of_tries = 1
    dfs_x, dfs_y, list_of_ending_indices = get_interval_data(number_of_tries, y_sub, df1_new)
    
    best_n_freqs_list = []
    errors_list = []
    predictions_list = []
    actual_list = []
    for i in range(number_of_tries): 
        end_index = list_of_ending_indices[i]
        df1_new_test_x = dfs_x[i]
        df1_new_test_y = dfs_y[i]
        begin = df1_new_test_x.columns.size
        end = begin + df1_new_test_y.columns.size
        x_axis_for_plotting = np.arange(begin, end)
        
        errors, ns = find_best_n_frequencies()
        predicted = predicted_freqs_list[errors.index(min(errors))]
        best_n_freqs = ns[errors.index(min(errors))]
        best_n_freqs_list.append(best_n_freqs)
        errors_list.append(min(errors))
#        print(min(errors))
        predictions_list.append(run_program(df1_new_test_x, x_axis_for_plotting, y_subtract, df1_new_test_y, n=5, plot_bool=True))
    return errors_list, best_n_freqs_list, predictions_list

errors_list, best_n_freqs_list, predictions_list = find_best_everything()
is_profitable = []
slopes_pred = []
slopes_act = []
x_pred = np.arange(1, len(actual_list[0]) + 1)
for i in range(len(predictions_list)):
    slope_actual, not_needed = best_fit_slope_and_intercept(x_pred, actual_list[i])
    slope_pred, not_needed = best_fit_slope_and_intercept(x_pred, predictions_list[i])
    slopes_pred.append(slope_pred)
    slopes_act.append(slope_actual)
    is_profitable.append(correct_direction(slope_actual, slope_pred))
average_error = np.average(errors_list)
#print(average_error)
#print(np.mean(is_profitable))

