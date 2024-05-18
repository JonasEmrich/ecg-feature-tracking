# Providing helper functions for loading ECG data, plotting and peak correction.
import wfdb
import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt

def load_ecg(record_name="1", lead='i', filtered=False, left=0, right=None):
    """
    Function to load an (example) ECG record from the LUDB.

    Params:
        record_name (str): The name of the ECG record to load.
        lead (str): The lead of the ECG record to load.

    Returns:
        time (ndarray): Array containing the time values of the ECG signal.
        ecg (ndarray): Array containing the preprocessed ECG signal.
        fs (int): The sampling frequency of the ECG signal.
        annotations (dict): Dictionary containing the annotations of the ECG signal.

    """

    def _get_peak_on_off(a, symbols=["N"]):
        ''' Auxiliary function to extract  on/peak/offset of annotations given the following format: ´["(", "N", ")", "(", "N"...]´ '''
        a_sym = np.array(a.symbol)

        # find peak and its on/offset
        peak_ix = np.where(np.in1d(a_sym, symbols))[0]
        on_ix = peak_ix - 1
        off_ix = peak_ix + 1

        # check bounds
        off_ix = off_ix[off_ix < len(a_sym)]
        on_ix = on_ix[on_ix >= 0]

        # check if it is really on/offset == "(" or ")"
        on = a.sample[on_ix[a_sym[on_ix] == "("]] - left
        off = a.sample[off_ix[a_sym[off_ix] == ")"]] - left
        peaks = a.sample[peak_ix] - left

        return on, peaks, off

    # Load the ECG data
    record = wfdb.rdsamp(record_name, pn_dir="qtdb/1.0.0/", sampfrom=left, sampto=right)
    lead_index = record[1]['sig_name'].index(lead)
    ecg = record[0][:, lead_index]
    fs = record[1]["fs"]
    if right is None:
        time = np.arange(ecg.size) / fs
    else:
        time = np.arange(left, right) / fs

    # Clean the ECG
    if filtered:
        ecg = nk.ecg_clean(ecg, sampling_rate=fs)

    # Load annotations
    annotations = {}
    a = wfdb.rdann(record_name, pn_dir="qtdb/1.0.0/", extension="q1c", sampfrom=left, sampto=right)
    annotations["P_on"], annotations["P"], annotations["P_off"] = _get_peak_on_off(a, symbols=["p"])
    annotations["R_on"], annotations["R"], annotations["R_off"] = _get_peak_on_off(a, symbols=['N', 'R', 'A', 'B', 'Q'])
    annotations["T_on"], annotations["T"], annotations["T_off"] = _get_peak_on_off(a, symbols=["t"])

    return time, ecg, fs, annotations


############################
#### Plotting functions ####
############################

def plot_waves(time, ecg, waves, title=None, xlim=[1,8]):
    """
    Plots a single ECG lead with morphology waves.

    Params:
        time (array-like): Array of time indices.
        ecg (array-like): Array of ECG signal.
        waves (dict): Dictionary containing morphology waves.
        title (str, optional): Title of the plot.
    """

    # Plotting options
    markers = [
            {'marker': '^', 'linewidth': 0, 'markersize': 6, 'markeredgewidth': 1},
            {'marker': 'x', 'linewidth': 0, 'markersize': 6, 'markeredgewidth': 2},
            {'marker': 'v', 'linewidth': 0, 'markersize': 6, 'markeredgewidth': 1}]
    colors = ['tab:blue',  'tab:green', 'tab:red' ,'tab:orange', 'tab:brown', 'tab:purple']

    morphologies = ["P", "Q", "R", "S", "T"]
    types = ["_on", "", "_off"]

    # Plot the ECG
    plt.figure(figsize=(10, 4))
    plt.plot(time, ecg, lw=0.75, color=colors[0])
    # iterate over all morphology waves
    for i, w in enumerate(morphologies):
        for t, marker in zip(types, markers):
            if not f"{w}{t}" in waves.keys():
                continue
            plt.plot(time[waves[f"{w}{t}"]], ecg[waves[f"{w}{t}"]], **marker, label=f"{w}{t}", color=colors[i+1])

    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [mV]')
    plt.legend(loc='upper right', title='waves')
    plt.title(title)
    plt.grid(True)
    plt.xlim(xlim)
    plt.show()

def plot_interbeat(waves, fs, title=None, xlim=None, ylim=None):
    # Plotting options
    colors = ['tab:blue',  'tab:green', 'tab:red' ,'tab:orange', 'tab:brown', 'tab:purple']
    style = ['-', '--', ':']

    morphologies = ["P", "Q", "R", "S", "T"]
    types = ["_on", "", "_off"]

    # Plot the ECG
    plt.figure(figsize=(10, 4))
    # iterate over all morphology waves
    for i, w in enumerate(morphologies):
        for t, ls in zip(types, style):
            if not f"{w}{t}" in waves.keys():
                continue
            plt.plot(np.diff(waves[f"{w}{t}"])/fs, label=f"{w}{t}", color=colors[i+1], linestyle=ls)

    plt.xlabel('Beat/Complex')
    plt.ylabel('Interbeat Interval [ms]')
    plt.legend(loc='upper right', title='Waves', bbox_to_anchor=(1.13, 1))
    plt.title(title)
    plt.grid(True)
    xlim = xlim if xlim is not None else [0, len(waves["R"])-2]
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()

def plot_dict_as_interbeat(waves, fs, title=None, xlim=None, ylim=None):
    # Plot the ECG
    plt.figure(figsize=(10, 4))
    # iterate over all morphology waves
    for w, wave in waves.items():
        
            plt.plot(np.diff(wave)/fs, label=f"{w}")

    plt.xlabel('Beat/Complex')
    plt.ylabel('Interbeat Interval [ms]')
    plt.legend(loc='upper right', title='Waves', bbox_to_anchor=(1.13, 1))
    plt.title(title)
    plt.grid(True)
    xlim = xlim if xlim is not None else [0, len(list(waves.values())[0])-2]
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()

def plot_results(zs, mu, annotations, waves, fs):
    N = len(waves)
    fig, axs = plt.subplots(N, 1, figsize=(10, 2*N))
    for i in range(N):
        axs[i].plot(np.diff(zs[:, i]/fs, axis=0), label=f"Detection", color="tab:grey")
        axs[i].plot(np.diff(mu[:, i]/fs, axis=0), label=f"Kalman Corrected", color="tab:blue")
        axs[i].plot(np.diff(annotations[waves[i]]/fs, axis=0), '--', label=f"Annotations", color="tab:red")
        
        axs[i].set_title(f"{waves[i]}-wave")
        axs[i].set_xlabel("Beats/Complexes")
        axs[i].set_ylabel("Interbeat Interval (ms)")
        axs[i].grid(True)
        axs[i].set_xlim([0, len(annotations[waves[i]])-2])

    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.45), ncol=3, fancybox=True, shadow=False, frameon=False)


#########################
#### peak correction ####
#########################
def is_max(signal, point):
    if point == 0 or point >= len(signal) - 1:
        return False
    return signal[point] > signal[point-1] and signal[point] > signal[point+1]

def is_min(signal, point):
    if point == 0 or point >= len(signal) - 1:
        return False
    return signal[point] < signal[point-1] and signal[point] < signal[point+1]

def count_extrema(signal, points):
    count = 0
    for p in points:
        if is_max(signal, p):
            count += 1
        if is_min(signal, p):
            count -= 1
    return count

def correct_to_local_extrema(signal, points, peak_type="both"):
    """ Corrects the given points to the nearest local maxima/minima in the signal."""
    corrected_points = []

    for i in range(len(points)):
        point = points[i]

        # check if point is at the beginning or end of the signal
        if point == 0 or point == len(signal) - 1:
            corrected_points.append(point)
            #print(point, "Point at the edge of the signal")
            continue

        # check if point is already local maximum
        if peak_type in ["max", "both"] and is_max(signal, point):
            corrected_points.append(point)
            #print(point, "already max")
            continue
        # check if point is already local minimum
        if peak_type in ["min", "both"] and is_min(signal, point):
            corrected_points.append(point)
            #print(point, "already min")
            continue

        # find nearest local maxima/minima
        left = point - 1
        right = point + 1
        while left >= 0 and right < len(signal):
            if peak_type in ["min", "both"] and is_min(signal, left):
                #print(left, "found min left")
                corrected_points.append(left)
                break
            if peak_type in ["min", "both"] and is_min(signal, right):
                #print(right, "found min right")
                corrected_points.append(right)
                break
            if peak_type in ["max", "both"] and is_max(signal, left):
                #print(left, "found max left")
                corrected_points.append(left)
                break
            if peak_type in ["max", "both"] and is_max(signal, right):
                #print(right, "found max right")
                corrected_points.append(right)
                break

            left -= 1
            right += 1

    return np.array(corrected_points)

def correct_waves_to_local_extrema(ecg, waves, peak_type):
    corrected_waves = {}
    for key, value in waves.items():
        corrected_waves[key] = correct_to_local_extrema(ecg, value, peak_type=peak_type)
    return corrected_waves

def correct_waves_to_major_extrema(ecg, waves, detected_waves):
    corrected_waves = {}
    for key, value in waves.items():
        peak_type = "max" if count_extrema(ecg, detected_waves[key]) > 0 else "both"
        peak_type = "min" if count_extrema(ecg, detected_waves[key]) < 0 else peak_type
        #print(key, peak_type)
        corrected_waves[key] = correct_to_local_extrema(ecg, value, peak_type=peak_type)
    return corrected_waves
