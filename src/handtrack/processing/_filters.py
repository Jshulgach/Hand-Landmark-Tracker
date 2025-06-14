import time
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def notch_filter(data, fs=4000, f0=60.0, Q=10, axis=1):
    """
    Applies a notch filter to the data to remove 60 Hz interference. Assumes data shape (n_channels, n_samples).
    A bandwidth of 10 Hz is recommended for 50 or 60 Hz notch filters; narrower bandwidths lead to
    poor time-domain properties with an extended ringing response to
    transient disturbances.

    Parameters:
        data (ndarray): Input data to be filtered.
        fs (int): Sampling frequency of the data.
        f0 (float): Frequency to be removed from the data (60 Hz).
        Q (int): Quality factor of the notch filter.

    Returns:
        nn.array:

    Example:
        out = notch_filter(signal_in, 30000, 60, 10);
    """
    b, a = iirnotch(f0, Q, fs)
    return filtfilt(b, a, data, axis=axis)


def lowpass_filter(data, cutoff, fs, order=4, axis=1):
    """
    Applies a lowpass filter to the data using a Butterworth filter.

    Parameters:
        data (ndarray): Input data to be filtered.
        cutoff (float): Cutoff frequency.
        fs (float): Sampling frequency of the data.
        order (int): Order of the filter.
        axis (int): Axis along which to apply the filter.

    Returns:
        ndarray: Filtered data.
    """
    b, a = butter(order, cutoff, btype="low", fs=fs)
    y = filtfilt(b, a, data, axis=axis)
    return y


def bandpass_filter(data, lowcut=10, highcut=500, fs=4000, order=4, axis=1, verbose=False):
    """
    Applies a bandpass filter to the data using a Butterworth filter.

    Parameters:
        data (ndarray): Input data to be filtered.
        lowcut (float): Low cutoff frequency.
        highcut (float): High cutoff frequency.
        fs (float): Sampling frequency of the data.
        order (int): Order of the filter.
        axis (int): Axis along which to apply the filter.
        verbose (bool): Whether to print filter parameters.

    Returns:
        ndarray: Filtered data.
    """
    b, a = butter(order, [lowcut, highcut], btype="bandpass", fs=fs)
    y = filtfilt(b, a, data, axis=axis)
    return y


def rectify(emg_data):
    """
    Rectifies EMG data by converting all values to their absolute values.

    Parameters:
        emg_data (numpy array): List of numpy arrays or pandas DataFrame items with filtered EMG data.

    Returns:
        rectified_data: List of rectified numpy arrays (same shape as input data).
    """
    return np.abs(emg_data)


def common_average_reference(emg_data, verbose=False):
    """
    Applies Common Average Referencing (CAR) to the multi-channel EMG data.

    Parameters:
        emg_data: 2D numpy array of shape (num_channels, num_samples).

    Returns:
        car_data: 2D numpy array after applying CAR (same shape as input).
    """
    if verbose:
        print("| Subtracting common average reference")
    # Compute the common average (mean across all channels at each time point)
    common_avg = np.mean(emg_data, axis=0)  # Shape: (num_samples,)

    # Subtract the common average from each channel
    car_data = emg_data - common_avg  # Broadcast subtraction across channels

    return car_data

def z_score_norm(data):
    """
    Apply z-score normalization to the input data.

    Parameters:
        data: 2D numpy array of shape (channels, samples).

    Returns:
        normalized_data: 2D numpy array of shape (channels, samples) after z-score normalization.
    """
    mean = np.mean(data, axis=1)[:, np.newaxis]
    std = np.std(data, axis=1)[:, np.newaxis]
    normalized_data = (data - mean) / std
    return normalized_data


# RMS (Root Mean Square)
def compute_rms(emg_window):
    """
    Compute the RMS of a given EMG window.

    Parameters:
        emg_window (np.ndarray): 1D numpy array representing the EMG window.

    Returns:
        float: RMS value of the EMG window.
    """
    return np.sqrt(np.mean(emg_window ** 2))