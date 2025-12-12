import os
import numpy as np

from scipy.interpolate import interp1d


def interpolate_array_to_timebase(source_t, target_t, data, axis=0):
    """
    Interpolate any ndarray `data` sampled at `source_t` to match `target_t`.

    Args:
        source_t (np.ndarray): Original time vector.
        target_t (np.ndarray): Desired time vector.
        data (np.ndarray): Data to interpolate, shape (N, ...).
        axis (int): Axis along which time is aligned (default: 0).

    Returns:
        np.ndarray: Interpolated data with shape adjusted to match `target_t`.
    """

    data = np.moveaxis(data, axis, 0)  # move time axis to front
    shape_rest = data.shape[1:]
    interp_data = []

    #for i in range(data.shape[1:]):  # multidimensional slice
    for idx in np.ndindex(shape_rest):
        #flat_data = data[(slice(None),) + np.unravel_index(i, data.shape[1:])]
        flat_data = data[(slice(None),) + idx]
        # Flatten the data along the time axis
        flat_data = flat_data.reshape(-1)
        f = interp1d(source_t, flat_data, kind='linear', fill_value='extrapolate')
        interp_data.append(f(target_t))
        # Create interpolation function
        #f = interp1d(source_t, flat_data, kind='linear', fill_value='extrapolate')
        #interp_data.append(f(target_t))

    #result = np.stack(interp_data, axis=1)
    #result = result.reshape((len(target_t),) + data.shape[1:])
    #return np.moveaxis(result, 0, axis)  # restore original axis position
    interp_data = np.stack(interp_data, axis=-1).reshape((len(target_t),) + shape_rest)
    return np.moveaxis(interp_data, 0, axis)  # restore original axis

def interpolate_landmarks_to_emg(emg_t, lm_t, landmarks, sync_offset=0):
    """
    Interpolate landmark data to EMG time vector after applying sync offset.

    Args:
        emg_t (np.ndarray): EMG time vector
        lm_t (np.ndarray): Landmark time vector
        landmarks (np.ndarray): Landmark data, shape (N_frames, 21, 3)
        sync_offset (float): Time offset in seconds to align landmarks with EMG

    Returns:
        np.ndarray: Interpolated landmark data (N_emg_samples, 21, 3)
    """
    lm_t_shifted = lm_t + sync_offset
    lm_interp = np.zeros((emg_t.shape[0], landmarks.shape[1], landmarks.shape[2]))

    for joint_idx in range(landmarks.shape[1]):
        for dim in range(landmarks.shape[2]):
            interp_func = interp1d(lm_t_shifted, landmarks[:, joint_idx, dim], kind='linear', fill_value='extrapolate')
            lm_interp[:, joint_idx, dim] = interp_func(emg_t)

    return lm_interp


