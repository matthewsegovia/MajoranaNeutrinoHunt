import numpy as np
from scipy.stats import linregress


def extract_peak_and_tail_slope(waveform):
    """
    Extract the peak index, peak value, and the tail slope of the waveform.
    Tail slope is computed from the last 500 samples of the waveform.
    """
    peak_index = np.argmax(waveform)  # Find the peak index
    peak_value = waveform[peak_index]  # Find the peak value

    # Calculate the tail slope using the last 500 samples
    tail_segment = waveform[-500:]
    time = np.arange(len(tail_segment))  # Time indices for the last 500 samples
    slope, intercept, _, _, _ = linregress(time, tail_segment)

    return peak_index, peak_value, slope
