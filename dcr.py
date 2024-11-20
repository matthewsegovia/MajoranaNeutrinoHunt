import numpy as np

def find_dcr(waveform):

    # Find peak index value
    peak_idx = np.argmax(waveform)

    # Find peak value
    peak_val = int(waveform[peak_idx])

    # We are only looking at the data after the peak
    data_after_peak = waveform[peak_idx:]

    # Get all time indices between peak and end of the time series
    time_indices = np.arange(peak_idx, len(waveform))

    # Calculate DCR region
    area_above_tail_slope = np.trapezoid(peak_val - data_after_peak, x=time_indices)

    return area_above_tail_slope
