import numpy as np
import pandas as pd
from scipy.stats import linregress


def rising_edge_slope(waveform, tp0):
    tp100 = np.argmax(waveform)
    waveform = waveform[1:]
    time = np.arange(len(waveform[tp0:tp100]))
    data = waveform[tp0:tp100]
    if len(data) == 0:
        return np.nan
    slope, intercept, r_value, p_value, std_err = linregress(time, data)
    return slope
