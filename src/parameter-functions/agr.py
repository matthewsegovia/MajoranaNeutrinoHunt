import numpy as np
import pandas as pd

def area_growth_rate(waveform):
    tp0 = int(waveform[0])
    waveform = waveform[1:]
    tp100 = np.argmax(waveform)
    tp100_val = waveform[tp100]
    tp_val = (tp100_val * 0.8)
    tp80 = np.argmin(np.abs(waveform[tp0:tp100] - tp_val))
    threshold = waveform[tp80]
    crossing_index = tp80
    window_start = crossing_index
    window_end = np.argmax(waveform)

    actual_area = np.sum(waveform[window_start:window_end+1])
    average_tail = np.linspace(waveform[window_start], waveform[window_end], window_end - window_start + 1)
    average_area = np.sum(average_tail)

    AGR = actual_area - average_area
    return AGR
