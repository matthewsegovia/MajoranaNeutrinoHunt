import h5py
import numpy as np
import pandas as pd

def max_tp(waveform):
    tp100 = np.argmax(waveform)
    return tp100

def tp(waveform, percent):
    tp0 = int(waveform[0])
    waveform = waveform[1:]
    tp100 = max_tp(waveform)
    tp100_val = waveform[tp100]
    tp_val = waveform[tp0] + (tp100_val * percent)
    tp = np.argmin(np.abs(waveform[tp0:] - tp_val))
    tp = tp + tp0
    return tp

def area_growth_rate(waveform):
    tp80 = tp(waveform, .8)
    threshold = waveform[tp80]
    crossing_index = np.where(waveform >= threshold)[0][0]
    window_start = crossing_index
    window_end = np.argmax(waveform)

    actual_area = np.sum(waveform[window_start:window_end+1])
    average_tail = np.linspace(waveform[window_start], waveform[window_end], window_end - window_start + 1)
    average_area = np.sum(average_tail)

    AGR = actual_area - average_area
    return AGR