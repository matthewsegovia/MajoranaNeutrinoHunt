import h5py
import numpy as np
import pandas as pd

def max_tp(waveform):
    tp100 = np.argmax(waveform)
    return tp100

def tp(waveform, percent, tp0):
    tp100 = max_tp(waveform)
    tp100_val = waveform[tp100]
    tp_val = waveform[tp0] + (tp100_val * percent)
    tp = np.argmin(np.abs(waveform[tp0:] - tp_val))
    tp = tp + tp0
    return tp

def inflection_points(waveform):
    tp100 = max_tp(waveform)
    tp80 = tp(waveform, .8)
    threshold = waveform[tp80]
    
    crossing_index = np.where(waveform >= threshold)[0][0]
    LQ80 = waveform[crossing_index:tp100]
    if len(LQ80) < 2:
        return 0
    num = tp100 - crossing_index
    x = crossing_index + np.arange(1, num + 1)
    second = np.gradient(np.gradient(LQ80))
    sign_changes = np.diff(np.sign(second))
    inflection_points = np.where((sign_changes == 2) | (sign_changes == -2))[0]
    return len(inflection_points)