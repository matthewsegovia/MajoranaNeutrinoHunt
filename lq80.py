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

def LQ80(waveform):
    tp80 = tp(waveform, .8)
    threshold = waveform[tp80]
    crossing_index = np.where(waveform >= threshold)[0][-1]
    charge_after_80 = np.sum(waveform[crossing_index:])
    return charge_after_80