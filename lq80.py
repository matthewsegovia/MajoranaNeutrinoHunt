# %%
import h5py
import numpy as np
import pandas as pd


def LQ80(waveform, tp0):
    tp100 = np.argmax(waveform)
    tp100_val = waveform[tp100]
    tp_val = waveform[tp0] + (tp100_val * 0.8)
    tp = np.argmin(np.abs(waveform[tp0:] - tp_val))
    tp80 = tp + tp0
    threshold = waveform[tp80]
    crossing_index = np.where(waveform >= threshold)[0][-1]
    charge_after_80 = np.sum(waveform[crossing_index:])
    return charge_after_80
