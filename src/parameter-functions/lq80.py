import numpy as np
import pandas as pd


def LQ80(waveform, tp0):
    tp100 = np.argmax(waveform)
    tp100_val = waveform[tp100]
    tp_val = (tp100_val * 0.8)
    tp80 = np.argmin(np.abs(waveform[tp0:tp100] - tp_val))
    charge_after_80 = np.sum(waveform[tp80:tp100])
    return charge_after_80
