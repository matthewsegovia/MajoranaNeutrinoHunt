import numpy as np
import pandas as pd


def LQ80(waveform, tp0):
    tp100 = np.argmax(waveform)
    tp100_val = waveform[tp100]
    tp_val = (tp100_val * 0.8)
    if tp0 < 0:
        tp0 = abs(tp0)  # Ensure tp0 is non-negative

    if tp100 <= tp0 or tp0 >= len(waveform):  # Check for invalid slice
        return 0  # Return 0 if no valid range exists

    # Compute the index where the waveform reaches 80% of the peak value
    tp_segment = waveform[tp0:tp100]
    if len(tp_segment) == 0:  # Double-check the segment length
        return 0
    tp80 = np.argmin(np.abs(waveform[tp0:tp100] - tp_val))
    charge_after_80 = np.sum(waveform[tp80:tp100])
    return charge_after_80
