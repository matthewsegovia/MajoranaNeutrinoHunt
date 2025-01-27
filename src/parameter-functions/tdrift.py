import numpy as np

# Function for tdrift extraction
def tdrift(waveform, tp0):
    peak = np.argmax(waveform)
    tdrift = (peak-tp0) * 0.999
    tdrift50 = (peak-tp0) * 0.5
    tdrift10 = (peak-tp0) * 0.1
    return tdrift, tdrift50, tdrift10
