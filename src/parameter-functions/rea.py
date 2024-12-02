# %% imports
import numpy as np
from scipy import interpolate
from scipy.stats import skew


def rea(waveform, tp0):

    end = np.where(waveform == np.max(waveform))[0][0]

    interp_range = np.arange(tp0, end + 1, 1)
    interp = interpolate.interp1d(np.arange(waveform.shape[0]), waveform, kind = 'linear')
    interp_vals = interp(interp_range)

    return skew(interp_vals)
