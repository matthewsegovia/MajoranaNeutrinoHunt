# %% imports
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.stats import skew

def rsa(waveform, tp0):
    
    x = np.arange(waveform.shape[0])
    end = np.where(waveform == np.max(waveform))[0][0]

    interp_range = np.arange(tp0, end + 1, 1)
    interp = interpolate.interp1d(x, waveform, kind = 'linear')
    interp_vals = interp(interp_range)

    skewness = skew(interp_vals)
    return(skewness)


""" 
# %%
with h5py.File('/Users/marcosanchez/Library/Mobile Documents/com~apple~CloudDocs/FA24/DSC 180A/MJD_Train_0.hdf5', 'r') as f:

    raw_waveform = np.array(f['raw_waveform'])
    index = np.random.choice(raw_waveform.shape[0])

    start = f['tp0'][index]


    print(rsa(raw_waveform[index], start))
""" 