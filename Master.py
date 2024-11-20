# %%
# imports
import importlib
import os
import sys
import h5py
import pandas as pd
import numpy as np

sys.path.append("/Users/marcosanchez/MajoranaHunt/MajoranaNeutrinoHunt")

#import parameter extraction functions
from tdrift import tdrift
from rea import rea
from dcr import find_dcr
from peakandtailslope import extract_peak_and_tail_slope
from current_amplitude import max_amplitude
from fourier_lfpr import normalized_fourier, lfpr


#dirs
scripts = "/Users/marcosanchez/MajoranaHunt/MajoranaNeutrinoHunt"
data = "/Users/marcosanchez/MajoranaHunt"

data_files = [f for f in os.listdir(data) if f.endswith(".hdf5")]
script_files = [f for f in os.listdir(scripts) if f.endswith(".py")]


# %%
results = []

with h5py.File(data_files[0], 'r') as f:
    #needed labels
    waveform = f['raw_waveform'][0]
    tp0 = f['tp0'][0]
    id = f['id'][0]

    #parameter extraction
    tdriftVal = tdrift(waveform, tp0) #99, 50, 10
    reaVal = rea(waveform, tp0)
    dcrVal = find_dcr(waveform)
    peakandtailVal = extract_peak_and_tail_slope(waveform)
    max_amp = max_amplitude(waveform)
    lfprVal = lfpr(normalized_fourier(waveform, tp0)[0])


    results.append({'id': id,
        'tdrift': tdriftVal[0],
        'tdrift50': tdriftVal[1],
        'tdrift10': tdriftVal[2],
        'rea': reaVal,
        'dcr': dcrVal,
        'peakindex': peakandtailVal[0],
        'peakvalue': peakandtailVal[1],
        'tailslope': peakandtailVal[2],
        'maxamplitude': max_amp,
        'lfpr': lfprVal

    })
    df = pd.DataFrame(results)
    print(df)
