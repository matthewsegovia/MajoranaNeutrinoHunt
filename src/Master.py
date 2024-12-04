# %%
# Imports
import os
import sys
import h5py
import pandas as pd
import numpy as np



# %%
# Get the base directory of the repository
repo_base_dir = os.getcwd()

# Import parameter extraction functions
from parameter_functions.tdrift import tdrift
from parameter_functions.rea import rea
from parameter_functions.dcr import find_dcr
from parameter_functions.peakandtailslope import extract_peak_and_tail_slope
from parameter_functions.current_amplitude import max_amplitude
from parameter_functions.fourier_lfpr import normalized_fourier, lfpr
from parameter_functions.lq80 import LQ80
from parameter_functions.agr import area_growth_rate
from parameter_functions.inflection import inflection_points
from parameter_functions.rising_edge import rising_edge_slope

# Get all HDF5 files in the data directory
data_dir = os.path.join(repo_base_dir, "data/") 
data_files = [f for f in os.listdir(str(data_dir)) if f.endswith(".hdf5")]

# %%
# Parameter extraction from all waveform data
results = []
for file in data_files:
    with h5py.File(file, 'r') as f:
        for wave in np.arange(f['raw_waveform'].shape[0]):
            # Needed labels
            waveform = f['raw_waveform'][wave]
            tp0 = f['tp0'][wave]
            id = f['id'][wave]

            # Parameter extraction
            tdriftVal = tdrift(waveform, tp0) # 99.9%, 50%, 10%
            reaVal = rea(waveform, tp0)
            dcrVal = find_dcr(waveform)
            peakandtailVal = extract_peak_and_tail_slope(waveform)
            max_amp = max_amplitude(waveform)
            lfprVal = lfpr(normalized_fourier(waveform, tp0)[0])
            lq80 = LQ80(waveform, tp0)
            agr = area_growth_rate(waveform)
            inflectionpts = inflection_points(waveform, tp0)
            res = rising_edge_slope(waveform, tp0)

            results.append({'id': id,
                'tdrift': tdriftVal[0],
                'tdrift50': tdriftVal[1],
                'tdrift10': tdriftVal[2],
                'rea': reaVal,
                'dcr': dcrVal,
                'peakindex': peakandtailVal[0],
                'peakvalue': peakandtailVal[1],
                'tailslope': peakandtailVal[2],
                'currentamp': max_amp,
                'lfpr': lfprVal,
                'lq80': lq80,
                'areagrowthrate': agr,
                'inflection point': inflectionpts,
                'risingedgeslope': res
            })
        df = pd.DataFrame(results)
    df.to_csv("results.csv", index=False)
