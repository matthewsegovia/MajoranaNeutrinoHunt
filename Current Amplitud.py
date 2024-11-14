#!/usr/bin/env python
# coding: utf-8

# In[2]:


import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

#savgol params
window_size = 101  
poly_order = 3  


# In[5]:


# read data
with h5py.File('MJD_NPML_2.hdf5', 'r') as f:
    waveform = np.array(f['raw_waveform'])


# In[22]:


def single_waveform(i):

    # Interpolate
    x = np.arange(len(waveform[i]))
    interp_func = interp1d(x, waveform[i], kind='cubic')  
    x_interp = np.linspace(0, len(waveform[i]) - 1, len(waveform[i]) * 10)  
    y_interp = interp_func(x_interp)
    
    
    y_interp = y_interp / np.max(np.abs(y_interp))
    
    # Savitzky-Golay filter 
    derivative = savgol_filter(y_interp, window_size, poly_order, deriv=1)
    
    # Find the maximum current amplitude
    max_amplitude = np.max(np.abs(derivative))
    
    return max_amplitude



# In[23]:


def generate_all_waveform():
    
    current_amplitudes = []



    for i in range(len(waveform)):
        x = np.arange(len(waveform[i]))
        interp_func = interp1d(x, waveform[i], kind='cubic')  
        x_interp = np.linspace(0, len(waveform[i]) - 1, len(waveform[i]) * 10)  
        y_interp = interp_func(x_interp)
     
        y_interp = y_interp / np.max(np.abs(y_interp))
        derivative = savgol_filter(y_interp, window_size, poly_order, deriv=1)
    
        max_amplitude = np.max(np.abs(derivative))
        current_amplitudes.append(max_amplitude)
        print('proceeding waveform',i)
        return current_amplitudes



# In[27]:





# In[ ]:




