import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

#savgol params
window_size = 101
poly_order = 3

def max_amplitude(waveform):

    # Interpolate
    x = np.arange(len(waveform))
    interp_func = interp1d(x, waveform, kind='cubic')
    x_interp = np.linspace(0, len(waveform) - 1, len(waveform) * 10)
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
