import numpy as np

# Apply Fast Fourier Transform, and normalize it
def normalized_fourier(waveform, tp0):
    peak = np.argmax(waveform)
    if tp0 >= peak:
        peak = min(len(waveform), tp0 + 1)
    frq = np.fft.fftfreq(len(waveform[tp0:peak]), d=1)
    magnitude = np.abs(np.fft.fft(waveform[tp0:peak]))
    normalized_fft = magnitude / np.max(magnitude)
    return normalized_fft, frq

# Low Frequency Power Ratio
def lfpr(frq_waveform,threshold=0.05):
    power_spectrum = np.abs(frq_waveform)**2
    low_frequency_power = np.sum(power_spectrum[frq_waveform < threshold])
    return low_frequency_power / np.sum(power_spectrum)
