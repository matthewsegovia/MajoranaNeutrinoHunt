import numpy as np
import matplotlib.pyplot as plt
import h5py

# Apply Fast Fourier Transform, and normalize it
def normalized_fourier(waveform, tp0):
    peak = np.argmax(waveform)
    frq = np.fft.fftfreq(len(waveform[tp0:peak]), d=1)
    magnitude = np.abs(np.fft.fft(waveform[tp0:peak]))
    normalized_fft = magnitude / np.max(magnitude)
    return normalized_fft, frq

# Low Frequency Power Ratio
def lfpr(frq_waveform,threshold=0.05):
    power_spectrum = np.abs(frq_waveform)**2
    low_frequency_power = np.sum(power_spectrum[frq_tdrift < threshold])
    return low_frequency_power / np.sum(power_spectrum)

#Visualization
files = [
    "MJD_NPML_1.hdf5",
    "MJD_NPML_2.hdf5",
    "MJD_Test_0.hdf5",
    "MJD_Test_1.hdf5",
    "MJD_Test_2.hdf5",
    "MJD_Test_3.hdf5",
    "MJD_Test_4.hdf5",
    "MJD_Test_5.hdf5",
    "MJD_Train_0.hdf5",
    "MJD_Train_1.hdf5",
    "MJD_Train_2.hdf5",
    "MJD_Train_3.hdf5",
    "MJD_Train_4.hdf5",
    "MJD_Train_5.hdf5",
    "MJD_Train_6.hdf5",
    "MJD_Train_7.hdf5",
    "MJD_Train_8.hdf5",
    "MJD_Train_9.hdf5",
    "MJD_Train_10.hdf5",
    "MJD_Train_11.hdf5",
    "MJD_Train_12.hdf5",
    "MJD_Train_13.hdf5",
    "MJD_Train_14.hdf5",
    "MJD_Train_15.hdf5"
]
file_path = 'data/'  + files[2]

with h5py.File(file_path, 'r') as file:
    # Load the raw waveforms
    raw_waveform = np.array(file["raw_waveform"])

    # Load other labels
    energy_label = np.array(file["energy_label"])
    psd_label_low_avse = np.array(file["psd_label_low_avse"])
    psd_label_high_avse = np.array(file["psd_label_high_avse"])
    psd_label_dcr = np.array(file["psd_label_dcr"])
    psd_label_lq = np.array(file["psd_label_lq"])
    tp0 = np.array(file["tp0"])
    detector = np.array(file["detector"])
    run_number = np.array(file["run_number"])
    id = np.array(file["id"])

    # Select a random index
    random_index = np.random.choice(raw_waveform.shape[0])

    # Get the random waveform
    random_waveform = raw_waveform[random_index]

    # Access the labels for the selected index
    energy_value = energy_label[random_index]
    psd_low_avse_value = psd_label_low_avse[random_index]
    psd_high_avse_value = psd_label_high_avse[random_index]
    psd_dcr_value = psd_label_dcr[random_index]
    psd_lq_value = psd_label_lq[random_index]
    tp0_value = tp0[random_index]
    detector_value = detector[random_index]
    run_number_value = run_number[random_index]
    id_value = id[random_index]

fft_tdrift, frq_tdrift = normalized_fourier(random_waveform, tp0_value)
peak = np.argmax(random_waveform)
time_tdrift = np.arange(peak-tp0_value)

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time_tdrift, random_waveform[tp0_value:peak])
plt.title("Random Waveform in Time Domain")
plt.xlabel("Time (µs)")
plt.ylabel("Amplitude")

# Plot the magnitude of the waveform in the frequency domain
plt.subplot(2, 1, 2)
plt.plot(frq_tdrift[:len(frq_tdrift)//4], np.abs(fft_tdrift)[:len(frq_tdrift)//4])
# Power Spectrum of transformed waveform
plt.plot(np.abs(frq_tdrift[:len(frq_tdrift)//4])**2, np.abs(fft_tdrift)[:len(frq_tdrift)//4], linestyle='--', color='red')
plt.title("Waveform in Frequency Domain")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Magnitude")

# Show the plots
plt.tight_layout()
plt.show()

# Low Frequency Power Ratio
print(f"Low Freqency Power Ratio of the graph: {lfpr(fft_tdrift)}")