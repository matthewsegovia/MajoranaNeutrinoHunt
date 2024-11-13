import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.stats import linregress
import argparse

def extract_peak_and_tail_slope(waveform):
    """
    Extract the peak index, peak value, and the tail slope of the waveform.
    Tail slope is computed from the last 500 samples of the waveform.
    """
    peak_index = np.argmax(waveform)  # Find the peak index
    peak_value = waveform[peak_index]  # Find the peak value

    # Calculate the tail slope using the last 500 samples
    tail_segment = waveform[-500:]  
    time = np.arange(len(tail_segment))  # Time indices for the last 500 samples
    slope, intercept, _, _, _ = linregress(time, tail_segment)

    return peak_index, peak_value, slope

def main(args):
    """
    Main function to load data from the provided HDF5 file and visualize the waveform.
    """
    # Load data from the specified HDF5 file
    with h5py.File(args.file_path, 'r') as file:
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

        # Get the random waveform and corresponding labels
        random_waveform = raw_waveform[random_index]
        energy_value = energy_label[random_index]
        psd_low_avse_value = psd_label_low_avse[random_index]
        psd_high_avse_value = psd_label_high_avse[random_index]
        psd_dcr_value = psd_label_dcr[random_index]
        psd_lq_value = psd_label_lq[random_index]
        tp0_value = tp0[random_index]
        detector_value = detector[random_index]
        run_number_value = run_number[random_index]
        id_value = id[random_index]

    # Extract peak index, peak value, and tail slope
    peak_index, peak_value, tail_slope = extract_peak_and_tail_slope(random_waveform)

    # Plot the waveform and relevant information
    plt.figure(figsize=(10, 5))
    plt.plot(random_waveform, label='Waveform')
    plt.title(f'Random Raw Waveform (Index: {random_index})')
    plt.xlabel('Time Index (μs)')
    plt.ylabel('ADC Counts')

    # Create a text box to display additional information
    textstr = (
        f"Peak Index: {peak_index}\n"
        f"Peak Value: {peak_value}\n"
        f"Tail Slope: {tail_slope}\n"
        f"Energy Label: {energy_value}\n"
        f"PSD Label Low Avse: {psd_low_avse_value}\n"
        f"PSD Label High Avse: {psd_high_avse_value}\n"
        f"PSD Label DCR: {psd_dcr_value}\n"
        f"PSD Label LQ: {psd_lq_value}\n"
        f"Start of Rising Edge: {tp0_value}\n"
        f"Detector: {detector_value}\n"
        f"Run Number: {run_number_value}\n"
        f"ID: {id_value}"
    )
    
    # Display text on the plot
    plt.gcf().text(0.45, 0.2, textstr, fontsize=10)

    # Highlight the start of the rising edge (tp0) on the plot
    plt.axvline(x=tp0_value, color='orange', linestyle='--', label='Rising Edge (tp0)')

    # Show the legend and the plot
    plt.legend()
    plt.show()

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Analyze waveform data from a single HDF5 file.")
    parser.add_argument('file_path', type=str, help="Path to the HDF5 file.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()  # Parse the command-line arguments
    main(args)  # Run the main function
