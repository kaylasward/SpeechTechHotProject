from pathlib import Path
import os

from scipy.io import wavfile
from scipy.signal import periodogram
from scipy.stats import linregress
from scipy.interpolate import interp1d
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt


PLOT_FILES_DIR = Path("./plots")
PLOT_FILES_DIR.mkdir(exist_ok=True)


obstruents = {
    "s-just": {
        "KS (ENG) Hot": "./audio_files/frics/just_KS_hot.wav",
        "KS (ENG) Neutral": "./audio_files/frics/just_KS_neutral.wav",
        "SD (IT) Hot": "./audio_files/frics/just_SD_hot.wav",
        "SD (IT) Neutral": "./audio_files/frics/just_SD_neutral.wav",
        "SL (SWE) Hot": "./audio_files/frics/just_SL_hot.wav",
        "SL (SWE) Neutral": "./audio_files/frics/just_SL_neutral.wav",
    },
    "f-before": {
        "KS (ENG) Hot": "./audio_files/frics/before_KS_hot.wav",
        "KS (ENG) Neutral": "./audio_files/frics/before_KS_neutral.wav",
        "SD (IT) Hot": "./audio_files/frics/before_SD_hot.wav",
        "SD (IT) Neutral": "./audio_files/frics/before_SD_neutral.wav",
        "SL (SWE) Hot": "./audio_files/frics/before_SL_hot.wav",
        "SL (SWE) Neutral": "./audio_files/frics/before_SL_neutral.wav",
    },
    # "e-before": {
    #     "KS (ENG) Hot": "./audio_files/vowels/that_KS_hot.wav",
    #     "KS (ENG) Neutral": "./audio_files/vowels/that_KS_neutral.wav",
    #     "SD (IT) Hot": "./audio_files/vowels/that_SD_hot.wav",
    #     "SD (IT) Neutral": "./audio_files/vowels/that_SD_neutral.wav",
    #     "SL (SWE) Hot": "./audio_files/vowels/that_SL_hot.wav",
    #     "SL (SWE) Neutral": "./audio_files/vowels/that_SL_neutral.wav",
    # },
}


# Function to load a WAV file and select only one channel if it's stereo
def load_wav_channel(audio_path, channel=0):
    sample_rate, audio = wavfile.read(audio_path)
    if audio.ndim > 1:  
        audio = audio[:, channel]
    return sample_rate, audio.astype(np.float32)


# Calculate the spectral tilt
def calculate_spectral_tilt(frequencies, psd):
    # Remvoe 0 or neg values if they exist
    valid_indices = (frequencies > 0) & (psd > 0)
    
    # Convert freqs to log scale
    log_freq = np.log10(frequencies[valid_indices])
    # Convert PSD values to log scale
    log_psd = 10 * np.log10(psd[valid_indices])
    
    # Get slope and intercept
    slope, intercept, _, _, _ = linregress(log_freq, log_psd)
    return slope, intercept


# Sequence that calculates the spectral tilt and gets the necessary items for plotting
def do_plot_calcs(wav_path):
    # Fix audio for calculations
    sample_rate, audio = load_wav_channel(wav_path)
    
    # Set lower bound for human auditory perception
    min_freq = 20
    # Set upper bound from Nyquist frequency
    max_freq = sample_rate / 2
    
    # Calculate the psd using scipy
    # Also returns sample frequencies used for calculating
    frequencies, psd = periodogram(audio, fs=sample_rate)

    # Returns numbers spaced evenly on a log scale based on freq bounds
    freq_arr = np.logspace(np.log10(min_freq), np.log10(max_freq), num=1000)
    
    # Create interpolation function from scipy based on PSD freqs
    interp_function = interp1d(frequencies, psd, bounds_error=False, fill_value="extrapolate")
    # Map the interpolated PSD data to log points for the plot
    psd_interp = interp_function(freq_arr)
    
    # Slope = signal power across freqs
    slope, intercept = calculate_spectral_tilt(freq_arr, psd_interp)
    
    return freq_arr, intercept, slope


# Plot a single obstruent
def plot_obstruent(obstruent_data, obstruent_label):
    plt.figure(figsize=(14, 8))
    colors = ["#b22234", "#009246", "#FECC02"]
    styles = ["-", "--"]

    # Do some calculations to get spectral tilt
    for idx, (label, wav_path) in enumerate(obstruent_data.items()):
        freq_arr, intercept, slope = do_plot_calcs(wav_path)

        # Plot the regression line for spectral tilt
        plt.plot(
            freq_arr,
            intercept + slope * np.log10(freq_arr),
            label=f"{label} (Slope: {slope:.2f})",
            color=colors[idx // 2],
            linestyle=styles[idx % 2],
        )

    plt.title(f"Spectral Tilt Comparison of /{obstruent_label.split("-")[0]}/ in '{obstruent_label.split("-")[1]}' between a Hot and Neutral Temperature Room")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [dB/Hz]")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(loc='lower left', bbox_to_anchor=(0, 0))
    plt.grid(True)
    
    filename = f"spectraltiltcomp_{obstruent_label}.png"
    full_path = os.path.join(PLOT_FILES_DIR, filename)
    plt.savefig(full_path)
    
    
# Iterate over each obstruent and plot the results
def plot_spectraltilt():
    for obstruent_label, obstruent_data in obstruents.items():
        plot_obstruent(obstruent_data, obstruent_label)
