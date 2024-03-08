from pathlib import Path

import audio_processing
import plot_spectraltilt
import plot_vowel_plots

# File paths
DATASET_FILE_PATH = Path("./test_dataset.csv")
ORIGINAL_AUDIO_DIR = Path("./original_audio")


# Hardcoded convenience
SPEAKER_NAMES = {"KS": "Kayla", "SD": "Samuele", "SL": "Stella"}


# Split up data by speaker for easier analysis
def split_dataset_by_speaker(full_dataset):
    datasets = {
        speaker: [item for item in full_dataset if item["speaker"] == speaker]
        for speaker in ["KS", "SD", "SL"]
    }
    return datasets["KS"], datasets["SD"], datasets["SL"]


full_vowel_dataset = audio_processing.get_all_vowel_data()
dataset_vowels_KS, dataset_vowels_SD, dataset_vowels_SL = split_dataset_by_speaker(
    full_vowel_dataset
)

full_fric_dataset = audio_processing.get_all_the_fric_data()
dataset_frics_KS, dataset_frics_SD, dataset_frics_SL = split_dataset_by_speaker(
    full_fric_dataset
)


plot_vowel_plots.plot_jitter(full_vowel_dataset)
plot_vowel_plots.plot_shimmer(full_vowel_dataset)

plot_spectraltilt.plot_spectraltilt()
