from pathlib import Path
from pydub import AudioSegment

import pandas as pd
import matplotlib.pyplot as plt
import csv
from statistics import mean
import seaborn as sns

import textgrids

import os
import re


import numpy as np
from scipy.io import wavfile
from scipy.signal import periodogram


# Provided py file from Jens
from audio_analysis import AudioAnalysis


# File paths
DATASET_FILE_PATH = Path("./test_dataset.csv")
VOWEL_AUDIO_FILES_DIR = Path("./audio_files/vowels")
FRIC_AUDIO_FILES_DIR = Path("./audio_files/frics")
ORIGINAL_AUDIO_DIR = Path("./original_audio")
PLOT_FILES_DIR = Path("./plots")

# Ensure newdirectories exists
VOWEL_AUDIO_FILES_DIR.mkdir(exist_ok=True)
FRIC_AUDIO_FILES_DIR.mkdir(exist_ok=True)

# Ensure audio_files directory exists
PLOT_FILES_DIR.mkdir(exist_ok=True)

# Hardcoded convenience
SPEAKER_NAMES = {"KS": "Kayla", "SD": "Samuele", "SL": "Stella"}


# Find original audio folder
def get_audio_path(speaker, condition):
    return ORIGINAL_AUDIO_DIR / f"{speaker}_{condition}.wav"


# Takes in file path to audio and trims it according to the dataset
def split_audio(path_to_audio, clip_data, type):
    full_audio = AudioSegment.from_wav(path_to_audio)
    audio_file_segment = full_audio[
        float(clip_data["start"]) * 1000 : float(clip_data["end"]) * 1000
    ]

    if type == "vowel":
        new_audio_path = (
            VOWEL_AUDIO_FILES_DIR
            / f"{clip_data['word']}_{clip_data['speaker']}_{clip_data['hotneutral']}.wav"
        )
    elif type == "fric":
        new_audio_path = (
            FRIC_AUDIO_FILES_DIR
            / f"{clip_data['word']}_{clip_data['speaker']}_{clip_data['hotneutral']}.wav"
        )
    # Uncomment the next line to export segments
    audio_file_segment.export(new_audio_path, format="WAV")
    return new_audio_path


# Creates the AudioAnalysis object for an audio clip
def create_audio_data_object(path_to_audio):
    # Create an instance of the AudioAnalysis object
    audio_analysis = AudioAnalysis()
    # Load audio file path
    audio_analysis.audio_file_path = path_to_audio
    # Load the audio data
    audio_analysis._load_data(path_to_audio, "audio")
    # Get the data
    audio_analysis.extract_features()

    return audio_analysis


"""
Due to changing the method of splitting audio in Praat, there are 2 versions of
getting the data. It's a bit messy because of time constraints and we just needed to parse 
the data ASAP.
"""


def process_vowel_data_item(item, clip_type="vowel"):
    audio_path = get_audio_path(item["speaker"], item["hotneutral"])
    new_audio_path = split_audio(audio_path, item, clip_type)
    item["audiofilepath"] = new_audio_path
    item["analysisobj"] = create_audio_data_object(new_audio_path)
    return item


def read_and_process_vowel_data(csv_file_path, clip_type="vowel"):
    full_dataset = []
    with open(csv_file_path, mode="r", newline="", encoding="utf-8") as csvfile:
        csv_vowel_file = csv.DictReader(csvfile)
        for item in csv_vowel_file:
            full_dataset.append(process_vowel_data_item(item, clip_type))

    return full_dataset


def get_all_vowel_data():
    full_dataset = read_and_process_vowel_data(DATASET_FILE_PATH)
    extra_data = get_extra_vowel_file()
    for item in extra_data:
        full_dataset.append(process_vowel_data_item(item))

    return full_dataset


def get_extra_vowel_file():
    # read in TextGrid
    praat_audio_textgrid = textgrids.TextGrid(
        "./original_audio/TextGrids/SL_hot_a_i_u.TextGrid"
    )

    vowel_list = []

    # iterate through first tier, which contains duration info
    for item in praat_audio_textgrid["Mary"]:
        # convert Praat to Unicode in the label
        vowel = item.text.transcode()

        if vowel != "":
            start = item.xmin
            end = item.xmax
            word_label = vowel

            vowel_list.append(
                {
                    "vowel": vowel,
                    "word": word_label,
                    "start": start,
                    "end": end,
                    "hotneutral": "hot",
                    "speaker": "SL",
                }
            )

    return vowel_list


def get_all_the_fric_data():
    full_dataset = []
    speaker_initials = SPEAKER_NAMES.keys()
    fricatives_data = []

    for temp in ["hot", "neutral"]:
        for speaker in speaker_initials:
            textgrid_path = f"./original_audio/TextGrids/{speaker}_{temp}.TextGrid"
            fric_list = get_fricative_start_stops(textgrid_path, temp, speaker)
            fricatives_data.extend(fric_list)

    for item in fricatives_data:
        audio_path = get_audio_path(item["speaker"], item["hotneutral"])
        new_audio_path = split_audio(audio_path, item, "fric")
        item["audiofilepath"] = new_audio_path
        item["analysisobj"] = create_audio_data_object(new_audio_path)

        full_dataset.append(item)

    return full_dataset


# Gets the start/stop times for each fricative from a TextGrid file
def get_fricative_start_stops(path_to_textgrid, hot_or_cold, speaker):
    # read in TextGrid
    praat_audio_textgrid = textgrids.TextGrid(path_to_textgrid)

    fric_list = []

    # iterate through first tier, which contains duration info
    for item in praat_audio_textgrid["fricatives"]:
        # convert Praat to Unicode in the label
        fric = item.text.transcode()

        if fric != "":
            start = item.xmin
            end = item.xmax
            word_label = ""

            for word in praat_audio_textgrid["ORT-MAU"]:
                if start > word.xmin and end < word.xmax:
                    word_label = word.text.transcode()

            fric_list.append(
                {
                    "fricative": fric,
                    "word": word_label,
                    "start": start,
                    "end": end,
                    "hotneutral": hot_or_cold,
                    "speaker": speaker,
                }
            )

    return fric_list
