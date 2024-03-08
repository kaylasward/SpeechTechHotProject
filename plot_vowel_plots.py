import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from statistics import mean
import os


PLOT_FILES_DIR = Path("./plots")
PLOT_FILES_DIR.mkdir(exist_ok=True)


# Plot the data related to the vowels such as jimmer and shimmer
def plot_vowel_data(measure_type, dataset):
    df = pd.DataFrame(dataset)

    # Replace the short speaker identifiers with the full ones
    df["speaker"] = df["speaker"].replace(
        {"KS": "KS (ENG)", "SD": "SD (IT)", "SL": "SL (SWE)"}
    )

    # Sort the data for the x axis
    df["word_vowel"] = df["word"] + "__" + df["vowel"]
    df.sort_values(by=["vowel", "word"], inplace=True)
    df["word_vowel"] = pd.Categorical(
        df["word_vowel"],
        categories=df["word_vowel"].unique(),
        ordered=True,
    )

    plt.figure(figsize=(12, 8))
    hotneutral_palette = {"hot": "#D7191C", "neutral": "#0072B2"}
    markers = {"KS (ENG)": "o", "SD (IT)": "s", "SL (SWE)": "^"}

    sns.scatterplot(
        data=df,
        x="word_vowel",
        y=measure_type,
        hue="hotneutral",
        style="speaker",
        markers=markers,
        palette=hotneutral_palette,
        s=100,
    )

    plt.title(
        f"Mean {measure_type.capitalize()} for Each Speaker in a Hot vs Neutral Temperature Room"
    )
    plt.ylabel(f"Mean {measure_type.capitalize()}")
    plt.xlabel("")
    plt.xticks(rotation=60)

    handles, labels = plt.gca().get_legend_handles_labels()
    new_labels = [
        "Temperature",
        "hot",
        "neutral",
        "Speaker",
        "KS (ENG)",
        "SD (IT)",
        "SL (SWE)",
    ]
    new_handles = [handles[0]] + handles[1:3] + [handles[3]] + handles[4:]
    plt.legend(new_handles, new_labels, loc="upper right", ncol=1)

    plt.tight_layout()

    filename = f"{measure_type}.png"
    full_path = os.path.join(PLOT_FILES_DIR, filename)
    plt.savefig(full_path)


# Add the data values to the dataset for easy plotting
def process_dataset(dataset, feature_data):
    new_dataset = []
    for i, datapoint in enumerate(dataset):
        values = datapoint["analysisobj"].feature_data[feature_data].values
        average_val = mean(values)

        if feature_data == "shimmerLocaldB_sma3nz":
            datapoint["meanshimmer"] = average_val
        elif feature_data == "jitterLocal_sma3nz":
            datapoint["meanjitter"] = average_val

        dataset[i] = datapoint
        new_dataset.append(datapoint)
    return new_dataset


def plot_jitter(dataset):
    jitter_dataset = process_dataset(dataset, "jitterLocal_sma3nz")
    plot_vowel_data("meanjitter", jitter_dataset)


def plot_shimmer(dataset):
    shimmer_dataset = process_dataset(dataset, "shimmerLocaldB_sma3nz")
    plot_vowel_data("meanshimmer", shimmer_dataset)
