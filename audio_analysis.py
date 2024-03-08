import opensmile
import librosa
import json


class AudioAnalysis:
    """
    A class for conducting audio analysis, including loading audio data, segmentations,
    spectrograms, extracting features using OpenSMILE, and visualizing data.

    Parameters
    ----------
    default_figsize : tuple of int, optional
        Default figure size for plots, by default (10, 4).
    sample_rate : int, optional
        Sample rate for audio loading, by default None which lets librosa decide.
    opensmile_feature_set : opensmile.FeatureSet, optional
        The feature set configuration for OpenSMILE, by default opensmile.FeatureSet.eGeMAPSv02.
    opensmile_feature_level : opensmile.FeatureLevel, optional
        The feature level for feature extraction, by default opensmile.FeatureLevel.LowLevelDescriptors.

    Attributes
    ----------
    audio_file_path : str or None
        Path to the audio file.
    segment_info_path : str or None
        Path to the segmentation information file.
    mapping_file_path : str or None
        Path to the mapping file (placeholder for future use).
    spectrogram_path : str or None
        Path to the spectrogram data file.
    audio_data : np.ndarray or None
        Loaded audio data.
    segment_data : dict or None
        Loaded segmentation data.
    spectrogram_data : np.ndarray or None
        Loaded spectrogram data.
    feature_data : pd.DataFrame or None
        Extracted feature data.
    default_figsize : tuple
        Default figure size for all plots.
    sample_rate : int or None
        Sample rate for audio loading.
    opensmile_feature_set : opensmile.FeatureSet
        OpenSMILE feature set used for feature extraction.
    opensmile_feature_level : opensmile.FeatureLevel
        OpenSMILE feature level for feature extraction.
    """

    def __init__(
        self,
        default_figsize=(10, 4),
        sample_rate=None,
        opensmile_feature_set=opensmile.FeatureSet.eGeMAPSv02,
        opensmile_feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    ):
        # Initialize attributes with default values or provided parameters
        self.audio_file_path = None
        self.segment_info_path = None
        self.mapping_file_path = None  # Placeholder for future use
        self.spectrogram_path = None
        self.audio_data = None
        self.segment_data = None
        self.spectrogram_data = None
        self.feature_data = None  # To store the extracted features
        self.default_figsize = default_figsize
        self.sample_rate = sample_rate
        self.opensmile_feature_set = opensmile_feature_set
        self.opensmile_feature_level = opensmile_feature_level

    def _load_data(self, file_path, data_type):
        """
        Loads data from a file path and handles errors uniformly.

        Parameters:
        - file_path: The path to the data file.
        - data_type: The type of data being loaded (e.g., 'audio', 'segmentation', 'spectrogram').

        Returns:
        - The loaded data, or None if loading failed.
        """
        try:
            if data_type == "audio":
                data, sr = librosa.load(file_path, sr=self.sample_rate)
                self.sampling_rate = sr
                return data
            elif data_type == "segmentation":
                with open(file_path, "r") as f:
                    return json.load(f)
            elif data_type == "spectrogram":
                return np.load(file_path)
        except Exception as e:
            self._log_error(f"Failed to load {data_type} data: {e}")
            return None

    def extract_features(self):
        """
        Extracts audio features from the loaded audio file using the OpenSMILE tool.

        This method requires that the audio file has been loaded beforehand. The features
        are extracted based on the specified OpenSMILE feature set and feature level set
        during object initialization. Extracted features are stored within the object for
        further analysis.

        Raises
        ------
        Exception
            If the feature extraction fails for any reason, an error message is logged, and
            the exception is raised.

        Notes
        -----
        - The method relies on the OpenSMILE configuration determined by `opensmile_feature_set`
          and `opensmile_feature_level` attributes.
        - Extracted features are stored in `self.feature_data`, a pandas DataFrame, which
          can be used for subsequent analysis or visualization.
        - A message indicating the shape of the extracted feature DataFrame is printed to
          standard output.
        """
        # We assume here that if audio data was loaded successfully, we have the path
        # if not self._ensure_audio_data_loaded():
        #     return

        # Initialize opensmile feature extractor with class attributes
        try:
            smile = opensmile.Smile(
                feature_set=self.opensmile_feature_set,
                feature_level=self.opensmile_feature_level,
            )

            self.feature_data = smile.process_file(self.audio_file_path)
            # print(f"Features extracted: {self.feature_data.shape}")
        except Exception as e:
            self._log_error(f"Failed to extract features: {e}")
