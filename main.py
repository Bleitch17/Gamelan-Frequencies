
import math
import numpy as np
import numpy.typing as npt

from scipy.io import wavfile

import audio
import blob
import dsp
import plotting
import threshold


# A ratio at which to stop decreasing the threshold on an RMS signal.
# Passed in as the "delta" argument to threshold.compute_threshold_float64(...) for metallophone Gamelans.
METALLOPHONE_RMS_THRESHOLD_RATIO_LIMIT: float = 1.02

# The minimum length of a key strike for the metallophone Gamelan instruments, in seconds.
METALLOPHONE_MIN_KEY_STRIKE_LENGTH_SECONDS: float = 0.5


def read_metallophone_frequencies(wav_file_path: str, is_strike_selection_plot_enabled: bool = False) -> list[float]:
    """
    Extract the dominant frequencies of key strikes from a metallophone recording.
    If the recording contains repeated key strikes of the same note, the mean frequency for that note will be returned in the output list.

    Parameters
    ----------
    wav_file_path:
        The path to a .wav file containing the recording of the key strikes. The key strikes in the recording are assumed to be ordered
        from lowest note (frequency) to highest note (frequency). Each key strike is expected to last for more than half a second.
    
    is_strike_selection_plot_enabled:
        A boolean flag that controls whether to display a plot of the audio recording, with the key strikes selected from the recording.
        Defaults to False.

    Returns
    -------
    out:
        A list of frequencies. Each frequency in the list represents the dominant frequency detected in a single key strike. The frequencies
        follow the same ordering as the key strikes in the recording, so out[0] is the dominant frequency for the first key strike.
    """
    sample_rate_hz, data = wavfile.read(wav_file_path)

    # Convert audio signal to normalized float64 format (values are in [-1.0, 1.0)).
    # Extracting the first channel (assuming both channels are the same).
    normalized_audio_data: npt.NDArray[np.float64] = audio.normalize_pcm_to_float64(data)[:, 0]

    frame_length_samples: int = math.floor(0.01 * sample_rate_hz)
    hop_length_samples: int = math.floor(0.005 * sample_rate_hz)

    padded_normalized_audio_data: npt.NDArray[np.float64] = dsp.pad_to_frame_view_float64(normalized_audio_data, frame_length_samples, hop_length_samples, pad_value=0.0)

    audio_data_rms: npt.NDArray[np.float64] = dsp.rms_float64(padded_normalized_audio_data, frame_length_samples, hop_length_samples, mode="average")
    rms_threshold: float = threshold.compute_threshold_float64(audio_data_rms, delta=METALLOPHONE_RMS_THRESHOLD_RATIO_LIMIT)

    min_blob_size: int = int(METALLOPHONE_MIN_KEY_STRIKE_LENGTH_SECONDS * sample_rate_hz)

    blob_boundaries: list[tuple[int, int]] = blob.get_blob_boundaries_above_threshold(audio_data_rms, rms_threshold)
    blob_boundaries = blob.filter_blobs_by_size(blob_boundaries, min_blob_size)

    if is_strike_selection_plot_enabled:
        # Unselected data is represented by np.nan in this array.
        selected_key_strikes: npt.NDArray[np.float64] = audio.replace_outside_boundaries_float64(padded_normalized_audio_data, blob_boundaries, value=np.nan)

        plotting.plot_signals_float64([padded_normalized_audio_data, selected_key_strikes, audio_data_rms], [rms_threshold], sample_rate_hz)

    blobs: list[npt.NDArray[np.float64]] = blob.get_blobs(padded_normalized_audio_data, blob_boundaries)
    
    return list(map(lambda blob: float(dsp.dominant_freq_float64(blob, sample_rate_hz)), blobs))


if __name__ == "__main__":
    wav_file_path: str = "wav/A.wav"
    print(read_metallophone_frequencies(wav_file_path, True))
