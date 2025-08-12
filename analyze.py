import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from scipy.io import wavfile

# Need this line on Linux so I can pip install matplotlib locally for Python versions different than the system version.
matplotlib.use("Qt5Agg")


# An activity threshold - continuous runs of samples that fall below this threshold are considered "noise".
THRESHOLD: float = 250.0


def is_region_nan(data: npt.NDArray[np.float64], region_start_index: int, region_length_samples: int) -> bool:
    """
    Check a region of data for continuous NAN values.
    The region checked is [region_start_index, region_start_index + region_length_samples).

    Return True if all data in the region is np.nan, and false otherwise.
    """
    return bool(np.all(np.isnan(data[region_start_index:region_start_index+region_length_samples])))


def create_not_nan_blob_mask(data_with_nan: npt.NDArray[np.float64], blob_length_s: float, sample_rate_hz: float) -> npt.NDArray[np.bool]:
    """
    Create a boolean mask array to identify continuous regions of data (aka "blobs") that are not all nan within the input array.
    If mask[i] == True, then data_with_nan[i] belongs to a blob, even if data_with_nan[i] is nan.

    Return the boolean mask array.
    """
    
    blob_mask: npt.NDArray[np.bool] = np.zeros_like(data_with_nan).astype(np.bool)
    
    blob_length_samples: int = math.floor(blob_length_s * sample_rate_hz)

    for index in range(0, len(data_with_nan) - blob_length_samples, blob_length_samples):
        if not is_region_nan(data_with_nan, index, blob_length_samples):
            blob_mask[index:index+blob_length_samples] = True

    return blob_mask


def get_region_boundaries(region_mask: npt.NDArray[np.bool], min_blob_size_s: float, sample_rate_hz: float) -> list[tuple[int, int]]:
    """
    Given a boolean mask where mask[i] == True indicates that i falls within a 
    """
    
    min_blob_size_samples: int = math.floor(min_blob_size_s * sample_rate_hz)
    
    # Compute the difference between adjacent elements
    diff = np.diff(region_mask)
    
    # Starts of runs: where diff == 1 (0->1 transition), add +1 to get start index
    run_starts = np.where(diff == 1)[0] + 1
    
    # Ends of runs: where diff == -1 (1->0 transition), index is end of run
    run_ends = np.where(diff == -1)[0]
    
    # Handle if the array starts with a run
    if region_mask[0] == 1:
        run_starts = np.insert(run_starts, 0, 0)
    # Handle if the array ends with a run
    if region_mask[-1] == 1:
        run_ends = np.append(run_ends, len(region_mask) - 1)
    
    # Return list of (start, end) pairs (inclusive)
    runs: list[tuple[int, int]] = list(zip(run_starts, run_ends))

    return [(int(start), int(end)) for start, end in filter(lambda indices: indices[-1] - indices[0] > min_run_size, runs)]


def create_selected_audio_data_array(audio_data: npt.NDArray[np.float64], run_indices: list[tuple[int, int]]) -> npt.NDArray[np.float64]:
    selected_mask: npt.NDArray[np.bool] = np.zeros_like(audio_data).astype(np.bool)

    for start_index, end_index in runs:
        selected_mask[start_index:end_index+1] = True

    selected_audio_data: npt.NDArray[np.float64] = audio_data.copy()
    selected_audio_data[~selected_mask] = np.nan

    return selected_audio_data


def plot_selected_data(audio_data: npt.NDArray[np.float64], selected_audio_data: npt.NDArray[np.float64], sample_rate_hz: float) -> None:
    t = np.linspace(0, audio_data.shape[0] / sample_rate_hz, audio_data.shape[0])

    plt.figure(figsize=(30, 10))
    plt.plot(t, audio_data)
    plt.plot(t, selected_audio_data, color='g')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Audio Waveform")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    sr, data = wavfile.read("wav/A.wav")
    print(sr, data.shape)

    audio_data: npt.NDArray[np.float64] = data[:, 0].astype(np.float64)
    print(audio_data.shape)
    
    above_threshold_audio_data: npt.NDArray = audio_data.copy()
    above_threshold_audio_data[audio_data < THRESHOLD] = np.nan

    # TODO - Pick up here, determine naming conventions, i.e.: region vs blob.

    region_mask: npt.NDArray[np.bool] = create_not_nan_blob_mask(data_with_nan=above_threshold_audio_data, blob_length_s=0.1, sample_rate_hz=sr)

    min_run_size_samples: int = math.floor(0.5 * sr)
    runs: list[tuple[int, int]] = get_region_boundaries(region_mask, min_run_size_samples)

    # Uncomment the line below to debug selection using the runs.
    plot_selected_data(audio_data, create_selected_audio_data_array(audio_data, runs), sr)
