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


def create_non_nan_region_mask(data_with_nan: npt.NDArray[np.float64], region_length_s: float, sample_rate_hz: float) -> npt.NDArray[np.bool]:
    """
    Create a boolean mask array to identify continuous regions of data that are not all nan within the input array.
    If mask[i] == True, then index i belongs to a non-nan region, even if data_with_nan[i] is nan.

    Return the boolean mask array.
    """
    
    non_nan_region_mask: npt.NDArray[np.bool] = np.zeros_like(data_with_nan).astype(np.bool)
    
    region_length_samples: int = math.floor(region_length_s * sample_rate_hz)

    for index in range(0, len(data_with_nan) - region_length_samples, region_length_samples):
        if not is_region_nan(data_with_nan, index, region_length_samples):
            non_nan_region_mask[index:index+region_length_samples] = True

    return non_nan_region_mask


def get_blob_boundaries(non_nan_region_mask: npt.NDArray[np.bool], min_blob_size_s: float, sample_rate_hz: float) -> list[tuple[int, int]]:
    """
    A blob is defined as a continuous set of samples that are assumed to contain useful data, as opposed to noise.
    
    This function accepts a boolean mask non_nan_region_mask where non_nan_region_mask[i] == True indicates that index i refers to a sample
    that falls in a "non-nan" region that may potentially contain useful data.

    This function returns pairs of inclusive blob boundary indices (start, end) such that the size of each blob is at least min_blob_size_s
    in terms of samples. 
    """
    
    min_blob_size_samples: int = math.floor(min_blob_size_s * sample_rate_hz)

    # From the numpy docs: https://numpy.org/doc/stable/reference/generated/numpy.diff.html
    # The first difference is given by out[i] = a[i + 1] - a[i].
    # So in this case, it is the later index minus the earlier index.
    # Therefore:
    #   diff[i] == 1 -> mask[i + 1] - mask[i] == 1 -> mask[i + 1] == 1, and mask[i] == 0. Since mask[i + 1] == 1, i + 1 is the first index of the region.
    #   diff[i] == -1 -> mask[i + 1] - mask[i] == -1 -> mask[i + 1] == 0, and mask[i] == 1. Since mask[i] == 1, i is the last index of the region. 
    # Important: need to cast non_nan_region_mask to int8 before performing the subtraction.
    non_nan_region_mask_diff: npt.NDArray[np.int8] = np.diff(non_nan_region_mask.astype(np.int8))

    # Need to add 1 to get the proper index based on above.
    blob_starts: npt.NDArray[np.intp] = np.where(non_nan_region_mask_diff == 1)[0] + 1

    # If the non_nan_region_mask starts with a 1, then treat index 0 as the start of a potential blob. 
    if non_nan_region_mask[0] == 1:
        blob_starts = np.insert(blob_starts, 0, 0)

    blob_ends: npt.NDArray[np.intp] = np.where(non_nan_region_mask_diff == -1)[0]

    if non_nan_region_mask[-1] == 1:
        blob_ends = np.append(blob_ends, len(non_nan_region_mask) - 1)
    
    blob_boundaries: list[tuple[int, int]] = list(zip(blob_starts, blob_ends))

    # Filter for blobs larger than the provided min_blob_size.
    return [(int(start), int(end)) for start, end in filter(lambda indices: indices[-1] - indices[0] > min_blob_size_samples, blob_boundaries)]


def create_selected_audio_data_array(audio_data: npt.NDArray[np.float64], blob_boundaries: list[tuple[int, int]]) -> npt.NDArray[np.float64]:
    selected_mask: npt.NDArray[np.bool] = np.zeros_like(audio_data).astype(np.bool)

    for start_index, end_index in blob_boundaries:
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

    non_nan_region_mask: npt.NDArray[np.bool] = create_non_nan_region_mask(data_with_nan=above_threshold_audio_data, region_length_s=0.1, sample_rate_hz=sr)

    blob_boundaries: list[tuple[int, int]] = get_blob_boundaries(non_nan_region_mask=non_nan_region_mask, min_blob_size_s=0.5, sample_rate_hz=sr)

    # Uncomment the line below to debug selection using the runs.
    # plot_selected_data(audio_data, create_selected_audio_data_array(audio_data, blob_boundaries), sr)

