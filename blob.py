import numpy as np
import numpy.typing as npt


def get_blob_boundaries_above_threshold(signal: npt.NDArray[np.float64], t: float) -> list[tuple[int, int]]:
    """
    Find blobs of the input signal where the data is above a specified threshold.

    Parameters
    ----------
    signal:
        The input signal.

    t:
        The threshold.
    
    Returns
    -------
    blob_boundaries:
        A list of tuples of the form (start, end) where start is a start index (inclusive) for a region above the threshold and
        end is an end index (inclusive) for a blob above the threshold. blob_boundaries[i] is the start and end boundary for blob i.
    """

    mask: npt.NDArray[np.bool] = (signal > t).astype(np.int32)

    if not np.any(mask):
        return []

    # From the numpy docs:
    # np.diff calculates the n-th discrete difference along the given axis. The first difference is given by:
    # out[i] = a[i + 1] - a[i]
    #
    # mask should be an array of 0's and 1's. 
    # mask[i] is 0 when signal[i] <= t, and 1 when signal[i] > t.
    # 
    # Note that region start and end indexes are inclusive.
    # when mask goes from 0 -> 1, diff[i] = mask[i + 1] - mask[i] = 1 - 0 = 1. So the starting index of the region is i + 1.
    # when mask goes from 1 -> 0, diff[i] = mask[i + 1] - mask[i] = 0 - 1 = -1. So the ending index of the region is i.
    diff: npt.NDArray[np.int32] = np.diff(mask)

    starts: list[int] = list(np.where(diff == 1)[0] + 1)
    ends: list[int] = list(np.where(diff == -1)[0])

    # The first sample in the signal may be above the threshold.
    if mask[0]:
        starts = [0] + starts
    
    # The last sample in the signal may be above the threshold.
    if mask[-1]:
        ends = ends + [len(signal) - 1]

    return list(zip(starts, ends))


def filter_blobs_by_size(blob_boundaries: list[tuple[int, int]], size: int) -> list[tuple[int, int]]:
    """
    Filter a list of blobs below a given size.

    Parameters
    ----------
    blob_boundaries:
        A list of blob endpoints. The endpoints are represented as tuples of the form (start, end) where start and end are inclusive indexes.
    
    size:
        The minimum size of the blobs to return.
    
    Returns
    -------
    out:
        A filtered list of blob endpoints where all the blobs are at least as large as size.
    """
    return list(filter(lambda bounds: bounds[-1] - bounds[0] + 1 >= size, blob_boundaries))


def get_blobs(signal: npt.NDArray[np.float64], blob_boundaries: list[tuple[int, int]]) -> list[npt.NDArray[np.float64]]:
    """
    Produce a list of blobs given an input signal and a list of blob boundaries.

    Parameters
    ----------
    signal:
        The input signal to take blobs from.

    blob_boundaries:
        A list of blob endpoints. The endpoints are represented as tuples of the form (start, end) where start and end are inclusive indexes.
    
    Returns
    -------
    out:
        A list of blobs taken from the input signal.
    """
    return [signal[start:end+1] for start, end in blob_boundaries]
