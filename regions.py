import numpy as np
import numpy.typing as npt


def get_regions_above_threshold(signal: npt.NDArray[np.float64], t: float) -> tuple[list[int], list[int]]:
    """
    Find regions of the input signal where the data is above a specified threshold.

    Parameters
    ----------
    signal:
        The input signal.

    t:
        The threshold.
    
    Returns
    -------
    regions:
        A tuple of the form (starts, ends) where starts is a list of start indexes (inclusive) for the regions above the threshold and
        ends is a list of end indexes (inclusive) for regions above the threshold.
    """

    mask: npt.NDArray[np.bool] = (signal > t).astype(np.int32)

    if not np.any(mask):
        return [], []

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

    return starts, ends
