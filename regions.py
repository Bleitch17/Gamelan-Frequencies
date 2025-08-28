import numpy as np
import numpy.typing as npt


def get_regions_above_threshold(signal: npt.NDArray[np.float64], t: float) -> list[tuple[int, int]]:
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
        A list of tuples of the form (start, end) where start is a start index (inclusive) for a region above the threshold and
        end is an end index (inclusive) for a region above the threshold. regions[i] is the start and end boundary for region i.
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


def filter_regions_by_size(regions: list[tuple[int, int]], size: int) -> list[tuple[int, int]]:
    """
    Filter a list of regions below a given size.

    Parameters
    ----------
    regions:
        A list of region endpoints. The endpoints are represented as tuples of the form (start, end) where start and end are inclusive indexes.
    
    size:
        The minimum size of the regions to return.
    
    Returns
    -------
    out:
        A filtered list of region endpoints where all the regions are at least as large as size.
    """
    return list(filter(lambda bounds: bounds[-1] - bounds[0] + 1 >= size, regions))
