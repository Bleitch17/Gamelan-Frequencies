import numpy as np
import numpy.typing as npt


def compute_threshold_float64(signal: npt.NDArray[np.float64], delta=1.01) -> float:
    """
    Compute a threshold for a given signal that captures the majority of activity in that signal.
    This function works by choosing a threshold at half the maximum point of the signal, and repeatedly lowering
    the threshold until the sum of all data points above the threshold changes by less than a specified delta parameter.

    Parameters
    ----------
    signal:
        The input signal to compute a threshold over. Values should be in the range [0.0, 1.0].
    
    delta:
        The change in sum at which to stop lowering the threshold.

    Returns
    -------
    out:
        A threshold (floating point number).
    """
    if np.min(signal) < 0.0:
        raise ValueError("Minimum value in the input signal is below 0. The input signal must have values in the range [0.0, 1.0].")

    maximum: float = np.max(signal)

    if maximum > 1.0:
        raise ValueError("Maximum value in the input signal is above 1. The input signal must have values in the range [0.0, 1.0].")

    current_threshold: float = maximum / 2.0
    current_score: float = np.sum(signal[signal > current_threshold])

    next_threshold: float = current_threshold / 2.0
    next_score: float = np.sum(signal[signal > next_threshold])

    while next_score / current_score > delta:
        current_threshold = next_threshold
        current_score = next_score

        next_threshold: float = current_threshold / 2.0
        next_score: float = np.sum(signal[signal > next_threshold])
    
    return next_threshold
