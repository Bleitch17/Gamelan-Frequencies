import numpy as np
import numpy.typing as npt


PCM_8_BIT_MAX: int = 128


def replace_outside_boundaries_float64(audio_signal: npt.NDArray[np.float64], boundaries: list[tuple[int, int]], value: np.float64 = 0.0) -> npt.NDArray[np.float64]:
    """
    Create a new signal where all samples in the input signal outside the given boundaries are replaced with the given value.

    Parameters
    ----------
    audio_signal:
        The input audio signal.

    boundaries:
        A list of (start, end) tuples of boundary indexes (inclusive).

    value:
        The value to replace samples outside the boundaries with.
        Defaults to 0.

    Returns
    -------
    out:
        An output signal where out[i] = audio_signal[i] if i is within any boundary range, otherwise out[i] = value.
    """
    replace_mask: npt.NDArray[np.bool] = np.zeros_like(audio_signal).astype(np.bool)

    for start_index, end_index in boundaries:
        replace_mask[start_index:end_index+1] = True

    output_signal: npt.NDArray[np.float64] = audio_signal.copy()
    output_signal[~replace_mask] = value

    return output_signal


def normalize_pcm_to_float64(audio_signal: npt.NDArray[np.integer]) -> npt.NDArray[np.float64]:
    """
    Create an audio signal in normalized floating point format [-1.0, 1.0) from an input audio signal
    in PCM format.

    Parameters
    ----------
    audio_signal:
        The input audio signal in PCM format. May be either 8-bit, 16-bit, or 32-bit PCM.
        Note that 24-bit PCM will be incorrectly scaled.

    Returns:
    --------
    out:
        An output audio signal in normalized float64 format with values in the range [1.0, 1.0).
    """
    if audio_signal.dtype == np.int16:
        return audio_signal.astype(np.float64) / np.iinfo(np.int16).max
    
    elif audio_signal.dtype == np.int32:
        return audio_signal.astype(np.float64) / np.iinfo(np.int32).max
    
    elif audio_signal.dtype == np.uint8:
        # 8-bit PCM is stored as an unsigned integer type, so need to shift to the middle of [0, 255]
        # before dividing to get [-1.0, 1.0).
        return (audio_signal.astype(np.float64) - PCM_8_BIT_MAX) / PCM_8_BIT_MAX
    
    else:
        raise ValueError(f"Unsupported dtype: {audio_signal.dtype}")
