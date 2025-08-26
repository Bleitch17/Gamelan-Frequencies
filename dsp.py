import math

import numpy as np
import numpy.typing as npt

from typing import Literal


def fft_float64(signal: npt.NDArray[np.float64], sample_frequency_hz: float) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128]]:
    """
    Perform the Fast Fourier Transform (FFT) on an input signal.

    Returns a tuple (frequencies, spectrum).
    """
    spectrum = np.fft.rfft(signal, norm="forward")

    # Note - Type hints for rfftfreq show np.floating, but assuming this defaults to np.float64?
    freqs = np.fft.rfftfreq(len(signal), 1 / sample_frequency_hz)

    return freqs, spectrum


def ifft_complex128(spectrum: npt.NDArray[np.complex128]) -> npt.NDArray[np.float64]:
    """
    Perform the Inverse FFT on an input spectrum.
    The input spectrum is expected to have been produced by the dsp.fft function.

    Returns the output of the iFFT operation.
    """
    
    return np.fft.irfft(spectrum, norm="forward")


def _is_signal_padded_float64(signal: npt.NDArray[np.float64], frame_length_samples: int, hop_length_samples: int) -> bool:
    """
    Check whether the input signal is padded such that it can be cleanly viewed as overlapping frames spaced every hop_length_samples samples.

    :param signal: The input signal to check.
    :param frame_length_samples: The length of each frame in the input signal, in samples.
    :param hop_length_samples: The distance between the start of each frame, in samples.
    For example, if the hop length is 100 samples, and the first frame starts at signal[0], the next frame will start at signal[100].

    :return: True if the input signal is properly padded and False otherwise.
    """
    if len(signal) < frame_length_samples:
        return False
    
    elif len(signal) == frame_length_samples:
        return True
    
    else:
        return (len(signal) - frame_length_samples) % hop_length_samples == 0


def pad_frames_float64(signal: npt.NDArray[np.float64], frame_length_samples: int, hop_length_samples: int, pad_value: np.float64) -> npt.NDArray[np.float64]:
    """
    Pad the last frame of the input signal so the output signal may be cleanly viewed as overlapping frames spaced every hop_length_samples samples.

    :param signal: The input signal to pad.
    :param frame_length_samples: The length of each frame in the input signal, in samples.
    :param hop_length_samples: The distance between the start of each frame, in samples.
    For example, if the hop length is 100 samples, and the first frame starts at signal[0], the next frame will start at signal[100].
    :param pad_value: The value with which to pad the input signal.

    :return: The padded output signal.
    """
    # TODO - Validate these inputs? E.g.: empty signal, 0 frame length, 0 hop length, hop length no larger than frame length, etc.

    signal_length_samples: int = len(signal)
    pad_length_samples: int = 0

    # If the signal is shorter than the given frame length, pad the signal so there is at least one frame in the output.
    if signal_length_samples < frame_length_samples:
        pad_length_samples = frame_length_samples - signal_length_samples
    
    # If The signal length is longer than a single frame, and the hop length evenly divides the signal length, then
    # the amount needed to fill up the last frame is the difference between the frame length and the hop length, since the last
    # frame already has hop_length_samples inside it.
    elif signal_length_samples % hop_length_samples == 0:
        pad_length_samples = frame_length_samples - hop_length_samples
    
    # If the signal length is longer than a single frame, and the hop length does not evenly divide the signal length, then
    # the amount needed to fill up the last frame is the difference between the frame length and the amount of samples already in
    # the last frame. Since the frames start every hop_length_samples, the amount of samples in the last frame is given by
    # (signal_length_samples % hop_length_samples).
    else:
        pad_length_samples = frame_length_samples - (signal_length_samples % hop_length_samples)
    
    return np.pad(signal, (0, pad_length_samples), constant_values=pad_value)


def _rms_frames_to_rms_signal(rms_frames: npt.NDArray[np.float64], frame_length_samples: int, hop_length_samples: int) -> npt.NDArray[np.float64]:
    """
    Produce a "smeared" RMS output signal given an RMS frames signal.

    :param rms_frames: The RMS frames signal. rms_frames[i] is the rms for frame i.
    :param frame_length_samples: The number of samples used to compute the RMS for each frame.
    :param hop_length_samples: The spacing of frames from the original signal the RMS was computed from.
    For example, if the hop length is 100 samples, and the first frame starts at signal[0], the next frame will start at signal[100].
    
    :return: The smeared RMS signal, in the same shape as the original signal the RMS was computed from.
    """
    sparse_rms_unpadded: npt.NDArray[np.float64] = np.zeros(len(rms_frames) * hop_length_samples)
    sparse_rms_unpadded[::hop_length_samples] = rms_frames
    
    sparse_rms = np.pad(sparse_rms_unpadded, (0, frame_length_samples - hop_length_samples), constant_values=0)

    sparse_counts_unpadded: npt.NDArray[np.float64] = np.zeros(len(rms_frames) * hop_length_samples)
    sparse_counts_unpadded[::hop_length_samples] = np.ones_like(rms_frames)

    sparse_counts = np.pad(sparse_counts_unpadded, (0, frame_length_samples - hop_length_samples), constant_values=0)

    convolved_rms: npt.NDArray[np.float64] = np.convolve(sparse_rms, np.ones(frame_length_samples), mode="full")[:-frame_length_samples+1]
    convolved_counts: npt.NDArray[np.float64] = np.convolve(sparse_counts, np.ones(frame_length_samples), mode="full")[:-frame_length_samples+1]

    return convolved_rms / convolved_counts


def rms_float64(signal: npt.NDArray[np.float64], frame_length_samples: int, hop_length_samples: int, mode: Literal["average", "last"] = "average") -> npt.NDArray[np.float64]:
    """
    Computes the RMS energy from an input signal.

    :param signal: The input signal to compute the RMS over. This input signal should first be padded using dsp.pad_frames_float64.
    If the input signal is not padded properly, it will be padded with 0 values accordingly.
    :param frame_length_samples: The length of each frame of the input signal, in samples, over which to compute the RMS over.
    :param hop_length_samples: The distance between the starting point of each frame, in samples.
    For example, if the hop length is 100 samples, and the first frame starts at signal[0], the next frame will start at signal[100].
    :param mode: Controls how the RMS is represented in the output signal.
    - "average": Each sample in the output signal is assigned the average (mean) RMS value of all frames that contain it.
    - "last": Each sample in the output signal is assigned the RMS value of the last, i.e.: frame with the largest starting index, that contains it.
    
    :return: The RMS signal, in the same shape as the (padded) input.
    """
    if mode not in {"average", "last"}:
        raise ValueError(f"Invalid mode: {mode}")

    padded_signal: npt.NDArray[np.float64] = signal

    if not _is_signal_padded_float64(signal, frame_length_samples, hop_length_samples):
        padded_signal = pad_frames_float64(signal, frame_length_samples, hop_length_samples, 0.0)

    padded_signal_windows: npt.NDArray[np.float64] = np.lib.stride_tricks.sliding_window_view(padded_signal, window_shape=frame_length_samples, writeable=False)
    padded_signal_frames: npt.NDArray[np.float64] = padded_signal_windows[::hop_length_samples]

    padded_signal_frames_rms: npt.NDArray[np.float64] = np.sqrt(np.mean(padded_signal_frames**2, axis=1))

    if mode == "average":
        return _rms_frames_to_rms_signal(padded_signal_frames_rms, frame_length_samples, hop_length_samples)
    
    else:
        return np.pad(np.repeat(padded_signal_frames_rms, hop_length_samples), (0, frame_length_samples - hop_length_samples), constant_values=padded_signal_frames_rms[-1])
