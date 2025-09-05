import math

import numpy as np
import numpy.typing as npt

from typing import Literal


def fft_float64(signal: npt.NDArray[np.float64], sample_rate_hz: float) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128]]:
    """
    Perform the Fast Fourier Transform (FFT) on an input signal.
    Note that this is performed using the "forward" normalization mode from numpy.fft.

    Parameters
    ----------
    signal:
        The input signal to compute the FFT over.
    
    sample_rate_hz:
        The sampling frequency, in Hz, used to produce the signal.

    Returns
    -------
    out:
        A tuple of the form (frequency_buckets, spectrum).
    """
    spectrum = np.fft.rfft(signal, norm="forward")

    # Note - Type hints for rfftfreq show np.floating, but assuming this defaults to np.float64?
    freqs = np.fft.rfftfreq(len(signal), 1 / sample_rate_hz)

    return freqs, spectrum


def ifft_complex128(spectrum: npt.NDArray[np.complex128]) -> npt.NDArray[np.float64]:
    """
    Perform the Inverse Fast Fourier Transform (FFT) on an input spectrum.
    Note that this is performed using the "forward" normalization mode from numpy.fft.

    Parameters
    ----------
    spectrum:
        The input spectrum to compute the Inverse FFT over.
    
    Returns
    -------
    out:
        The output of the Inverse FFT operation. Corresponds to the original input signal passed into dsp.fft_float64.
    """
    
    return np.fft.irfft(spectrum, norm="forward")


def dominant_freq_float64(signal: npt.NDArray[np.float64], sample_rate_hz: float) -> float:
    """
    Estimate the dominant frequency of a real-valued signal by:
    1. Applying a Hann Window.
    2. Taking the magnitude of the FFT spectrum of the signal.
    3. Using quadratic (parabolic) interpolation around the peak bin.

    Parameters
    ----------
    signal:
        The input signal.
    
    sample_rate_hz:
        The sampling rate used to create the input signal, in Hz.

    Returns
    -------
    out:
        An estimate of the dominant frequency in the input signal.
    """
    signal_length_samples: int = len(signal)

    signal_with_window: npt.NDArray[np.float64] = signal * np.hanning(signal_length_samples)

    freqs, spectrum = fft_float64(signal_with_window, sample_rate_hz)
    magnitude: npt.NDArray[np.float64] = np.abs(spectrum)

    dominant_freq_index: int = np.argmax(magnitude)

    # Parabolic interpolation code from ChatGPT:
    if 1 <= dominant_freq_index <= len(magnitude) - 2:
        alpha: float = np.log(magnitude[dominant_freq_index - 1])
        beta: float = np.log(magnitude[dominant_freq_index])
        gamma: float = np.log(magnitude[dominant_freq_index + 1])

        p: float = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)

    else:
        p: float = 0.0
    
    return freqs[dominant_freq_index] + p * (freqs[1] - freqs[0])


def is_signal_frame_view_float64(signal: npt.NDArray[np.float64], frame_length_samples: int, hop_length_samples: int) -> bool:
    """
    Check whether the input signal can be cleanly viewed as overlapping frames of a given length, spaced a given hop distance apart.

    Parameters
    ----------
    signal:
        The input signal to check.

    frame_length_samples:
        The length of each frame, in samples, in the desired view of the input signal.

    hop_length_samples:
        The spacing of each frame, in samples, in the desired view of the input signal.
        For example, if the hop length is 100 samples, the first frame is assumed to start at signal[0],
        the second frame is assumed to start at signal[100], the third frame is assumed to start at
        signal[200], etc.

    Returns
    -------
    out:
        True if the input signal may be viewed as overlapping frames of length frame_length_samples spaced hop_length_samples apart,
        and False otherwise.
    """
    if len(signal) < frame_length_samples:
        return False
    
    # In this case, there is a single frame in the input signal.
    elif len(signal) == frame_length_samples:
        return True
    
    else:
        return (len(signal) - frame_length_samples) % hop_length_samples == 0


def pad_to_frame_view_float64(signal: npt.NDArray[np.float64], frame_length_samples: int, hop_length_samples: int, pad_value: np.float64 = 0.0) -> npt.NDArray[np.float64]:
    """
    Pad the last frame of the input signal so the output signal may be cleanly viewed as overlapping frames of a given length, spaced a given hop distance apart.

    Parameters
    ----------
    signal:
        The input signal to pad, if needed.

    frame_length_samples:
        The length of each frame, in samples, in the desired view of the input signal.
    
    hop_length_samples:
        The spacing of each frame, in samples, in the desired view of the input signal.
        For example, if the hop length is 100 samples, the first frame is assumed to start at signal[0],
        the second frame is assumed to start at signal[100], the third frame is assumed to start at
        signal[200], etc.
    
    pad_value:
        The value to pad the last frame of the input signal with, if needed. Defaults to 0.

    Returns
    -------
    out:
        The output signal padded such that it may be cleanly viewed as may be viewed as overlapping frames of length frame_length_samples spaced hop_length_samples apart.
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


def _rms_frames_to_average_rms_signal_float64(rms_frames: npt.NDArray[np.float64], frame_length_samples: int, hop_length_samples: int) -> npt.NDArray[np.float64]:
    """
    Produce an average RMS output signal given an input RMS signal.

    Parameters
    ----------
    rms_frames:
        The input RMS signal. This signal is expected to be produced by computing the RMS of overlapping frames of an original signal.
        I.e.: rms_frames[i] is the RMS of the ith frame of the original signal.

    frame_length_samples:
        The length of each frame, in samples, from which the ith RMS value in rms_frames was computed.
    
    hop_length_samples:
        The spacing between each frame, in samples, from which the RMS values in rms_frames were computed.
        For example, let "signal" be the original signal that rms_frames was computed from.
        If the hop length is 100 samples, the first frame is assumed to start at signal[0],
        the second frame is assumed to start at signal[100], the third frame is assumed to start at
        signal[200], etc.
    
    Returns
    -------
    average_rms:
        The output average RMS signal, with the same shape as the original signal from which rms_frames was computed from.
        Let the original signal from which rms_frames was computed from be called "signal."
        average_rms[i] is the average (mean) value of all the RMS frames that signal[i] belongs to.
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

    Parameters
    ----------
    signal:
        The input signal to compute the RMS over. This signal will be viewed as a sequence of overlapping frames. If it cannot be cleanly viewed as
        a sequence of overlapping frames, i.e.: the last frame has missing values, then the input signal will be padded with 0 until the last frame is full.
    
    frame_length_samples:
        The length of each frame, in samples, in the framed view of the input signal.
    
    hop_length_samples:
        The distance between each frame, in samples, in the framed view of the input signal.
        For example, if the hop length is 100 samples, the first frame is assumed to start at signal[0],
        the second frame is assumed to start at signal[100], the third frame is assumed to start at
        signal[200], etc.

    mode:
        Controls how the output RMS energy signal is computed. Defaults to "average".
        - "average": Each sample in the output signal is the average RMS value of all frames in the input signal at that sample's position.
        - "last" Each sample in the output signal is assigned the RMS value of the last, i.e.: frame with the largest starting index, that contains it.
    
    Returns
    -------
    out:
        The RMS energy signal. This output signal will have the same shape as the input signal if the input signal was padded such that it could be cleanly
        viwed as a series of overlapping frames of length frame_length_samples starting every hop_length_samples samples.
    """
    if mode not in {"average", "last"}:
        raise ValueError(f"Invalid mode: {mode}")

    padded_signal: npt.NDArray[np.float64] = signal

    if not is_signal_frame_view_float64(signal, frame_length_samples, hop_length_samples):
        padded_signal = pad_to_frame_view_float64(signal, frame_length_samples, hop_length_samples, 0.0)

    padded_signal_windows: npt.NDArray[np.float64] = np.lib.stride_tricks.sliding_window_view(padded_signal, window_shape=frame_length_samples, writeable=False)
    
    # Can consider this a "frame matrix"
    padded_signal_frames: npt.NDArray[np.float64] = padded_signal_windows[::hop_length_samples]

    padded_signal_frames_rms: npt.NDArray[np.float64] = np.sqrt(np.mean(padded_signal_frames**2, axis=1))

    if mode == "average":
        return _rms_frames_to_average_rms_signal_float64(padded_signal_frames_rms, frame_length_samples, hop_length_samples)
    
    else:
        return np.pad(np.repeat(padded_signal_frames_rms, hop_length_samples), (0, frame_length_samples - hop_length_samples), constant_values=padded_signal_frames_rms[-1])


def rms_to_db_float64(rms_signal: npt.NDArray[np.float64], epsilon: np.float64 = 1e-12) -> npt.NDArray[np.float64]:
    """
    Convert an input RMS signal into decibels (db).

    Parameters
    ----------
    rms_signal:
        The input RMS signal to convert.
    
    epsilon:
        A small constant to ensure values close to 0 in the input do not blow up to negative infinity in the output.
        Defaults to 1e-12.

    Returns
    -------
    out:
        The input signal converted to db. 
    """
    return 20 * np.log10(np.maximum(rms_signal, epsilon))


def stft_float64(signal: npt.NDArray[np.float64], sample_rate_hz: float, frame_length_samples: int, hop_length_samples: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128]]:
    """
    Take the Short Time Fourier Transform (STFT) of the input signal. Note that the input signal will be automatically padded to the desired frame view
    based on the frame length and hop length parameters.

    Parameters
    ----------
    signal:
        The signal to take the STFT of.
    
    sample_rate_hz:
        The sampling rate used to create the input signal.
    
    frame_length_samples:
        The length of each frame, in samples, in the framed view of the input signal.
    
    hop_length_samples:
        The distance between each frame, in samples, in the framed view of the input signal.
        For example, if the hop length is 100 samples, the first frame is assumed to start at signal[0],
        the second frame is assumed to start at signal[100], the third frame is assumed to start at
        signal[200], etc.
    
    Returns
    -------
    freqs, spectrum:
        A tuple containing the frequency buckets of the STFT and a 2D array of shape (num_frames, len(freqs)) where each row, i, is the FFT output spectrum for frame i.
    """

    padded_signal: npt.NDArray[np.float64] = signal

    if not is_signal_frame_view_float64(signal, frame_length_samples, hop_length_samples):
        padded_signal = pad_to_frame_view_float64(padded_signal, frame_length_samples, hop_length_samples)
    
    padded_signal_windows: npt.NDArray[np.float64] = np.lib.stride_tricks.sliding_window_view(padded_signal, window_shape=frame_length_samples, writeable=False)
    
    # Can consider this a "frames matrix"
    padded_signal_frames: npt.NDArray[np.float64] = padded_signal_windows[::hop_length_samples]

    spectrum: npt.NDArray[np.complex128] = np.fft.rfft(padded_signal_frames, axis=1, norm="forward")
    freqs: npt.NDArray[np.float64] = np.fft.rfftfreq(len(padded_signal_frames[0]), 1 / sample_rate_hz)

    return freqs, spectrum
