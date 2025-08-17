import math

import numpy as np
import numpy.typing as npt


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


def _pad_frames_float64(signal: npt.NDArray[np.float64], frame_length_samples: int, hop_length_samples: int, pad_value: np.float64) -> npt.NDArray[np.float64]:
    """
    Pad the last frame of the input signal so the output signal may be cleanly viewed as frames of a given length with starting samples separated by the hop length.

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

    # Assume there will be at least one frame.
    if signal_length_samples < frame_length_samples:
        pad_length_samples = frame_length_samples - signal_length_samples
    
    elif signal_length_samples % hop_length_samples == 0:
        pad_length_samples = frame_length_samples - hop_length_samples
    
    else:
        pad_length_samples = frame_length_samples - (signal_length_samples % hop_length_samples)
    
    return np.pad(signal, (0, pad_length_samples), constant_values=pad_value)


def rms_float64(signal: npt.NDArray[np.float64], frame_length_s: float, hop_length_s: float, sample_rate_hz: float, pad_value: np.float64 = 0.0) -> npt.NDArray[np.float64]:
    """
    Computes the RMS energy from an input signal.

    :param signal: The input signal to compute the RMS over.
    :param frame_length_s: The length of each frame of the input signal, in seconds, over which to compute the RMS over.
    :param hop_length_s: The distance between the starting point of each frame, in seconds.
    :param sample_rate_hz: The sampling rate, in Hz.
    :param pad_value: The value with which to pad the input signal, in the case where the last frame is smaller than the provided frame length.
    Defaults to 0.

    :return: The RMS of signal. TODO - What is the shape, etc.?
    """ 
    frame_length_samples: int = math.floor(frame_length_s * sample_rate_hz)
    hop_length_samples: int = math.floor(hop_length_s * sample_rate_hz)

    padded_signal: npt.NDArray[np.float64] = _pad_frames_float64(signal, frame_length_samples, hop_length_samples, pad_value)

    padded_signal_windows: npt.NDArray[np.float64] = np.lib.stride_tricks.sliding_window_view(padded_signal, window_shape=frame_length_samples, writeable=False)
    padded_signal_frames: npt.NDArray[np.float64] = padded_signal_windows[::hop_length_samples]

    padded_signal_frames_rms: npt.NDArray[np.float64] = np.sqrt(np.mean(padded_signal_frames**2, axis=1))

    return np.repeat(padded_signal_frames_rms, hop_length_samples)[:len(signal)]
