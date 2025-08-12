import numpy as np
import numpy.typing as npt


def fft(signal: npt.NDArray[np.float64], sample_frequency_hz: float) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128]]:
    """
    Perform the Fast Fourier Transform (FFT) on an input signal.

    Returns a tuple (frequencies, spectrum).
    """
    spectrum = np.fft.rfft(signal, norm="forward")
    freqs = np.fft.rfftfreq(len(signal), 1 / sample_frequency_hz)

    return freqs, spectrum


def ifft(spectrum: npt.NDArray[np.complex128]) -> npt.NDArray[np.float64]:
    """
    Perform the Inverse FFT on an input spectrum.
    The input spectrum is expected to have been produced by the dsp.fft function.

    Returns the output of the iFFT operation.
    """
    
    return np.fft.irfft(spectrum, norm="forward")
