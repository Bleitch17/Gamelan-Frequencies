import matplotlib

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

import dsp

# Need this line on Linux so I can pip install matplotlib locally for Python versions different than the system version.
matplotlib.use("Qt5Agg")


def plot_signals_float64(signals: list[npt.NDArray[np.float64]], sample_rate_hz: float, horizontal_lines: list[float]) -> None:
    """
    Display a matplotlib plot with one or more lines given an input array of signals.

    Parameters
    ----------
    signals:
        The list of signals to plot. Note that if any sample in a signal is np.nan, it will not appear on the plot.
        Assumes all signals are 1D, and have the same length. Assumes all signals were created by sampling at the same rate.

    sample_rate_hz:
        The sampling rate at which the input signals were created.

    horizontal_lines:
        A list of y values to plot, each as a flat horizontal line.
    """
    figure: plt.Figure = plt.figure(figsize=(30, 10))
    figure.suptitle(f"Signals, Sample Rate = {sample_rate_hz:.1f}")

    time_marks: npt.NDArray[np.float64] = np.linspace(0, signals[0].shape[0] / sample_rate_hz, signals[0].shape[0])

    for signal in signals:
        plt.plot(time_marks, signal)
    
    for value in horizontal_lines:
        plt.axhline(value)
    
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Audio Waveform")

    plt.tight_layout()
    plt.show()


def plot_spectrum_float64(signal: npt.NDArray[np.float64], sample_rate_hz: float, frequency_range_hz: tuple[float, float]) -> None:
    """
    Display a matplotlib plot of the Fourier Spectrum of the input signal within a given frequency range.

    Parameters
    ----------
    signal:
        The input signal to plot the spectrum of.

    sample_rate_hz:
        The sampling rate at which the input signals were created.
    
    frequency_range_hz:
        A tuple of the form (min, max) where min and max bound the smallest and largest frequencies to show in the plot,
        I.e.: no frequency smaller than or equal to min and larger then or equal to max will appear in the plot.
    """
    freqs, spectrum = dsp.fft_float64(signal, sample_rate_hz)
    min_freq_hz, max_freq_hz = frequency_range_hz

    freqs = freqs[freqs < max_freq_hz]
    spectrum = spectrum[:len(freqs)]

    freqs = freqs[freqs > min_freq_hz]
    spectrum = spectrum[-len(freqs):]

    magnitude: npt.NDArray[np.float128] = np.abs(spectrum)

    plt.figure(figsize=(24, 12))
    plt.plot(freqs, magnitude)
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)

    plt.show()


def plot_test() -> None:
    """
    Testing some capabilities of matplotlib.
    """
    x = np.linspace(0, 2*np.pi, 50)

    fig, ax = plt.subplots()

    line1, = ax.plot(x, np.sin(x), linestyle='-', color='C0', label='sin')
    line2, = ax.plot(x, np.cos(x), linestyle='--', color='C1', label='cos')

    theta = np.linspace(0, 2*np.pi, 40)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)

    line3, = ax.plot(circle_x, circle_y, linestyle='-', color='C2', label='circle')

    # Add new points to one of the lines (extend data)
    new_x = np.linspace(2*np.pi, 3*np.pi, 20)
    new_y = np.sin(new_x)

    # Concatenate old and new points
    xdata = np.concatenate([line1.get_xdata(), new_x])
    ydata = np.concatenate([line1.get_ydata(), new_y])

    # Update the line’s data
    line1.set_data(xdata, ydata)

    # Rescale axes to fit the updated data
    ax.relim()
    ax.autoscale_view()

    # Decorate
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True)

    plt.show()
