import matplotlib

import numpy as np
import numpy.typing as npt
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

import cent
import dsp

from metallophone import GamelanMetallophone
from notecontainers import RepresentativeNotes

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
    figure.suptitle(f"Signals, Sample Rate = {sample_rate_hz:.1f}Hz")

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


def plot_test_lines() -> None:
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


def plot_test_numerals() -> None:
    x = np.arange(5)
    y = np.arange(5)

    plt.plot(x, y, marker="o")

    yticks = np.arange(5)
    ylabels = ["I", "II", "III", "IV", "V"]

    plt.yticks(yticks, ylabels)

    # Optional: limit y-axis to only that range
    plt.ylim(-0.5, 4.5)
    plt.grid(True)

    plt.show()


def metallophone_to_coordinates(metallophone: GamelanMetallophone, reference_freq_hz: float) -> list[tuple[int, float]]:
    """
    Convert the notes stored in a given GamelanMetallophone instance to coordinates (octave, cents) pairs.

    Parameters
    ----------
    metallophone:
        The GamelanMetallophone instance that contains the notes to convert to coordinates.

    reference_freq_hz:
        A reference frequency to use in the cents calculation.

    Returns
    -------
    out:
        A list of (octave, cents) tuples.
    """

    return list(map(lambda note: (note.symbol.octave, cent.cents_freqs(reference_freq_hz, note.freq_hz)), metallophone.notes))


def metallophone_spike_plot(upper_representative_notes: RepresentativeNotes, lower_representative_notes: RepresentativeNotes, leader: GamelanMetallophone | None = None) -> None:
    """
    Create a spike plot for a metallophone family, given the representative notes from the upper and lower instruments.

    Parameters
    ----------
    upper_representative_notes:
        A RepresentativeNotes instance for the upper metallophones in the family.
    
    lower_representative_notes:
        A RepresentativeNotes instance for the lower metallophones in the family.

    leader:
        A GamelanMetallophone instance for the leading metallophone in the family. May be None, in which case it will not be included
        in the plot.
    """

    _, ax = plt.subplots()

    # Use Roman Numeral labels for the octaves.
    yticks: npt.NDArray[np.integer] = np.arange(6)[1:]
    ylabels = ["I", "II", "III", "IV", "V"]

    plt.yticks(yticks, ylabels)

    upper_note_bases: dict[str, float] = {}
    lower_note_bases: dict[str, float] = {}

    for note_name in upper_representative_notes.get_note_names():
        coords: list[tuple[int, float]] = upper_representative_notes.get_octaves(note_name)

        octaves: list[int] = list(map(lambda coord: coord[0], coords))
        cents: list[float] = list(map(lambda coord: coord[1] + 100.0 - ((coord[0] - 1) * 1200), coords))

        upper_note_bases[note_name] = cents[0]

        ax.plot(cents, octaves, linestyle="-", color="orange")

    for note_name in lower_representative_notes.get_note_names():
        coords: list[tuple[int, float]] = lower_representative_notes.get_octaves(note_name)

        octaves: list[int] = list(map(lambda coord: coord[0], coords))
        cents: list[float] = list(map(lambda coord: coord[1] + 100.0 - ((coord[0] - 1) * 1200), coords))

        lower_note_bases[note_name] = cents[0]

        ax.plot(cents, octaves, linestyle="-", color="blue")

    if leader is not None:
        coords: list[tuple[int, float]] = metallophone_to_coordinates(leader, lower_representative_notes.get_reference_frequency_hz())

        octaves: list[int] = list(map(lambda coord: coord[0], coords))
        cents: list[float] = list(map(lambda coord: coord[1] + 100.0 - ((coord[0] - 1) * 1200), coords))

        ax.plot(cents, octaves, marker="o", linestyle="None", color="lime")

    # Determining the custom x ticks.
    x_tick_locations: list[float] = []
    x_tick_symbols: list[str] = []

    for note_name in upper_note_bases.keys():
        lower_point: float = lower_note_bases[note_name]
        upper_point: float = upper_note_bases[note_name]

        x_tick_locations.append((lower_point + upper_point) / 2)
        x_tick_symbols.append(note_name)

    ax.tick_params(axis="x", which="both", bottom=False, top=False)

    plt.xticks(x_tick_locations, x_tick_symbols)

    # Creating proxy artists for the plot legend:
    proxy_artists = [mlines.Line2D([], [], color="blue"), mlines.Line2D([], [], color="orange")]
    labels: list[str] = ["Lower", "Upper"]

    if leader is not None:
        proxy_artists.append(mlines.Line2D([], [], color="lime"))
        labels.append("Leader")

    ax.legend(proxy_artists, labels)

    ax.grid(axis="y")
    ax.set_xlabel("Cents")
    ax.set_ylabel("Octaves")

    plt.show()
