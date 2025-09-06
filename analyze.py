
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from scipy.io import wavfile

import dsp
import blob
import threshold

# Need this line on Linux so I can pip install matplotlib locally for Python versions different than the system version.
# matplotlib.use("Qt5Agg")


def create_selected_audio_data_array(audio_data: npt.NDArray[np.float64], blob_boundaries: list[tuple[int, int]]) -> npt.NDArray[np.float64]:
    selected_mask: npt.NDArray[np.bool] = np.zeros_like(audio_data).astype(np.bool)

    for start_index, end_index in blob_boundaries:
        selected_mask[start_index:end_index+1] = True

    selected_audio_data: npt.NDArray[np.float64] = audio_data.copy()
    selected_audio_data[~selected_mask] = np.nan

    return selected_audio_data


def plot_timeseries_data(arrays: list[npt.NDArray], h_lines: list[float], sample_rate_hz: float) -> None:
    figure: plt.Figure = plt.figure(figsize=(30, 10))
    figure.suptitle("Gamelan Frequency Analysis")

    t: npt.NDArray[np.float64] = np.linspace(0, arrays[0].shape[0] / sample_rate_hz, arrays[0].shape[0])

    for array in arrays:
        plt.plot(t, array)
    
    for h_line in h_lines:
        plt.axhline(h_line)

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Audio Waveform")

    plt.tight_layout()
    plt.show()


def plot_spectrum(blob_data: npt.NDArray[np.float64], sample_rate_hz: float, freq_cutoff_hz: float) -> None:
    freqs, spectrum = dsp.fft_float64(blob_data, sample_rate_hz)
    
    freqs = freqs[freqs < freq_cutoff_hz]
    spectrum = spectrum[:len(freqs)]

    magnitude: npt.NDArray[np.float128] = np.abs(spectrum)

    print(freqs[np.argmax(magnitude)])

    plt.figure(figsize=(24, 12))
    plt.plot(freqs, magnitude)
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)

    plt.show()


def convert_audio_to_normalized_float64(audio_data: npt.NDArray) -> npt.NDArray[np.float64]:
    """
    Handles:
    - Unsigned 8-bit PCM
    - Signed 16/24/32-bit PCM
    - 32-bit float WAVs
    """

    if np.issubdtype(audio_data.dtype, np.integer):
        if data.dtype == np.uint8:
            # Shift to [-128, 127]
            shifted_data: npt.NDArray[np.int16] = audio_data.astype(np.int16) - 128
            
            # Convert to [-1.0, 1.0) by dividing by (127 + 1) = 128.
            return shifted_data.astype(np.float64) / (np.iinfo(np.int8).max + 1)

        else:
            return audio_data.astype(np.float64) / (np.iinfo(audio_data.dtype).max + 1)
    
    elif np.issubdtype(audio_data.dtype, np.floating):
        if np.any(np.abs(audio_data) > 1.0):
            print(f"Warning: values greater than 1.0 detected in audio data with floating point type {audio_data.dtype}. Clipping to [-1.0, 1.0].")
            
            return np.clip(audio_data, -1.0, 1.0).astype(np.float64)
        
        return audio_data.astype(np.float64)


def test_plot() -> None:
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


if __name__ == "__main__":
    file_name: str = "wav/K.wav"
    
    sample_rate_hz, data = wavfile.read(file_name)
    print(f"Sample rate: {sample_rate_hz}Hz")
    print(f"Audio data shape: {data.shape}")
    print(f"Raw audio data type: {data.dtype}")

    # Extracting the first channel (assuming both channels are the same).
    normalized_audio_data: npt.NDArray[np.float64] = convert_audio_to_normalized_float64(data)[:, 0]
    normalized_audio_data = normalized_audio_data

    frame_length_samples: int = math.floor(0.01 * sample_rate_hz)
    hop_length_samples: int = math.floor(0.005 * sample_rate_hz)

    padded_normalized_audio_data: npt.NDArray[np.float64] = dsp.pad_to_frame_view_float64(normalized_audio_data, frame_length_samples, hop_length_samples, pad_value=0.0)

    audio_data_rms: npt.NDArray[np.float64] = dsp.rms_float64(padded_normalized_audio_data, frame_length_samples, hop_length_samples, mode="average")

    t: float = threshold.compute_threshold_float64(audio_data_rms, delta=1.02)
    
    min_blob_size: int = int(0.5 * sample_rate_hz)
    
    blob_boundaries: list[tuple[int, int]] = blob.get_blob_boundaries_above_threshold(audio_data_rms, t)
    blob_boundaries = blob.filter_blobs_by_size(blob_boundaries, 0.5 * sample_rate_hz)

    # Uncomment the lines below to debug selection using the runs.
    # selected_audio_data: npt.NDArray[np.float64] = create_selected_audio_data_array(padded_normalized_audio_data, blob_boundaries)
    # plot_timeseries_data([padded_normalized_audio_data, audio_data_rms, selected_audio_data], [t], sample_rate_hz)

    blobs: list[npt.NDArray[np.float64]] = blob.get_blobs(padded_normalized_audio_data, blob_boundaries)

    blob_frequencies: list[float] = list(map(lambda blob: float(dsp.dominant_freq_float64(blob, sample_rate_hz)), blobs))

    print(f"{file_name}: {blob_frequencies}")

    # stft_frame_length_samples: int = int(2 * sample_rate_hz)
    # stft_hop_length_samples: int = int(stft_frame_length_samples // 2)

    # stft_freqs, stft_spectrum = dsp.stft_float64(blobs[0], sample_rate_hz, stft_frame_length_samples, stft_hop_length_samples)

    # max_freq_indices: npt.NDArray[np.int32] = np.abs(stft_spectrum).argmax(axis=1)
    # max_freqs: npt.NDArray[np.float64] = stft_freqs[max_freq_indices]
    # print(max_freqs)

    # plot_timeseries_data([max_freqs], [], sample_rate_hz)
