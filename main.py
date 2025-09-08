import recording


if __name__ == "__main__":
    recording_freqs: list[float] = recording.parse_metallophone_frequencies_from_wav("A.wav", True)    
    print(f"Recording Frequencies (Hz): {recording_freqs}, Length: {len(recording_freqs)}")

