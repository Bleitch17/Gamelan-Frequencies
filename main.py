import recording


if __name__ == "__main__":
    recording_freqs: list[float] = recording.parse_metallophone_frequencies_from_wav("A.wav", True)    
    print(f"Recording Frequencies (Hz): {recording_freqs}, Length: {len(recording_freqs)}")

    # TODO - Create csv parsers.
    # From the metallophone names, get a list of file names and metallophone labels.
    # From the metallophone notes, get a map of instrument names -> note symbols.