import gamelan_files
import recording


if __name__ == "__main__":
    recording_freqs: list[float] = recording.parse_metallophone_frequencies_from_wav("A.wav", False)    
    print(f"Recording Frequencies (Hz): {recording_freqs}, Length: {len(recording_freqs)}")

    print(gamelan_files.parse_metallophone_labels_csv("metallophone_labels.csv"))
    print(gamelan_files.parse_metallophone_notes_csv("metallophone_notes.csv"))
