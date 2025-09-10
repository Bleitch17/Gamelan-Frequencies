from metallophone import GamelanMetallophone, GamelanMetallophoneLabel
from note import Note, NoteSymbol

import gamelan_files
import recording


if __name__ == "__main__":
    # For each metallophone recording file and metallophone label pair:
    # 1. Parse the frequencies from the recording file.
    # 2. Look up the note symbols associated with the metallophone in the notes map using the name from the label.
    # 3. Create a tuple of notes from the frequencies and the symbol.
    # 4. Create an instrument instance from the label and notes, then add the instrument instance to a list.
    
    file_to_label_map: dict[str, GamelanMetallophoneLabel] = gamelan_files.parse_metallophone_labels_csv("metallophone_labels.csv")
    name_to_note_symbols_map: dict[str, NoteSymbol] = gamelan_files.parse_metallophone_notes_csv("metallophone_notes.csv")

    for wav_file_name, metallophone_label in file_to_label_map.items():
        recording_freqs: list[float] = []
        note_symbols: list[NoteSymbol] = name_to_note_symbols_map[metallophone_label.name]

        # Workarounds for ugal recording, contained some recording errors?
        # From looking at the recording, it appears the first 3 notes were repeated at the end?
        if wav_file_name == "I.wav":
            recording_freqs = recording.parse_metallophone_frequencies_from_wav(
                wav_file_name,
                rms_threshold_ratio_delta=1.055,
                min_strike_length_s=3.0,
                repeated_strike_delta_hz=3.0
            )[:len(note_symbols)]

        else:
            recording_freqs = recording.parse_metallophone_frequencies_from_wav(wav_file_name)

        if len(recording_freqs) != len(note_symbols):
            print(f"Error parsing file: {wav_file_name}. Parsed {len(recording_freqs)} frequencies but expected {len(note_symbols)} notes.")
