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

    metallophones: list[GamelanMetallophone] = []

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

        notes: tuple[Note, ...] = tuple(Note(symbol, freq) for symbol, freq in zip(note_symbols, recording_freqs))

        metallophones.append(GamelanMetallophone(metallophone_label, notes))

    # To make the plots, need to create lines. The x-axis of the plot is in cents, and the y axis is in octaves (starting at octave 1).
    # To make a line, need to gather all notes with a particular symbol, from metallophones that are either the upper or lower in their pair,
    # and average the frequencies of each octave to create a representative note for that octave.
    #
    # Note that the ugal (leading metallophone) will be left out, since not sure whether it's upper, lower, or (most likely) both.
    #
    # Plan:
    # 1. Group the metallophones into two lists: upper and lower.
    # 2. For each metallophone group (upper, lower) create a nested dictionary structure with note_symbol -> octaves, octave -> frequency samples
    #    mappings.
    # 3. Combine the frequency samples into an average frequency, so the mapping looks like: note_symbol -> octaves, octave -> avg_frequency
    # 4. Combine the note symbol, octave, and avg_frequency to create representative notes.
    #    The mapping should look like: note_symbol -> representative notes (sorted non-decreasing by octave).
    # 5. From the two mappings for upper and lower notes, find the lowest avg frequency to use as the reference.
    #    This should be the lower I_1 frequency, but probably worth double checking.
    # 6. Using the reference frequency, create new upper and lower mappings as note_symbol -> cents, where the index of the cents value + 1 is the
    #    corresponding octave.
    # 7. Write a function that accepts the two mappings to create a plot. Optionally add the ugal points in as green dots to see where they lie.
