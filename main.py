from metallophone import GamelanMetallophone, GamelanMetallophoneLabel
from note import Note, NoteSymbol
from notecontainers import RepresentativeNotes, SampledNotes

import gamelanfiles
import plotting
import recording


def get_metallophones(file_to_label_map: dict[str, GamelanMetallophoneLabel], name_to_note_symbols_map: dict[str, NoteSymbol]) -> list[GamelanMetallophone]:
    """
    Create a list of GamelanMetallophone instances from the given information.

    Parameters
    ----------
    file_to_label_map:
        A dictionary with the names of .wav files as the keys, and GamelanMetallophoneLabels as the values.
        The .wav files contain audio recordings of consective key strikes for a particular metallophone.
        The key strikes in the recording are assumed to be ordered from lowest note (frequency) to highest note (frequency).
        Each key strike is expected to last for more than half a second, and two key strikes should not overlap.
        A given key may be struck multiple times in a row, in which case the average frequency of the repeated strikes is
        used for that key.
    
    note_to_symbols_map:
        A dictionary with the names of metallophones as the keys, and lists of NoteSymbols as the values.
        This maps a particular metallophone to the notes it has.

    Returns
    -------
    out:
        A list of GamellanMetallophone instances.
    """
    # For each metallophone recording file and metallophone label pair:
    # 1. Parse the frequencies from the recording file.
    # 2. Look up the note symbols associated with the metallophone in the notes map using the name from the label.
    # 3. Create a tuple of notes from the frequencies and the symbol.
    # 4. Create an instrument instance from the label and notes, then add the instrument instance to a list.
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
    
    return metallophones


def split_metallophones(metallophones: list[GamelanMetallophone]) -> tuple[GamelanMetallophone | None, list[GamelanMetallophone], list[GamelanMetallophone]]:
    """
    Split a list of GamelanMetallophones into the leading, upper, and lower metallophones.

    Parameters
    ----------
    metallophones:
        The list of GamelanMetallophones to split. There should be at most one leading metallophone in the list.

    Returns
    -------
    out:
        A tuple of the form (leading_metallophone, upper_metallophones, lower_metallophones) where leading_metallophone is a single GamelanMetallophone
        instance, and *_metallophones are lists of GamelanMetallophone instances. If there is no leading metallophone in the input list, None is returned
        in place of a leading metallophone instance.
    """
    leading_metallophone: GamelanMetallophone | None = None
    upper_metallophones: list[GamelanMetallophone] = []
    lower_metallophones: list[GamelanMetallophone] = []

    for metallophone in metallophones:
        if metallophone.is_upper:
            upper_metallophones.append(metallophone)
        
        elif metallophone.is_lower:
            lower_metallophones.append(metallophone)
        
        # The leading metallophone will be classified as neither the upper nor lower of a pair.
        elif leading_metallophone is None:
            leading_metallophone = metallophone

        else:
            raise ValueError(f"{split_metallophones.__name__}: Provided more than one leading metallophone in the input list.")
    
    return leading_metallophone, upper_metallophones, lower_metallophones


if __name__ == "__main__":
    file_to_label_map: dict[str, GamelanMetallophoneLabel] = gamelanfiles.parse_metallophone_labels_csv("metallophone_labels.csv")
    name_to_note_symbols_map: dict[str, NoteSymbol] = gamelanfiles.parse_metallophone_notes_csv("metallophone_notes.csv")

    metallophones: list[GamelanMetallophone] = get_metallophones(file_to_label_map, name_to_note_symbols_map)
    
    leading_metallophone, upper_metallophones, lower_metallophones = split_metallophones(metallophones)

    # Flattening the notes tuple stored in each metallophone before passing into SampledNotes constructor.
    lower_sampled_notes: SampledNotes = SampledNotes([note for metallophone in lower_metallophones for note in metallophone.notes])
    lower_representative_notes: RepresentativeNotes = RepresentativeNotes(sampled_notes=lower_sampled_notes)
    
    upper_sampled_notes: SampledNotes = SampledNotes([note for metallophone in upper_metallophones for note in metallophone.notes])
    upper_representative_notes: RepresentativeNotes = RepresentativeNotes(sampled_notes=upper_sampled_notes, reference_freq_hz=lower_representative_notes.get_reference_frequency_hz())

    # plotting.metallophone_spike_plot(upper_representative_notes, lower_representative_notes)
    plotting.metallophone_spike_plot(upper_representative_notes, lower_representative_notes, leading_metallophone)
