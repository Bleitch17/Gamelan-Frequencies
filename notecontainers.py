from collections import defaultdict
from collections.abc import Iterable

from metallophone import GamelanMetallophone
from note import Note

import cent


class SampledNotes:
    """
    A class for storing collections of note names, their associated octaves, and per-octave frequency samples (in Hz).

    Parameters
    ----------
    notes:
        A collection of notes to initialize the SampledNotes instance with. May be empty, in which case the SampledNotes
        instance is initialized to empty. Additional collections of notes may be added after initialization.
    """

    def __init__(self, notes: list[Note]):
        self._map: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

        for note in notes:
            self._add_note(note)

    def _add_note(self, note: Note) -> None:
        """
        Private helper method to add a single note to this collection.

        Parameters
        ----------
        note:
            The note to add to this collection.
        """

        name: str = note.symbol.name
        octave: int = note.symbol.octave
        freq_hz: float = note.freq_hz

        self._map[name][octave].append(freq_hz)

    def add_notes(self, notes: Note | Iterable[Note]) -> None:
        """
        Add either a single note or an iterable collection of notes to this collection.

        Parameters
        ----------
        notes:
            The note(s) to add to this collection.
        """

        if isinstance(notes, Iterable):
            for note in notes:
                self._add_note(note)
            
        else:
            self._add_note(note)
    
    def add_metallophone_notes(self, metallophone: GamelanMetallophone) -> None:
        """
        Add the notes from a GamelanMetallophone instance to this collection.

        Parameters
        ----------
        metallophone:
            The metallophone to add the notes from.
        """

        self.add_notes(metallophone.notes)

    def get_note_names(self) -> list[str]:
        """
        Retrieve all the unique note names stored in this SampledNotes instance.

        Returns
        -------
        out:
            A list of the unique note names stored in this SampledNotes instance.
        """

        return list(self._map.keys())
    
    def get_note_octaves(self, note_name: str) -> list[int]:
        """
        Retrieve all the octaves stored in this SampledNotes instance for a specified note name.

        Parameters
        ----------
        note_name:
            The name of the note to get the octaves for.
        
        Returns
        -------
        out:
            A list of all octaves stored for the specified note name.
        """

        return list(self._map.get(note_name, {}).keys())

    def get_average_note_sample(self, note_name: str, octave: int) -> float:
        """
        Calculate the average (mean) frequency sample for a specified note at a specified octave.

        Parameters
        ----------
        note_name:
            The name of the note to calculate the average sample for.

        octave:
            The octave to calculate the average sample for.

        Returns
        -------
        out:
            The average (mean) frequency sample for a specified note at a specified octave.
        """

        octave_map: dict[int, list[float]] = self._map.get(note_name, {})

        if not octave_map:
            raise ValueError(f"{type(self).__name__}: No notes with the given name: {note_name} are stored in this collection.")
        
        samples: list[float] = octave_map.get(octave, [])

        if not samples:
            raise ValueError(f"{type(self).__name__}: The given octave: {octave} is not stored for the given note: {note_name} in this collection.")

        return sum(samples) / len(samples)


class RepresentativeNotes:
    """
    A class for retrieving notes represented in cents.

    Parameters
    ----------
    sampled_notes:
        A SampledNotes instance. In the case where a note (name + octave) has more than one frequency sample, the average (mean) frequency is
        computed for that note before being converted to cents.

    reference_freq_hz:
        The reference frequency to use for the cents calculation. If a reference is not provided, the lowest average (mean) frequency in
        sampled_notes is used as the reference.
    """

    def __init__(self, sampled_notes: SampledNotes, reference_freq_hz: float | None = None):
        self._map: dict[str, dict[int, float]] = defaultdict(defaultdict)
        self._ref_freq_hz: float | None = None

        for note_name in sampled_notes.get_note_names():
            for octave in sampled_notes.get_note_octaves(note_name):
                self._map[note_name][octave] = sampled_notes.get_average_note_sample(note_name, octave)

        # If no reference frequency is given, search for the lowest avg frequency to use as the reference.
        if reference_freq_hz is None:
            for note_name in self._map.keys():
                for octave in self._map[note_name].keys():
                    freq_hz: float = self._map[note_name][octave]
                    
                    if ref_freq_hz is None:
                        ref_freq_hz = freq_hz
                    else:
                        ref_freq_hz = freq_hz if freq_hz < ref_freq_hz else ref_freq_hz

        else:
            ref_freq_hz = reference_freq_hz
        
        # Perform the cents calculations using ref_freq_hz
        for note_name in self._map.keys():
            for octave in self._map[octave].keys():
                freq_hz: float = self._map[note_name][octave]

                self._map[note_name][octave] = cent.cents_freqs(ref_freq_hz, freq_hz)
        
    def get_reference_frequency_hz(self) -> float:
        """
        Get the reference frequency, in Hz, used in the cents calculation.

        Returns
        -------
        out:
            The reference frequency, in Hz.
        """

        if self._ref_freq_hz is None:
            raise RuntimeError(f"{type(self).__name__}: The stored reference frequency is None.")

        return self._ref_freq_hz
