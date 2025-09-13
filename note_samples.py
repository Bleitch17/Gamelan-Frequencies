from collections import defaultdict

from metallophone import GamelanMetallophone
from note import Note


class NoteSamples:
    """
    A class for storing collections of note names, their associated octaves, and per-octave frequency samples (in Hz).

    Parameters
    ----------
    notes:
        A collection of notes to initialize the NoteSamples instance with. May be empty, in which case the NoteSamples
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
