from dataclasses import dataclass


@dataclass(frozen=True)
class NoteSymbol:
    """
    A class to represent a note symbol, e.g.: A_4, C#-3, etc.

    Parameters
    ----------
    name:
        The name of the note.

    octave:
        The octave the note belongs to, starting from 1.
    
    separator:
        The separator string used in the string representation of the note symbol.
        Defaults to "_".
    
    Attributes
    ----------
    name:
        The name of the note.

    octave:
        The octave the note belongs to, starting from 1.
    
    separator:
        The separator string used in the string representation of the note symbol.
    """

    name: str
    octave: int
    separator: str = "_"

    def __post_init__(self):
        if not self.name or not self.separator:
            raise ValueError(f"{type(self).__name__}: Empty name: {self.name!r} or empty separator: {self.separator!r}")
        
        if ' ' in self.name:
            raise ValueError(f"{type(self).__name__}: A valid name should not have whitespace, instead got: {self.name!r}")

        if self.octave < 1:
            raise ValueError(f"{type(self).__name__}: Octave must be at least 1, instead got: {self.octave!r}")

    def __str__(self) -> str:
        return f"{self.name}{self.separator}{self.octave}"

    @classmethod
    def from_string(cls, note_string: str, separator: str = "_") -> "NoteSymbol":
        """
        Create a NoteSymbol instance from a string representation, e.g.: "A_4", "C#-3", etc.

        Parameters
        ----------
        note_string:
            The string representation of the note. Expected format is {name}{separator}{octave},
            where name is any string, separator is any string, and octave is a positive integer.
        
        separator:
            The separator used in the string representation of the note symbol. Should be the same
            separator used in note_string.
            Defaults to "_".

        Returns
        -------
        NoteSymbol
            The constructed NoteSymbol instance.
        """
        if not note_string or not separator:
            raise ValueError(f"{cls.__name__}: Empty note_string: {note_string!r} or empty separator: {separator!r}")
        
        if separator not in note_string:
            raise ValueError(f"{cls.__name__}: Separator {separator!r} not contained in note string: {note_string!r}")

        note_symbol_segments: list[str] = note_string.split(separator)

        if len(note_symbol_segments) != 2:
            raise ValueError(f"{cls.__name__}: Expected separator {separator!r} to appear at most once in note string: {note_string!r}")
        
        name, octave = note_symbol_segments

        if not octave.isdigit():
            raise ValueError(f"{cls.__name__}: Expected octave to be digits, instead got {octave!r}")
        
        octave = int(octave)

        if octave < 1:
            raise ValueError(f"{cls.__name__}: Octave must be at least 1, instead got: {octave!r}")
        
        return cls(name, octave, separator)


@dataclass(frozen=True)
class Note:
    """
    A class for representing musical notes.

    Parameters
    ----------
    symbol:
        The sybol of the note.

    freq_hz:
        The frequency, in Hz, of the tone the note represents.

    Attributes
    ----------
    symbol:
        The sybol of the note.

    freq_hz:
        The frequency, in Hz, of the tone the note represents.
    """

    symbol: NoteSymbol
    freq_hz: float

    def __post_init__(self):
        if self.freq_hz <= 0.0:
            raise ValueError(f"{type(self).__name__}: Expected a frequency above 0 Hz, instead got: {self.req_hz!r}")
    
    def __str__(self) -> str:
        return str(self.symbol)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(symbol={self.symbol!r}, freq_hz={self.freq_hz!r})"
