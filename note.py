from dataclasses import dataclass


@dataclass(frozen=True)
class Note:
    """
    A class for representing musical notes.

    Parameters
    ----------
    symbol:
        The sybol of the note. This symbol will be used as the string representation of the note.

    freq_hz:
        The frequency, in Hz, of the tone the note represents.

    Attributes
    ----------
    symbol:
        The sybol of the note. This symbol will be used as the string representation of the note.

    freq:
        The frequency, in Hz, of the tone the note represents.
    """

    def __init__(self, symbol: str, freq_hz: float) -> None:
        self.symbol: str = symbol
        self.freq: float = freq_hz
    
    def __str__(self) -> str:
        return self.symbol

    def __repr__(self) -> str:
        return f"Note(symbol={self.symbol}, freq={self.freq})"
