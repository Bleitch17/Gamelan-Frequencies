from note import Note


class Metallophone:
    """
    A class to represent the Metallophone instrument used in Gamelan.

    Parameters
    ----------
    name:
        The name of the Metallophone, e.g.: jegogan, kantilan, ugal, etc.
    
    wav_file_path:
        A path to a .wav file containing key strikes for all the keys this instrument has.
        These key strikes are expected to be ordered from lowest note (frequency) to highest note (frequency).
        A particular key may be struck more than once in a row.
        Each key strike is expected to last at least 0.5 seconds, and two different key strikes may not overlap.

    note_symbols:
        A collection of note symbols corresponding to the notes represented by this instrument.
        The symbols are expected to be ordered from lowest note to highest note, aligning with the order of the key
        strikes in the recording .wav file.

    is_lower:
        If true, indicates this metallophone is the lower of a tuned pair of metallophones.
    
    is_upper:
        If true, indicates this metallophone is the upper of a tuned pair of metallophones.

    Attributes
    ----------
    name:
        The name of the Metallophone, e.g.: jegogan, kantilan, ugal, etc.
    
    
    """

    def __init__(self, name: str, wav_file_path: str, note_symbols: tuple[str, ...], is_lower: bool, is_upper: bool) -> None:
        pass