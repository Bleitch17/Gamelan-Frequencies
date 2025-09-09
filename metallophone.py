from dataclasses import dataclass

from note import Note


UPPER_PAIR_STR_SUFFIX: str = "+"
LOWER_PAIR_STR_SUFFIX: str = "-"


@dataclass(frozen=True)
class GamelanMetallophoneLabel:
    """
    A class to represent a label format for the Metallophone instrument used in Gamelan music.

    Parameters
    ----------
    name:
        The name of the instrument.

    is_upper:
        True if the instrument is the upper metallophone of a tuned pair, otherwise False.

    is_lower:
        True if the instrument is the lower metallophone of a tuned pair, otherwise False.

    Attributes
    ----------
    name:
        The name of the instrument.

    is_upper:
        True if the instrument is the upper metallophone of a tuned pair, otherwise False.

    is_lower:
        True if the instrument is the lower metallophone of a tuned pair, otherwise False.
    """

    name: str
    is_upper: bool
    is_lower: bool

    def __post_init__(self):
        if not self.name:
            raise ValueError(f"{type(self).__name__}: Expected non-empty name.")
        
        if ' ' in self.name:
            raise ValueError(f"{type(self).__name__}: A valid name should not include whitespace, instead got: {self.name!r}")

        if self.is_upper and self.is_lower:
            raise ValueError(f"{type(self).__name__}: A Metallophone cannot both be the upper and lower Metallophone of a tuned pair.")

    @classmethod
    def from_string(cls, label_string: str) -> "GamelanMetallophoneLabel":
        """
        Create a GamelanMetallophoneLabel from a string representation.

        Parameters
        ----------
        label_string:
            The string representation of the label. Expected format is {name}{+ or - or empty}.
            The + indicates this is the upper metallophone of a tuned pair, the - indicates this is the lower.
            If no + or - is provided, then this is the leading instrument, not part of a tuned pair.

        Returns
        -------
        GamelanMetallophoneLabel
            The constructed GamelanMetallophoneLabel instance.
        """
        if len(label_string) <= 1:
            raise ValueError(f"{cls.__name__}: A valid label string must have more than 1 character, instead got: {label_string!r}")

        if ' ' in label_string:
            raise ValueError(f"{cls.__name__}: A valid label string should not have whitespace, instead got: {label_string!r}")

        if label_string[:-1].count(UPPER_PAIR_STR_SUFFIX) or label_string[:-1].count(LOWER_PAIR_STR_SUFFIX):
            raise ValueError(f"{cls.__name__}: A valid label string should include at most one '+' or one '-' at the end of the string, instead got: {label_string!r}")

        if label_string.endswith(UPPER_PAIR_STR_SUFFIX):
            return cls(label_string[:-1], True, False)

        elif label_string.endswith(LOWER_PAIR_STR_SUFFIX):
            return cls(label_string[:-1], False, True)
        
        else:
            return cls(label_string, False, False)


class GamelanMetallophone:
    """
    A class to represent the Metallophone instrument used in Gamelan music.

    Parameters
    ----------
    label:
        The label of the Metallophone, e.g.: jegogan+, kantilan-, ugal, etc.
    
    notes:
        The notes the instrument has.

    Attributes
    ----------
    name:
        The name of the Metallophone, e.g.: jegogan, kantilan, ugal, etc.
    
    notes:
        The notes the instrument has.

    is_lower:
        If true, indicates this metallophone is the lower of a tuned pair of metallophones.
    
    is_upper:
        If true, indicates this metallophone is the upper of a tuned pair of metallophones.
    """

    def __init__(self, label: GamelanMetallophoneLabel, notes: tuple[Note, ...]):
        self.name: str = label.name
        self.notes: tuple[Note, ...] = tuple([note for note in notes])
        self.is_lower: bool = label.is_lower
        self.is_upper: bool = label.is_upper

    def __repr__(self) -> str:
        return f"{type(self).__name__}(label={GamelanMetallophoneLabel(self.name, self.is_upper, self.is_lower)!r}, notes={self.notes!r})"
