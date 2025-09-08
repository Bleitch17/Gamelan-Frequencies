import math

from note import Note


def cents_freqs(ref_freq_hz: float, freq_hz: float) -> float:
    """
    Returns the distance, in cents, of a specified frequency from a reference frequency.

    A cent is a logarithmic unit of measure used for musical intervals.
    An octave is defined as a doubling in frequency. If one note is twice the frequency of a second note,
    the first note is an octave above the second note.
    Twelve-tone equal temperament divides the octave ino 12 semitones of 100 cents each.
    
    See: https://en.wikipedia.org/wiki/Cent_(music)

    Parameters
    ----------
    ref_freq_hz:
        The reference frequency, in Hz.

    freq_hz:
        A frequency, in Hz, to compare to the reference frequency.

    Returns
    -------
    out:
        The distance, in cents, of freq_hz from ref_freq_hz.
        out = 1200 * log2(freq_hz / ref_freq_hz)
    """
    return 1200 * math.log2(freq_hz / ref_freq_hz)


def cents_notes(ref_note: Note, note: Note) -> float:
    """
    Returns the distance, in cents, of a specified note from a reference note.

    A cent is a logarithmic unit of measure used for musical intervals.
    An octave is defined as a doubling in frequency. If one note is twice the frequency of a second note,
    the first note is an octave above the second note.
    Twelve-tone equal temperament divides the octave ino 12 semitones of 100 cents each.
    
    See: https://en.wikipedia.org/wiki/Cent_(music)

    Parameters
    ----------
    ref_note:
        The reference note.

    note:
        A note to compare with the reference note.

    Returns
    -------
    out:
        The distance, in cents, of the specified note from the reference note.
    """
    return cents_freqs(ref_note.freq, note.freq)
