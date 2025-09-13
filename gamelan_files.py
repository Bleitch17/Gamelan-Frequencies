from metallophone import GamelanMetallophoneLabel
from note import NoteSymbol

import pandas as pd


FILE_NAME_COL_HEADER: str = "FileName"
METALLOPHONE_LABEL_COL_HEADER: str = "MetallophoneLabel"
METALLOPHONE_NAME_COL_HEADER: str = "MetallophoneName"
SUM_COL_HEADER: str = "Sum"


def parse_metallophone_labels_csv(metallophone_labels_csv_path: str) -> dict[str, GamelanMetallophoneLabel]:
    """
    Parse a map of key strike recording file names to metallophone labels from the Metallophone Labels .csv file.

    The Metallophone Labels .csv file is expected to have two columns, and one header row.
    The header of the first column is expected to be 'FileName' and the header of the second column is expected to be 'MetallophoneLabel'
    The first column should contain the names of .wav files containing audio recordings of key strikes for a particular instrument.
    Each file name should be unique, and contain key strikes for only one instrument.
    The second column should contain a label string for the instrument in the recording.

    Example file format::

        FileName,MetallophoneLabel
        A.wav,jegogan+
        B.wav,jegogan-
        C.wav,calung+
        ...

    Parameters
    ----------
    metallophone_labels_csv_path:
        The path to the Metallophone Labels .csv file.

    Returns
    -------
    out:
        A map of key strike recording file names to metallophone label objects.
    """
    labels_dataframe: pd.DataFrame = pd.read_csv(metallophone_labels_csv_path)

    return { row[FILE_NAME_COL_HEADER]: GamelanMetallophoneLabel.from_string(row[METALLOPHONE_LABEL_COL_HEADER]) for _, row in labels_dataframe.iterrows() }


def parse_metallophone_notes_csv(metallophone_notes_csv_path: str) -> dict[str, list[NoteSymbol]]:
    """
    Parse a map of metallophone names to note symbol lists from the Metallophone Names .csv file.
    The note symbols in each list are ordered in non-decreasing frequency of the corresponding note.

    The Metallophone Notes .csv file is expected to have a variable number of columns and one header row.
    The header of the first column is expected to be 'MetallophoneName' and the header of the last column is expected to be 'Sum'.
    The headers of the columns in between the first and last column are expected to be the symbols of notes.
    The notes columns should be ordered by non-decreasing frequency of the associated note.
    The first column will contain the names of various metallophone instruments.
    The next notes columns will contain either a '1' if the instrument in that row contains the note, or '0' otherwise.
    The last sum collumn will contain the sum of all the 1's in each row.

    Example file format::

        MetallophoneName,I_1,O_1,E_1,...,Sum
        jegogan,1,1,1,...,3
        calung,0,0,0,...,0
        ...
    
    Parameters
    ----------
    metallophone_notes_csv_path:
        The path to the Metallophone Notes .csv file.

    Returns
    -------
    out:
        A mapping from a metallophone name to a list of note symbols ordered by non-decreasing frequency of the associated note.
    """
    notes_dataframe: pd.DataFrame = pd.read_csv(metallophone_notes_csv_path)

    note_column_headers: pd.Series = notes_dataframe.columns[1:-1]

    # Find any note symbol columns where there aren't any 1's.
    empty_note_columns: pd.Series = note_column_headers[(notes_dataframe[note_column_headers].sum() == 0)]

    if empty_note_columns.shape[0] > 0:
        raise ValueError(f"{parse_metallophone_notes_csv.__name__}: Found empty note columns: {empty_note_columns}")

    name_to_notes_map: dict[str, NoteSymbol] = {}

    for _, row in notes_dataframe.iterrows():
        metallophone_name: str = row[METALLOPHONE_NAME_COL_HEADER]
        note_symbols: list[NoteSymbol] = [NoteSymbol.from_string(note_symbol_str) for note_symbol_str in note_column_headers if row[note_symbol_str] == 1]

        if len(note_symbols) != row[SUM_COL_HEADER]:
            raise ValueError(f"{parse_metallophone_notes_csv.__name__}: Mismatch between row sum and number of notes in Metallophone Notes .csv file. Row: {row}")

        name_to_notes_map[metallophone_name] = note_symbols
    
    return name_to_notes_map
