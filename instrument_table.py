import numpy as np
from astropy.table import Table


def instrument_table(filename, dates):
    """
    Read instrument calender into memory.
    Generate '~astropy.table.Table' object of instrument calendar throughout scheduling period.
    If the calendar is missing a night in the scheduling period, assign 'null' to all fields.

    Columns in InstTable:

        Key         Dtype   Description
        ---         -----   -----------
        'date'      str     calendar date
        'insts'     str     names of instruments installed on telescope
        'gmos_fpu'  str     GMOS focal plane unit
        'gmos_disp' str     GMOS disperser
        'f2_fpu'    str     Flamingos-2 focal plane unit

    Parameters
    ----------
    filename : str
        Instrument configuration file name

    dates : list or array of str
        Dates of scheduling period in YYYY-MM-DD format (eg. ['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04'])

    Returns
    -------
    '~astropy.table.Table'
    """

    verbose = False

    # Read file
    inputlines = []
    with open(filename, 'r') as f:
        [inputlines.append(line.strip('\n').split('\t')) for line in f]
        f.close()

    # instrument information table from file
    instlines = np.array(inputlines[2:])

    if verbose:
        print(dates)
        [print(instline) for instline in instlines]

    # Get calendar rows of dates in scheduling period.
    # If calendar is missing date, assigned 'null' to all fields.
    inst_rows = []
    for i in range(len(dates)):

        if verbose:
            print('date', dates[i])

        i_row = np.where(instlines[:,0] == dates[i])[0][:]

        if verbose:
            print('i_row', i_row)

        if len(i_row) == 1:
            inst_rows.append(instlines[i_row[0]])
        else:
            inst_rows.append([dates[i], 'null', 'null', 'null', 'null'])

    if verbose:
        [print(row) for row in inst_rows]

    return Table(np.array(inst_rows), names=['date', 'insts', 'gmos_fpu', 'gmos_disp', 'f2_fpu'])
