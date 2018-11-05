# Matt Bonnyman 2018-07-12

import re
import numpy as np
from astropy.table import Table


def catalog_table(otfile, verbose = False):
    """
    Read OT browser catalog file and return columns in '~astropy.table' object.
    Numbers are appended to the end of repeated column names.
    Format requires that column names are along line 8, and observation data begins at line 10.

    Parameters
    ----------
    otfile : str
        OT catalog file name

    Returns
    -------
    '~astropy.table'
    """

#    import sys  
#
#    reload(sys)  
#    sys.setdefaultencoding('utf8')

#    encoding=utf8

    # cattext = []
    rows = np.array([])
    with open(otfile, 'r', encoding="utf-8") as readcattext:  # read file into memory.
        # [print(line.split('\t')) for line in readcattext]
        # [cattext.append(line.split('\t')) for line in readcattext]  # Split fields where tabs ('\t') are found.
        nline = 0
        for line in readcattext:
            nline += 1
            values = line.rstrip("\n").split('\t')
            # Don't includes invalid lines, e.g. obs with no targets have ra=null, no Acq
            if nline > 10 and values[5] != 'null' and 'Acquisition' not in values[15]:
                # In case no charged time
                if values[14] == '':
                    values[14] = '00:00:00'
                if rows.size > 0:
                    rows = np.vstack([rows,values])
                else:
                    rows = np.array(values)
            elif nline == 9:
                colnames = np.array(values)
        readcattext.close()

    # colnames = np.array(cattext[8])
    # rows = np.array(cattext[10:])
    # print(colnames)
    # print(rows.shape)

    if verbose:
        print('\notcat attribute names...', colnames)

    existing_names = []
    for i in range(0, len(colnames)):

        # remove special characters, trim ends of string, replace whitespace with underscore
        string = colnames[i].replace('.', '')
        string = re.sub(r'\W', ' ', string)
        string = string.strip()
        string = re.sub(r' +', '_', string)
        string = string.lower()

        if string == 'class':  # change attribute name (python doesn't allow attribute name 'class')
            string = 'obs_class'

        if np.isin(string, existing_names):  # add number to end of repeated attribute name

            rename = True
            j = 0

            while rename:

                if j >= 8:  # if number 9 is reached, make next number 10
                    tempstring = string + '_1' + chr(50 + j - 10)
                else:
                    tempstring = string + '_' + chr(50 + j)

                if np.isin(tempstring, existing_names):  # if name taken, increment number and check again
                    j += 1
                else:
                    string = tempstring
                    rename = False

        existing_names.append(string)  # add name to library of used names

        if verbose:
            print(string)

    cattable = Table()

    # print(len(existing_names))
    for i in range(len(existing_names)):
        if verbose:
            print(existing_names[i], rows[:, i])
        cattable[existing_names[i]] = rows[:, i]  # add column with to table with column name.

    if verbose:
        print('\nFound '+str(len(rows))+' observations in '+str(otfile))
        print(cattable)

    return cattable
