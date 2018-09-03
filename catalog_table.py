# Matt Bonnyman 2018-07-12

import re
import numpy as np
from astropy.table import Table


def catalog_table(otfile):
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
    verbose = False

    cattext = []
    with open(otfile, 'r') as readcattext:  # read file into memory.
        # [print(line.split('\t')) for line in readcattext]
        [cattext.append(line.split('\t')) for line in readcattext]  # Split lines where tabs ('\t') are found.
        readcattext.close()

    colnames = np.array(cattext[8])
    rows = np.array(cattext[10:])

    if verbose:
        print('\notcat attribute names...', colnames)

    existing_names = []
    for i in range(0, len(colnames)):

        # remove special charaters, trim ends of string, replace whitespace with underscore
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
    for i in range(len(existing_names)):
        if verbose:
            print(existing_names[i], rows[:, i])
        cattable[existing_names[i]] = rows[:, i]  # add column with to table with column name.

    if verbose:
        print('\nFound '+str(len(rows))+' observations in '+str(otfile))

    return cattable
