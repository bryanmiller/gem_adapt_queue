import re
import numpy as np
from astropy.time import Time
from astropy.table import Table, Column


def programtable(gemprgid, partner, pi, prog_time, alloc_time, partner_time, active, prog_start, prog_end,
                 too_status, scirank, observations):
    """
    Create an '~astropy.table.Table' program data structure.

    Parameters
    ----------
    gemprgid : string
        list of unique program identifiers

    partner : string
        list of Gemini partner names

    pi : string
        list of personal investigators

    prog_time : string
        list of program charged times

    alloc_time : string
        list of program allocated times

    partner_time : string
        list of partner charged times

    active : boolean
        list of program statuses

    prog_start : string
        list of program start dates

    prog_end : string
        list of program end dates

    complete : string
        list of program completion statuses

    too_status : string
        list of ToO statuses ('Rapid', 'Standard', 'None')

    scirank : string
        list of science ranking bands

    observations : string
        list of observations in program


    Returns
    -------
    progtable : '~astropy.table.Table'

        note: if program is not found in observation table, values are set to null

        Columns
        -------
        gemprgid : string
            gemini program identifier

        partner : string
            gemini partner name

        pi : string
            principle investigator

        prog_time : '~astropy.unit.quantity.Quantity'
            program hours completed hours

        alloc_time : '~astropy.unit.quantity.Quantity'
            program hours allocated

        prog_comp : floats
            fractions of completed/allocated hours

        partner_time :'~astropy.unit.quantity.Quantity'
            hours of partner time charged

        active : boolean
            program active status

        prog_start : '~astropy.time.core.Time'
            program activation time

        prog_end : '~astropy.time.core.Time'
            program deactivation time

        complete : boolean
            program completion status

        too_status : string
            target of Opportunity type: 'Rapid', 'Standard', or 'None'

        scirank : str
            science ranking band (1, 2, 3, 4)

        observations : list of strings
            identifiers of observations in program
    """

    progtable = Table()

    progtable['gemprgid'] = Column(gemprgid)
    progtable['partner'] = Column(partner)
    progtable['pi'] = Column(pi)

    prog_time_hr = np.array(prog_time, dtype=float)  # convert from string to float
    alloc_time_hr = np.array(alloc_time, dtype=float)  # convert from string to float
    # Some programs, e.g. ENG, will have 0 allocated time, give them 10
    ii = np.where(alloc_time_hr == 0.0)[0]
    if (len(ii) > 0):
        alloc_time_hr[ii] = 10.
    prog_comp = prog_time_hr/alloc_time_hr  # fraction completed

    progtable['prog_time'] = Column(prog_time_hr, unit='hr')
    progtable['alloc_time'] = Column(alloc_time_hr, unit='hr')
    progtable['prog_comp'] = Column(prog_comp)
    progtable['partner_time'] = Column(np.array(partner_time, dtype=float), unit='hr')
    progtable['active'] = Column(active)
    progtable['prog_start'] = Column(Time(prog_start))
    progtable['prog_end'] = Column(Time(prog_end))
    progtable['complete'] = Column(prog_comp >= 1.)
    progtable['too_status'] = Column(too_status)
    progtable['scirank'] = Column(scirank)
    progtable['observations'] = Column(observations)

    return progtable


def read_exechours(filename, verbose = False):
    """
    Read exechours_SEMESTER.txt file and return columns as '~astropy.table.Table'.

    Parameters
    ----------
    filename : string
        program exec hours text file name.

    Returns
    -------
    progtable : '~astropy.table.Table'
        Program data table

        Columns
        -------
        'prog_ref' : str
            program references

        'alloc_time' : str
            number of hours allocated to program

        'elaps_time' : str
            number of hours of elapsed time

        'notcharged_time' : str
            number of hours not charged

        'partner_time' : str
            number of hours charged to partner

        'prog_time' : str
            number of hours charged to program

    """

    filetext = []
    with open(filename, 'r') as readtext:  # read file into memory.
        # Split lines where commas ',' are found.  Remove newline characters '\n'.
        [filetext.append(re.sub('\n', '', line).split(',')) for line in readtext]
        readtext.close()

    # # For testing, set times elapsed, non-charged, partner, and program to zero.
    # for i in range(len(filetext)):
    #     for j in range(2, len(filetext[i])):
    #         filetext[i][j] = '0.00'

    if verbose:
        [print(line) for line in filetext]

    rows = np.array(filetext[3:])
    columns = ['prog_ref', 'alloc_time', 'elaps_time', 'notcharged_time', 'partner_time', 'prog_time']

    exechourstable = Table()
    for i in range(len(columns)):
        exechourstable[columns[i]] = rows[:, i]

    if verbose:
        print(exechourstable)

    return exechourstable


def get_proginfo(exechourtable, prog_ref_obs, obs_id, partner, pi, band, too_status, verbose = False):
    """
    Add columns for pi, partner, too_status, obs, and scirank to exechours table.
    This information is not found in the execHours file, and is therefore retrieved
    from the observation table.

    Parameters
    ----------
    exechourtable : '~astropy.table.Table'
        exechours text file table created by progtable.read_exechours().

        Columns
        -------
        'prog_ref' : str
            program references

        'alloc_time' : str
            number of hours allocated to program

        'elaps_time' : str
            number of hours of elapsed time

        'notcharged_time' : str
            number of hours not charged

        'partner_time' : str
            number of hours charged to partner

        'prog_time' : str
            number of hours charged to program

    prog_ref : np.array of str
        program reference identifiers of observations in queue

    obs_id : np.array of str
        observation identifiers

    partner : np.array of str
        gemini partner name

    pi : np.array of str
        principle investigators of observations in queue

    band : np.array of str
        science ranking bands of observations (1, 2, 3, 4)

    too_status : np.array of str
        too types of observation ('Rapid', 'Standard', or 'None')

    Returns
    -------
    progtable : '~astropy.table.Table'
        Table of programs in queue.

        Columns
        -------
        'prog_ref' : str
            program references

        'alloc_time' : str
            number of hours allocated to program

        'elaps_time' : str
            number of hours of elapsed time

        'notcharged_time' : str
            number of hours not charged

        'partner_time' : str
            number of hours charged to partner

        'prog_time' : str
            number of hours charged to program

        'obs' : list of str
            observations in program

        'pi' : str
            program principle investigator

        'scirank' : str
            program science ranking band (1, 2, 3, 4)

    """

    # Create empty table and new columns as lists
    prog_obs = []
    prog_pi = []
    prog_partner = []
    prog_band = []
    prog_too = []

    progs = np.unique(prog_ref_obs)


    if verbose:
        print('queue programs', progs)
        print('queue obs', obs_id)

    for i in range(len(exechourtable)):

        # if program in queue, get index of current program in program exec hours table
        if exechourtable[i]['prog_ref'] in progs:
            j = np.where(progs == exechourtable[i]['prog_ref'])[0][:][0]  # should only have 1 match
            ii = np.where(exechourtable[i]['prog_ref'] == progs[j])[0][:]
            jj = np.where(prog_ref_obs == progs[j])[0][:]  # get indices of all program observations in queue
            prog_obs.append(list(obs_id[jj]))  # save obs IDs as list
            prog_partner.append(partner[jj][0])  # get partner name from first obs in program
            prog_too.append(too_status[jj[0]])  # get too_status from first obs in program
            prog_pi.append(pi[jj[0]])  # get PI from first obs in program (should be same for all)
            prog_band.append(band[jj[0]])  # get band from first obs in program (should be same for all)
        else:  # set all to null if program info not found
            prog_obs.append([])
            prog_too.append('null')
            prog_partner.append('null')
            prog_pi.append('null')
            prog_band.append('null')

    if verbose:
        print('prog_pi', prog_pi)
        print('prog_band', prog_band)
        print('prog_obs', prog_obs)

    exechourtable['obs'] = Column(prog_obs)
    exechourtable['pi'] = Column(prog_pi)
    exechourtable['partner'] = Column(prog_partner)
    exechourtable['scirank'] = Column(prog_band)
    exechourtable['too_status'] = Column(prog_too)

    return exechourtable
