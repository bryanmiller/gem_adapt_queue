import numpy as np
import astropy.units as u

import weights as weights


def _getcalday(instcal, date):
    """
    Get index of row for current night in instrument calendar table.
    """
    for i in range(len(instcal['date'])):
        if instcal['date'][i] in date:
            return i
    print('\n Date '+date+' not found in instrument configuration calendar.')
    raise ValueError('Date '+date+' not found in instrument configuration calendar.')


def _checkinst(inst, disp, fpu, insttable):
    """
    Check observation instrument requirements against tonight's instrument configuration.


    Parameters
    ----------
    inst : str
        Observation instrument

    disp : str
        Observation instrument disperser (or 'null')

    fpu : str
        Observation instrument focal plane unit (or 'null')

    insttable : '~astropy.table.Table'
        Instrument configuration calender.

    i_cal : int
        Index of row for tonights instrument configuration.

    Returns
    -------
    boolean. True or False if the observation requirements are satisfied by the current configuration.
    """

    # print('\ninst, disp, fpu: ',inst,disp,fpu)
    if inst not in insttable['insts']:
        # print('Not available')
        return False
    elif inst in insttable['insts']:
        # print('Available')
        if 'GMOS' in inst:
            # print('GMOS')
            return ((disp in insttable['gmos_disp']) or ('null' in insttable['gmos_disp'])) and \
                   ((fpu in insttable['gmos_fpu']) or ('null' in insttable['gmos_fpu']))
        elif 'Flamingos' in inst:
            # print('Flamingos')
            return (fpu == insttable['f2_fpu']) or (insttable['f2_fpu'] == 'null')
        else:
            # print('Not GMOS or F2')
            return True


def i_progs(gemprgid, prog_ref):
    """
    Match program identifier strings in prog_ref to strings in gemprgid.
    Return array of indices.

    Parameters
    ----------
    gemprgid : list of strings
        Gemini program id column from program table

    prof_ref :
        Gemini program id column from observation table

    Returns
    -------
    i_progs : int array
        Array of indices.

    """

    verbose = False

    i_progs = np.full(len(prog_ref), -1)
    prog_refs = np.unique(prog_ref)

    if verbose:
        print(prog_refs)

    for prog in prog_refs:
        j = np.where(gemprgid == prog)[0][0]  # should only have one value in array
        jj = np.where(prog_ref == prog)[0][:]  # observations in queue from program 'prog'
        i_progs[jj] = j
        if verbose:
            print(prog, j, jj)

    return i_progs


def instconfig(inst, disp, fpu, insttable, date):

    """
    Select observations to add to tonight's queue.

    Parameters
    ----------
    obs : '~astropy.table.Table'
        All observations in queue

    insttable : '~astropy.table.Table'
        Instrument configuration calender.

    date : str
        tonights date

    Returns
    -------
    numpy integer array
        indices of observations selected for tonight's queue
    """
    verbose = False

    i_cal = _getcalday(instcal=insttable, date=date)
    bools = [_checkinst(inst=inst[j], disp=disp[j], fpu=fpu[j], insttable=insttable[i_cal]) for j in range(len(inst))]

    if verbose:
        print(' \n Verbose: select_obs.selectinst...')
        print(' Date: ', insttable['date'][i_cal])
        print(' Available inst.: ', insttable['insts'][i_cal])
        print(' GMOS disperser: ', insttable['gmos_disp'][i_cal])
        print(' GMOS fpu: ', insttable['gmos_fpu'][i_cal])
        print(' F2 fpu: ', insttable['f2_fpu'][i_cal])
        print('')
        vprint = '\t{0:12.10} {1:12.10} {2:12.10} {3:12.10}'
        print(vprint.format('Boolean', 'Inst.', 'Disperser', 'FPU'))
        print(vprint.format('-------', '-----', '---------', '---'))
        for k in range(len(inst)):
            print(vprint.format(str(bools[k]), inst[k], disp[k], fpu[k]))

    return np.where(bools)[0][:]


def selectqueue(cattable):

    """
    Select observations from catalog to add to Queue.

    Parameters
    ----------
    cattable : '~gemini_otcat.Gcatfile'
        OT catalog info

    Returns
    -------
    numpy integer array
        indices of observations selected for queue
    """
    return np.where(np.logical_and(np.logical_or(cattable['obs_status'] == 'Ready', cattable['obs_status'] == 'Ongoing'),
                                   np.logical_or(cattable['obs_class'] == 'Science',
                                   np.logical_and(np.logical_or(cattable['inst'] == 'GMOS', cattable['inst'] == 'bHROS'),
                                   np.logical_or(cattable['obs_class'] == 'Nighttime Partner Calibration',
                                                 cattable['obs_class'] == 'Nighttime Program Calibration')))))[0][:]
