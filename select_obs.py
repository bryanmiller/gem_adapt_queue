import numpy as np
import astropy.units as u

def selectqueue(catinfo):

    """
    Select observations from catalog to add to Queue.

    Parameters
    ----------
    catinfo : '~gemini_otcat.Gcatfile'
        OT catalog info

    Returns
    -------
    numpy integer array
        indices of observations selected for queue
    """
    return np.where(np.logical_and(np.logical_or(catinfo.obs_status == 'Ready', catinfo.obs_status == 'Ongoing'),
                                   np.logical_or(catinfo.obs_class == 'Science',
                                   np.logical_and(np.logical_or(catinfo.inst == 'GMOS', catinfo.inst == 'bHROS'),
                                   np.logical_or(catinfo.obs_class == 'Nighttime Partner Calibration',
                                                 catinfo.obs_class == 'Nighttime Program Calibration')))))[0][:]

def selectinst(i_obs, obs, instcal, datestring):

    """
    Select observations to add to tonight's queue.

    Parameters
    ----------
    obs : '~gemini_otcat.Gobservations'
        All observations in queue

    inst_calender :
        Instrument and component calender

    Returns
    -------
    numpy integer array
        indices of observations selected for tonight's queue
    """

    def getcalday(instcal, datestring):
        for i in range(len(instcal['date'])):
            if instcal['date'][i] in datestring:
                return i
        print('\n Error: Date '+datestring+' not found in instrument configuration calendar.')
        raise ValueError

    def checkinst(inst, disp, fpu, instcal):
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

        instcal : '~gemini_instruments.Instruments'
            Instrument configuration calender object.

        i_cal : int
            Index for current date intrument configuration in instcal

        Returns
        -------
        boolean. True or False if the observation requirements are satisfied by the current configuration.
        """

        # print('\ninst, disp, fpu: ',inst,disp,fpu)
        if inst not in instcal['insts']:
            # print('Not available')
            return False
        elif inst in instcal['insts']:
            # print('Available')
            if 'GMOS' in inst:
                # print('GMOS')
                return ((disp in instcal['gmos_disp']) or ('null' in instcal['gmos_disp'])) and \
                       ((fpu in instcal['gmos_fpu']) or ('null' in instcal['gmos_fpu']))
            elif 'Flamingos' in inst:
                # print('Flamingos')
                return (fpu == instcal['f2_fpu']) or (instcal['f2_fpu'] == 'null')
            else:
                # print('Not GMOS or F2')
                return True

    i_cal = getcalday(instcal=instcal, datestring=datestring)
    bools = [checkinst(inst=obs.inst[j], disp=obs.disperser[j], fpu=obs.fpu[j], instcal=instcal[i_cal]) for j in
             i_obs]

    verbose = False
    if verbose:
        print(' \n Verbose: select_obs.selectinst...')
        print(' Date: ', instcal['date'][i_cal])
        print(' Available inst.: ', instcal['insts'][i_cal])
        print(' GMOS disperser: ', instcal['gmos_disp'][i_cal])
        print(' GMOS fpu: ', instcal['gmos_fpu'][i_cal])
        print(' F2 fpu: ', instcal['f2_fpu'][i_cal])
        print('')
        vprint = '\t{0:12.10} {1:12.10} {2:12.10} {3:12.10}'
        print(vprint.format('Boolean', 'Inst.', 'Disperser', 'FPU'))
        print(vprint.format('-------', '-----', '---------', '---'))
        for k in i_obs:
            print(vprint.format(str(bools[k]), obs.inst[k], obs.disperser[k], obs.fpu[k]))

    return np.where(bools)[0][:]

def selecttimewindow(time_consts, timeinfo):

    """
    Check if there is overlap between an observations time constraints
    and the current observing window in TimeInfo.  Return True or False.

    Parameters
    ----------
    time_consts : list of dictionaries
        Gemini observation time constraints (eg. time_consts = [{'start': '~astropy.time.Time',
            'duration': '~astropy.units.Quantity', 'repeat': int, 'period': '~astropy.time.Time'},...])

    timeinfo : '~gemini_classes.TimeInfo'
        observing period info for the current night

    Returns
    -------
    boolean
    """
    verbose = False
    if verbose:
        print('\n Window: ',timeinfo.start.iso,timeinfo.end.iso)
        print(' Time const: ')

    if time_consts is None:
        if verbose: print('\t'+str(time_consts))
        return True
    else:
        for time_const in time_consts: # Cycle through constraints

            # Check case of infinite window length
            if time_const['duration']==-1.0:
                if verbose: print('\t' + str(time_const['start'].iso) + ' to forever')
                if (time_const['start']-timeinfo.end)<0.*u.h:
                    return True
                else:
                    continue

            # Check case of single time window
            if time_const['repeat']==0:
                pend = time_const['start']+time_const['duration']
                if verbose: print('\t' + str(time_const['start'].iso) + ' to ' + str(pend.iso))
                if (timeinfo.start <= pend) and (time_const['start'] <= timeinfo.end):
                    return True

            # Check case of repeating time window
            if time_const['repeat'] != 0:
                nrepeat = time_const['repeat'] + 1
                i = 0
                checknextperiod = True
                while checknextperiod:
                    pstart = time_const['start']+time_const['period']*i
                    pend = pstart + time_const['duration']
                    if verbose: print('\t' + str(pstart.iso) + ' to ' + str(pend.iso))
                    if (timeinfo.start <= pend) and (pstart <= timeinfo.end):
                        return True
                    else:
                        i = i+1
                    if (i>=nrepeat) or (timeinfo.end<pstart):
                        checknextperiod = False

    # If no cases return True, return False.
    return False

def selectincomplete(tot_time, obs_time):
    """
    Select observation that are incomplete.
    Computes and rounds remaining observation times to 1/10th hour.
    Returns true for remaining times > 0

    Parameters
    ----------
    tot_time : array of '~astropy.units.Quantity'
        Total observation times

    obs_time : array of '~astropy.units.Quantity'
        Amount of tot_time completed/observed.

    Returns
    -------
    array of booleans.

    """
    return np.where(np.round((tot_time - obs_time) * 10) / 10 > 0)[0][:]