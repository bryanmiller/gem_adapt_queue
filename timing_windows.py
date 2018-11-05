# Matt Bonnyman 27 July 2018
# This module contains several functions for converting and constrain Gemini observing time constraints.
# get_timing_windows is the main method.


# import time as t
import astropy.units as u
from astropy.time import Time
from multiprocessing import cpu_count
from joblib import Parallel, delayed
import numpy as np
import re

from dt import deltat
from target_table import target_table


def time_window_indices(utc, time_wins, dt):
    """
    Convert the times in time_wins to indices in utc.

    Parameters
    ----------
    utc : 'astropy.time.core.Time' np.array
        UTC time grid for scheduling period (i.e. night)

    dt : 'astropy.units'
        size of time grid spacing

    time_wins : 'astropy.time.core.Time' pair(s)
        observation time windows during scheduling period.

        Example
        -------
        An observation with 4 time windows within the current night...
        time_wins = [
                     [<Time object: scale='utc' format='unix' value=1522655300.0>,
                      <Time object: scale='utc' format='unix' value=1522655388.0>],
                     [<Time object: scale='utc' format='unix' value=1522657440.0>,
                      <Time object: scale='utc' format='unix' value=1522657548.0>],
                     [<Time object: scale='utc' format='unix' value=1522659600.0>,
                      <Time object: scale='utc' format='unix' value=1522659708.0>],
                     [<Time object: scale='utc' format='unix' value=1522661760.0>,
                      <Time object: scale='utc' format='unix' value=1522661868.0>]
                    ]
    """
    verbose = False

    if verbose:
        print('dt', dt)
        print('utc range', utc[0].iso, (utc[-1] + dt).iso)
        print(time_wins)

    nt = len(utc)

    i_time_wins = []

    if len(time_wins) == 0:
        return i_time_wins
    else:
        for win in time_wins:

            if verbose:
                print('obs window', win[0].iso, win[1].iso)

            # Get index of start of window
            win[0].format = 'jd'
            if win[0] <= utc[0]:
                i_start = 0
                i = 0
            else:
                for i in range(nt):

                    # print(utc[i].iso, win[0].iso, (utc[i] + dt).iso)
                    # print(type(utc[i]), type(win[0]), type((utc[i] + dt)))
                    # print(utc[i].scale, win[0].scale, (utc[i] + dt).scale)
                    # print(utc[i].format, win[0].format, (utc[i] + dt).format)
                    # print(utc[i].value, win[0].value, (utc[i] + dt).value)
                    # print(utc[i].value <= win[0].value, win[0] < utc[i] + dt)

                    # Note: there is a astropy.time.Time comparison error that is
                    # due to rounding error (issue: 'Float comparison issues with time
                    # and quantity #6970').  It appears that as Time objects are manipulated
                    # they are converted to TAI then back to UTC.
                    # As a result, equal times were occasionally considered
                    # neither equal or unequal, raising an error during this algorithm.
                    # As a work around Time are now compared using their JD float values.
                    if utc[i].value <= win[0].value < (utc[i] + dt).value:
                        i_start = i
                        break

            # estimate the index of the end of the window.
            # round down to closest integer
            ntwin = int((win[1] - win[0]).to('hour')/dt)
            i = i + ntwin

            # Get index of end of window
            win[1].format = 'jd'
            if i >= nt:
                i_end = nt - 1
            else:
                for j in range(i, nt):
                    if utc[j].value <= win[1].value < (utc[j] + dt).value:
                        i_end = j
                        break

            if verbose:
                print('index window boundaries', i_start, i_end)
                print('corresponding time grid times', utc[i_start].iso, utc[i_end].iso)

            i_time_wins.append([i_start, i_end])

        if verbose:
            print('i_time_wins:')
            [print(i_win) for i_win in i_time_wins]

        return i_time_wins


def i_time(times, timegrid):
    """
    Return, for a list of times, the indices at which they appear in a time grid.
    Note: time strings must be in formats accepted by '~astropy.time.Time'. Preferably ISO format.

    Parameters
    ----------
    times : list or array of str
        times to check in formats accepted by '~astropy.time.Time'

    timegrid : list or array of str
        time grid lf observing window in formats accepted by '~astropy.time.Time'

    Returns
    -------
    bool
    """
    if len(times) == 0:
        return []
    else:
        timegrid = Time(timegrid)
        times = Time(times)
        i_times = np.zeros(len(times), dtype=int)
        for i in range(0, len(times)):
            for j in range(0, len(timegrid) - 1):
                if timegrid[j] < times[i] < timegrid[j + 1]:
                    i_times[i] = j
                    break
        return i_times


def checkwindow(times, timegrid):
    """
    Check which times are within the boundaries of a time grid.  Return an array of booleans.
    Note: time strings must be in formats accepted by '~astropy.time.Time'. Preferably ISO format.

    Parameters
    ----------
    times : list or array of str
        times to check

    timegrid : list or array of str
        times in grid (in formats accepted by '~astropy.time.Time')

    Returns
    -------
    np.array of booleans
    """
    start = Time(timegrid[0])
    end = Time(timegrid[-1])
    bools = np.full(len(times), False)
    for i in range(0, len(times)):
        time = Time(times[i])
        if start < time < end:
            bools[i] = True
    return bools


def convconstraint(time_const, start, end, current_time=None):
    """
    Convert and compute time windows within scheduling period from time constraints 'time_const' for and
    observation in the ObsTable structure.

    Parameters
    ----------
    time_const : str
        Time constraint for Gemini Observation formatted as in the catalog browser ascii dump.

        Format
        ------
        time_const = '[{start, duration, repeats, period}, {start, duration, repeats, period}, ...]'

               start    :   unix time in milliseconds (-1 = current)
            duration    :   window length in milliseconds (-1 = infinite)
             repeats    :   number of repeats (-1 = infinite)
              period    :   milliseconds between window start times

    start : '~astropy.time.core.Time'
        Scheduling period start time.

    end : '~astropy.time.core.Time'
        Scheduling period end time.

    current_time : '~astropy.time.core.Time' or None
        Current time in simulation (for triggering ToO time constraints).

    Returns
    -------
    time_win : list of '~astropy,time.core.Time' arrays, or None
        Array of time pairs of time windows overlapping with scheduling period.  Returns None is no time windows
        overlap with the scheduling window.

    Example
    -------
    >>> from timing_windows import convconstraint
    >>> start = '2018-01-01 00:00:00'
    >>> end = '2019-01-01 00:00:00'
    >>> time_const = '[{start, duration, repeats, period}, {start, duration, repeats, period}, ...]'
    >>> time_wins = convconstraint(time_const, start, end)
    """
    verbose = False

    if verbose:
        print('\ntime_const', time_const)
        print('start', start)
        print('end', end)

    infinity = 3. * 365. * 24. * u.h  # infinite time duration

    # Split individual time constraint strings into list
    string = re.sub('[\[{}\]]', '', time_const).split(',')  # remove brackets
    string = [tc.strip() for tc in string]  # remove whitespace

    if verbose:
        print('Constraint strings: ', string)

    if string[0] == '':  # if no time constraints
        return [[start, end]]

    else:  # if observation has time constraints
        obs_win = []  # observation time windows
        tc = [re.findall(r'[+-]?\d+(?:\.\d+)?', val) for val in string]  # split numbers into lists
        tc.sort()  # sort constraints into chronological order

        if verbose:
            print('Ordered constraints: ', tc)

        for const in tc:  # cycle through constraints

            # time window start time t0 (unix time format milliseconds).
            # for ToOs, the timing constraint must begin at the time of arrival to the queue.
            # To do this, set the ToO program start to the time of arrival, and give the
            # ToO observation time constraint a t0 value of -1.  In this case, the time constraint
            # will begin from the new program start time.
            t0 = float(const[0])
            if t0 == -1:  # -1 = current time in simulation
                t0 = current_time
            else:
                t0 = Time((float(const[0]) * u.ms).to_value('s'), format='unix', scale='utc')

            duration = float(const[1])   # duration (milliseconds)
            if duration == -1.0:  # infinite
                pass
                duration = infinity
            else:
                duration = duration / 3600000. * u.h

            repeats = int(const[2])  # number of repetitions
            if repeats == -1:  # infinite
                repeats = 1000

            period = float(const[3]) / 3600000. * u.h  # period between windows (milliseconds)

            if verbose:
                print('t0.iso, duration, repeats, period: ', t0.iso, duration, repeats, period)

            n_win = repeats + 1  # number of time windows in constraint including repeats
            win_start = t0
            for j in range(n_win):  # cycle through time window repeats
                win_start = win_start + period  # start of current time window
                win_end = win_start + duration  # start of current time window
                if verbose:
                    print('j, window: ', j, [win_start.iso, win_end.iso])
                # save time window if there is overlap with schedule period
                if win_start < end and start < win_end:
                    obs_win.append([win_start, win_end])
                    if verbose:
                        print('\nadded window')
                elif win_end < start:  # go to next next window if current window precedes schedule period
                    pass
                else:  # stop if current window is past schedule period
                    break

        if not obs_win:
            return None
        else:
            return obs_win


def twilights(twilight_evening, twilight_morning, obs_windows):
    """
    Confine observation timing constraints within nautical twilights.

    Parameters
    ----------
    twilight_evening : '~astropy.time.core.Time' array
        Evening twilight time for scheduling period (UTC)

    twilight_morning : '~astropy.time.core.Time' array
        Morning twilight time for scheduling period (UTC)

    obs_windows : list of '~astropy.time.core.Time' pairs, or None
        Observation timing window time-pairs in UTC.
        Each observation can have any number of time windows.

    Returns
    -------
    new_windows : list of lists of '~astropy.time.core.Time' pairs or None
        New list of time windows constrained within twilights.
    """
    verbose = False

    new_windows = []

    if obs_windows is not None and len(obs_windows) != 0:

        for i in range(len(twilight_evening)):  # cycle through nights

            if verbose:
                print('\ntwilights: ', twilight_evening[i].iso, twilight_morning[i].iso)

            for j in range(len(obs_windows)):  # cycle through time windows

                if verbose:
                    print('time_const[' + str(j) + ']:', obs_windows[j][0].iso, obs_windows[j][1].iso)

                # save time window if there is overlap with schedule period
                if obs_windows[j][0] < twilight_morning[i] and twilight_evening[i] < obs_windows[j][1]:
                    # Add window with either twilight times or window edges as boundaries (whichever are innermost).
                    new_windows.append([max([twilight_evening[i], obs_windows[j][0]]),
                                        min([twilight_morning[i], obs_windows[j][1]])])

                    if verbose:
                        print('\tadded:', max([twilight_evening[i], obs_windows[j][0]]).iso,
                              min([twilight_morning[i], obs_windows[j][1]]).iso)

        if verbose:
            print('new_windows:')
            [print('\t', new_window[0].iso, new_window[1].iso) for new_window in new_windows]

        if len(new_windows) == 0:
            return None
        else:
            return new_windows
    else:
        return None


def instrument(i_obs, obs_inst, obs_disp, obs_fpu, obs_mos, insts, gmos_disp, gmos_fpu, gmos_mos, f2_fpu, f2_mos,
               verbose = False):
    """
    Constrain observation timing constraints in accordance with the installed instruments
    and component configuration on the current time.
    Output indices of observations matching the nightly instruments and components.

    Parameters
    ----------
    i_obs : integer array
        indices in obs_inst, obs_disp, and obs_fpu to check for current night.

    obs_inst : list of strings
        Observation instruments

    obs_disp : list of strings
        Observation dispersers

    obs_fpu : list of string
        Observation focal plane units

    obs_mos : list of string
        Observation custom mask name

    insts : list of strings
        Instruments installed on current night

    gmos_disp : list of strings
        GMOS disperser installed on current night

    gmos_fpu : list of strings
        GMOS focal plane units (not MOS) installed on current night

    gmos_mos : list of strings
        GMOS MOS masks installed on current night

    f2_fpu : list of strings
        Flamingos-2 focal plane units (not MOS) installed on current night

    f2_mos : list of strings
        Flamingos-2 MOS masks installed on current night

    Returns
    -------
    in_obs : integer array
        List indices for observations matching tonight's instrument configuration.
    """

    if verbose:
        print('Installed instruments')
        print('insts', insts)
        print('gmos_disp', gmos_disp)
        print('gmos_fpu', gmos_fpu)
        print('gmos_mos', gmos_mos)
        print('f2_fpu', f2_fpu)
        print('f2_mos', f2_mos)
        print('i_obs', i_obs)

    if len(i_obs) == 0:
        return []

    else:
        in_obs = []

        # Select i_obs from observation lists
        obs_inst = obs_inst[i_obs]
        obs_disp = obs_disp[i_obs]
        obs_fpu = obs_fpu[i_obs]

        for i in range(len(obs_inst)):

            if verbose:
                print('obs_inst[i], obs_disp[i], obs_fpu[i], obs_mos[i]', obs_inst[i], obs_disp[i], obs_fpu[i], obs_mos[i])

            if obs_inst[i] in insts or insts == 'all':

                if 'GMOS' in obs_inst[i]:
                    if ((obs_disp[i] in gmos_disp) or ('all' in gmos_disp))\
                        and (((obs_fpu[i] in gmos_fpu) or ('all' in gmos_fpu))\
                        or ((obs_mos[i] in gmos_mos) or (('all' in gmos_mos) and ('Custom Mask' == obs_fpu[i])))):
                        in_obs.append(i)
                        if verbose:
                            print('Added i =', i)
                elif 'Flamingos' in obs_inst[i]:
                    if ((obs_fpu[i] in f2_fpu) or ('all' in f2_fpu))\
                        or ((obs_mos[i] in f2_mos) or (('all' in f2_mos) and ('Custom Mask' == obs_fpu[i]))):
                        in_obs.append(i)
                        if verbose:
                            print('Added i =', i)
                else:
                    in_obs.append(i)
                    if verbose:
                        print('Added i =', i)

        return in_obs


def nightly_calendar(twilight_evening, twilight_morning, time_windows):
    """
    Sort observation time windows by nightly observing window.

    Parameters
    ----------
    twilight_evening : '~astropy.time.core.Time'
        Evening twilight time for scheduling period (UTC)

    twilight_morning : '~astropy.time.core.Time'
        Morning twilight time for scheduling period (UTC)

    time_windows : list of lists of '~astropy.time.core.Time' pairs
        Array of time windows for all observations.

    Returns
    -------
    i_obs : int array
        Indices of observations with a time_window during the night of the provided date.

    obs_windows : array of '~astropy.time.core.Time' pair(s)
        Observation time windows for current night corresponding to 'i_obs'.
    """
    verbose = False

    # define start of current day as local noon
    night_start = twilight_evening
    night_end = twilight_morning

    if verbose:
        print('\nDate window (start,end): ', night_start.iso, night_end.iso)

    i_obs = []  # list of current night's observations
    obs_windows = []  # time windows corresponding to i_obs
    for i in range(len(time_windows)):  # cycle through observations

        if verbose:
            print('\tobs i:', i)

        if time_windows[i] is not None:

            obs_wins = []

            for j in range(len(time_windows[i])):  # cycle through time windows

                if verbose:
                    print('\t\ttime_window[' + str(i) + '][' + str(j) + ']:',
                          time_windows[i][j][0].iso, time_windows[i][j][1].iso)

                # save index if there is overlap with schedule period
                if time_windows[i][j][1] >= night_start and night_end >= time_windows[i][j][0]:
                    obs_wins.append(time_windows[i][j])
                    if verbose:
                        print('\t\t\tadded window')
                # else:
                #     print('\t\tnot added')

            # if time window(s) overlapped with night, save obs index and window(s)
            if len(obs_wins) != 0:
                i_obs.append(i)
                obs_windows.append(obs_wins)
                if verbose:
                    print('\t\tadded obs index'
                          ' to list')

        else:
            if verbose:
                print('\t\t\ttime_window[' + str(i) + ']:', time_windows[i])
            pass

    # if verbose:
    #     print('i_obs', i_obs)
    #     print('obs_windows', obs_windows)

    return i_obs, obs_windows


def elevation_const(targets, i_wins, elev_const):
    """
    Restrict time windows for elevation constraints.
    If all time windows for a given observation are removed, let  time_window[i] = NoneType.

    Parameters
    ----------
    targets : '~astropy.table.Table'
        Target table for current night (Columns: 'i', 'id', 'ZD', 'HA', 'AZ', 'AM', 'mdist').

    i_wins : list of lists
        Observation time windows as time grid indices.  Each observation may have one or more time windows.

        Example
        -------
        i_wins = [
                  [[0,2], [4, 10]],
                  [[0,20]],
                  [[0,10], [15,20],
                  ...]

    elev_const : list of dictionaries
        Elevation constraints of observations in observation table
        (dictionary keys: {'type':str, 'min': float or '~astropy.units', 'max': float or '~astropy.units'}).

    Returns
    -------
    targets : '~astropy.table.Table' target table
        Target table for current night with time windows constrained to meet elevation constraints.
        If an observation has no remaining time windows, the table cell is given NoneType.
    """

    verbose = False

    if len(targets) != 0:

        for i in range(len(targets)):  # cycle through rows in table

            # set new window boundaries to -1 to start
            i_start = -1
            i_end = -1

            if verbose:
                print()
                print(targets['i'].data[i], elev_const[i])

            j = targets['i'].data[i]  # elevation constraint index of target

            # Get time grid window indices for elevation constraint
            if elev_const[j]['type'] == 'Hour Angle':
                if verbose:
                    print('\nHour Angle!')
                    print(targets['HA'].quantity[i])
                    print(elev_const[i]['min'])
                    print(elev_const[i]['max'])
                    # print(targets['HA'].quantity[i] >= elev_const[i]['min'])
                    # print(targets['HA'].quantity[i] <= elev_const[i]['max'])

                # get indices of hour angles within constraint limits
                ii = np.where(np.logical_and(
                    targets['HA'].quantity[i] >= elev_const[j]['min'],
                    targets['HA'].quantity[i] <= elev_const[j]['max'])
                )[0][:]

                if verbose:
                    print('ii', ii)

                # save boundaries of indices within constraint
                if len(ii) != 0:
                    i_start = ii[0]
                    i_end = ii[-1]

            elif elev_const[j]['type'] == 'Airmass':
                if verbose:
                    print('\nAirmass!')
                    print(targets['AM'][i])

                # get indices of airmass within constraint limits
                ii = np.where(np.logical_and(targets['AM'][i] >= elev_const[j]['min'],
                                             targets['AM'][i] <= elev_const[j]['max'])
                              )[0][:]
                if verbose:
                    print('ii', ii)

                # save boundaries of indices within constraint
                if len(ii) != 0:
                    i_start = ii[0]
                    i_end = ii[-1]

            else:  # skip to next observation if current one has no elevation constraints
                if verbose:
                    print('No elevation constraint!')
                continue

            # Set new time windows boundaries if observation had elevation constraint
            if i_start != -1 and i_end != -1:

                if verbose:
                    print('i_start, i_end: ', i_start, i_end)

                # Cycle through observation time windows for current night.
                # Adjust each window to satisfy elevation constraint.
                j = 0

                while True:
                    if verbose:
                        print('initial time window:',
                              i_wins[i][j][0],
                              i_wins[i][j][1])

                    # If current time window overlaps with elevation constraint window, set new window.
                    if i_wins[i][j][0] <= i_end and i_start <= i_wins[i][j][1]:

                        # Change window to portion of overlap.
                        i_wins[i][j] = ([max([i_start, i_wins[i][j][0]]),
                                         min([i_end, i_wins[i][j][1]])])
                        j = j + 1
                        if verbose:
                            print('\toverlap of windows:',
                                  i_wins[i][j-1][0],
                                  i_wins[i][j-1][1])

                    else:  # Delete window if there is no overlap
                        if verbose:
                            print('i_wins[i],j', i_wins[i], j, type(i_wins[i]), type(i_wins[i][j]))
                        del i_wins[i][j]
                        if verbose:
                            print('\tdelete window')

                    if j == len(i_wins[i]):
                        break

                if len(i_wins[i]) == 0:
                    i_wins[i] = None

            if verbose:
                print('new observation time windows for tonight:')
                if i_wins[i] is not None:
                    for j in range(len(i_wins[i])):
                        print([i_wins[i][j][0], i_wins[i][j][1]])
                else:
                    print(i_wins[i])

    return i_wins


def get_timing_windows(site, timetable, moon, obs, progs, instcal, current_time=None, verbose=False, debug=False):
    """
    Main timing windows algorithm.  This is the main method that generates timing windows and the
    target data tables.

    It performs the following sequence of steps using functions from timing_windows.py and target_table.py.

    1. Convert timing window constraints
    2. Constrain within plan boundaries and program activation dates
    3. Constrain within twilights
    4. Organize time windows by date
    5. Constrain within instrument calendar
    6. Generate a list of nightly target data tables (from target_table.py)
    7. Constrain windows within elevation constraints

    Return list of target data tables.

    Parameters
    ----------
    site : 'astroplan.Observer'
        Observatory site object

    timetable : 'astropy.table.Table'
        Time data table generated by time_table.py

    moon : 'astropy.table.Table'
        Moon data table generated by moon_table.py

    obs : 'astropy.table.Table'
        Observation data table generated by observation_table.py

    progs : 'astropy.table.Table'
        Program status data table generated by program_table.py

    instcal : 'astropy.table.Table'
        instrument calendar table generated by instrument_table.py

    current_time : 'astropy.time.core.Time' [DEFAULT = None]
        Current time in simulation (used for setting start time of ToO time constraint)

    Returns
    -------
    targetcal : list of 'astropy.table.Table'
        List of target data tables generated by target_table.py.

    """

    verbose_progress = verbose  # print progress
    verbose = debug  # basic outputs
    verbose2 = debug  # detailed outputs

    # ====== Convert timing constraints to time windows ======
    if verbose_progress:
        print('...timing windows (convert time constraints)')

    # Compute all time windows of observations within scheduling period boundaries or program activation/deactivation
    # times.  Whichever are constraining.

    # print(obs['i_prog'].data[0])
    # print(obs['obs_id'].data[0])
    # print(progs['gemprgid'].data[obs['i_prog'].data[0]])
    # print(progs['prog_start'].data[obs['i_prog'].data[0]].iso)
    # print(progs['prog_end'].data[obs['i_prog'].data[0]].iso)
    # print(max(timetable['twilight_evening'].data[0], progs['prog_start'].data[obs['i_prog'].data[0]]))
    # print(min(timetable['twilight_morning'].data[-1], progs['prog_end'].data[obs['i_prog'].data[0]]))

    ncpu = cpu_count()
    time_windows = Parallel(n_jobs=ncpu)(
        delayed(convconstraint)(
            time_const=obs['time_const'][i],
            start=max(timetable['twilight_evening'].data[0], progs['prog_start'].data[obs['i_prog'].data[i]]),
            end=min(timetable['twilight_morning'].data[-1], progs['prog_end'].data[obs['i_prog'].data[i]]),
            current_time=current_time)
        for i in range(len(obs)))

    # # Use standard for loop for troubleshooting
    # time_windows = [timing_windows.convconstraint(time_const=obs['time_const'][i],
    #                                               start=timetable['twilight_evening'][0],
    #                                               end=timetable['twilight_morning'][-1])
    #                 for i in range(len(obs))]

    # ====== Timing windows (twilights) ======
    if verbose_progress:
        print('...timing windows (twilights)')
    # Constrain time windows to within nautical twilights
    time_windows = Parallel(n_jobs=ncpu)(delayed(twilights)(twilight_evening=timetable['twilight_evening'].data,
                                                          twilight_morning=timetable['twilight_morning'].data,
                                                          obs_windows=time_windows[i])
                                       for i in range(len(obs)))

    # ====== Sort time windows and observation indices by day ======
    if verbose_progress:
        print('...timing windows (organize into nights)')
    # By this point, timing windows are sorted by observation.
    # Reorganize time windows such that they are sorted by night.
    # For each night, make an array of indices corresponding to the
    # available observation time windows on that night.
    # Make a second array containing the corresponding timing window(s).
    i_obs_nightly = []  # Lists of observation indices (one per night).
    time_windows_nightly = []  # Lists of corresponding time windows.
    for i in range(len(timetable['date'])):
        i_obs_tonight, time_windows_tonight = \
            nightly_calendar(twilight_evening=timetable['twilight_evening'][i],
                             twilight_morning=timetable['twilight_morning'][i],
                             time_windows=time_windows)
        i_obs_nightly.append(np.array(i_obs_tonight))
        time_windows_nightly.append(time_windows_tonight)

    # # Use for loop for easier troubleshooting
    # time_windows = [timing_windows.twilights(twilight_evening=timetable['twilight_evening'].data,
    #                                          twilight_morning=timetable['twilight_morning'].data,
    #                                          obs_windows=time_windows[i])
    #                 for i in range(len(obs))]

    # for i in range(len(time_windows_nightly)):
    #     for j in range(len(time_windows_nightly[i])):
    #         print(i_obs_nightly[i][j], time_windows_nightly[i][j])

    # ====== Timing windows (instrument calendar) ======
    if verbose_progress:
        print('...timing windows (instrument calendar)')
    # Constrain time windows according to the installed instruments and
    # component configuration on each night

    i_obs_insts = Parallel(n_jobs=ncpu)(delayed(instrument)(i_obs=i_obs_nightly[i],
                                                          obs_inst=obs['inst'].data,
                                                          obs_disp=obs['disperser'].data,
                                                          obs_fpu=obs['fpu'].data,
                                                          obs_mos=obs['custom_mask_mdf'].data,
                                                          insts=instcal['insts'].data[i],
                                                          gmos_disp=instcal['gmos_disp'].data[i],
                                                          gmos_fpu=instcal['gmos_fpu'].data[i],
                                                          gmos_mos=instcal['gmos_mos'].data[i],
                                                          f2_fpu=instcal['f2_fpu'].data[i],
                                                          f2_mos = instcal['f2_fpu'].data[i])
                                            for i in range(len(timetable['date'])))

    # # Use for loop for easier troubleshooting
    # i_obs_insts = [instrument(i_obs=i_obs_nightly[i],
    #                           obs_inst=obs['inst'].data,
    #                           obs_disp=obs['disperser'].data,
    #                           obs_fpu=obs['fpu'].data,
    #                           insts=instcal['insts'].data[i],
    #                           gmos_disp=instcal['gmos_disp'].data[i],
    #                           gmos_fpu=instcal['gmos_fpu'].data[i],
    #                           f2_fpu=instcal['f2_fpu'].data[i])
    #                for i in range(len(timetable['date']))]

    # Get nightly observation indices and time windows from results of the instrument calendar
    for i in range(len(timetable['date'])):
        i_obs_nightly[i] = [i_obs_nightly[i][j] for j in i_obs_insts[i]]
        time_windows_nightly[i] = [time_windows_nightly[i][j] for j in i_obs_insts[i]]

    # print observation indices and corresponding time windows on each night
    if verbose2:
        for i in range(len(time_windows_nightly)):
            for j in range(len(time_windows_nightly[i])):
                print('obs index: ', i_obs_nightly[i][j])
                if time_windows_nightly[i][j] is None:
                    print('\t', None)
                else:
                    for window in time_windows_nightly[i][j]:
                        print('\t', window[0].iso, window[1].iso)

    # ====== Convert time windows to time grid indices ======
    if verbose_progress:
        print('...time window indices')
    dt = deltat(time_strings=timetable['utc'][0][0:2])  # time grid spacing
    i_wins_nightly = []
    for i in range(len(time_windows_nightly)):
        # i_wins_tonight = Parallel(n_jobs=10)(delayed(time_window_indices)(utc=timetable['utc'].data[i],
        #                                                                   time_wins=time_windows_nightly[i][j],
        #                                                                   dt=dt)
        #                                      for j in range(len(time_windows_nightly[i])))

        i_wins_tonight = [time_window_indices(utc=timetable['utc'].data[i],
                                              time_wins=time_windows_nightly[i][j],
                                              dt=dt)
                          for j in range(len(time_windows_nightly[i]))]
        i_wins_nightly.append(i_wins_tonight)

    # for i in range(len(i_wins_nightly)):
    #     for j in range(len(i_wins_nightly[i])):
    #         print(i_obs_nightly[i][j], i_wins_nightly[i][j])

    # ====== Target Calendar ======
    if verbose_progress:
        print('...target data')
    # Create list of 'astropy.table.Table' objects (one table per night).
    # Each table stores the positional data of each available target throughout
    # the night.
    # targetcal = Parallel(n_jobs=10)(delayed(target_table)(i_obs=i_obs_nightly[i],
    #                                                       latitude=site.location.lat,
    #                                                       lst=timetable['lst'].data[i] * u.hourangle,
    #                                                       utc=timetable['utc'].data[i],
    #                                                       obs_id=obs['obs_id'].data,
    #                                                       obs_ra=obs['ra'].quantity,
    #                                                       obs_dec=obs['dec'].quantity,
    #                                                       moon_ra=moon['ra'].data[i] * u.deg,
    #                                                       moon_dec=moon['dec'].data[i] * u.deg)
    #                                 for i in range(len(timetable['date'])))

    targetcal = [target_table(i_obs=i_obs_nightly[i],
                              latitude=site.location.lat,
                              lst=timetable['lst'].data[i] * u.hourangle,
                              utc=timetable['utc'].data[i],
                              obs_id=obs['obs_id'].data,
                              obs_ra=obs['ra'].quantity,
                              obs_dec=obs['dec'].quantity,
                              moon_ra=moon['ra'].data[i] * u.deg,
                              moon_dec=moon['dec'].data[i] * u.deg)
                 for i in range(len(timetable['date']))]

    # ====== Timing windows (elevation constraint) ======
    if verbose_progress:
        print('...timing windows (elevation constraint)')

    # Constrain timing windows to satisfy elevation constraints.
    # (can be either airmass or hour angle limits).
    # targetcal = Parallel(n_jobs=10)(delayed(elevation_const)(targets=targetcal[i],
    #                                                          elev_const=obs['elev_const'])
    #                                 for i in range(len(timetable['date'])))

    # Use for loop for troubleshooting
    i_wins_nightly = [elevation_const(targets=targetcal[i],
                                      i_wins=i_wins_nightly[i],
                                      elev_const=obs['elev_const'].data)
                      for i in range(len(timetable['date']))]

    # ====== Add time window column to target tables ======
    for i in range(len(targetcal)):
        targetcal[i]['i_wins'] = i_wins_nightly[i]

    # ====== Clean up target tables ======
    # Remove observations from target calender if the elevation constraint
    # process removed all timing windows for a given night.
    # Remove corresponding rows from tables in targetcal.
    for targets in targetcal:

        if len(targets) != 0:

            if verbose2:
                # print observation indices and remaining timing windows
                # one current night.
                for j in range(len(targets)):
                    print('\t', targets['i'][j])
                    if targets['i_wins'][j] is not None:
                        for k in range(len(targets['i_wins'][j])):
                            print('\t\t', targets['i_wins'][j][k][0], targets['i_wins'][j][k][1])
                    else:
                        print('\t\t', targets['i_wins'][j])

            # get indices for current night of observations with no remaining timing windows
            # ii_del = np.where([len(targets['time_wins'].data[j]) == 0 for j in range(len(targets))])[0][:]
            ii_del = np.where(targets['i_wins'].data == None)[0][:]
            if verbose2:
                print('ii_del', ii_del)

            # delete these rows from the corresponding targetcal target_tables
            if len(ii_del) != 0:
                for j in sorted(ii_del, reverse=True):  # delete higher indices first
                    targets.remove_row(j)

    # print target tables
    if verbose:
        [print(targets) for targets in targetcal]

    if verbose2:
        # print nightly observations and time windows
        for i in range(len(targetcal)):
            if len(targetcal[i]) != 0:
                print('\ntargetcal[i][\'i\']:\n', targetcal[i]['i'].data)
                print('\nNight (start,end):', timetable['utc'][i][0].iso, timetable['utc'][i][-1].iso)
                print('\nTwilights:', timetable['twilight_evening'][i].iso, timetable['twilight_morning'][i].iso)
                print('Date:', timetable['date'][i])
                print('time_windows:')
                for j in range(len(targetcal[i]['i_wins'])):
                    print('\ti:', targetcal[i]['i'][j])
                    if targetcal[i]['i_wins'][j] is not None:
                        for k in range(len(targetcal[i]['i_wins'].data[j])):
                            print('\t\t', targetcal[i]['i_wins'].data[j][k][0],
                                  targetcal[i]['i_wins'].data[j][k][1])
                    else:
                        print('\t\t', 'None')
            else:
                print('\ntargetcal[i][\'i\']:\n', targetcal[i])

    return targetcal


def test_checkwindow():
    print('\ntest_checkwindow()...')

    times = ['2018-07-02 03:20:00', '2018-07-02 06:45:00', '2018-07-03 02:30:00', '2018-07-03 04:45:00']
    utc = ['2018-07-01 22:49:57.001', '2018-07-02 10:37:57.001']
    print('times to check', times)
    print('window', utc)
    print(checkwindow(times, utc))

    assert checkwindow(times, utc).all() == np.array([True, True, False, False]).all()
    print('Test successful!')
    return


def test_i_time():
    print('\ntest_i_time()...')

    times = ['2018-07-02 03:20:00', '2018-07-02 06:45:00']
    utc = ['2018-07-01 22:49:57.001', '2018-07-01 23:49:57.001', '2018-07-02 00:49:57.001', '2018-07-02 01:49:57.001',
           '2018-07-02 02:49:57.001', '2018-07-02 03:49:57.001', '2018-07-02 04:49:57.001', '2018-07-02 05:49:57.001',
           '2018-07-02 06:49:57.001', '2018-07-02 07:49:57.001', '2018-07-02 08:49:57.001', '2018-07-02 09:49:57.001']
    print('times to get indices', times)
    print('time array', utc)
    print(i_time(times, utc))

    assert i_time(times, utc).all() == np.array([4, 7]).all()
    print('Test successful!')
    return


def test_constraint():
    print('\ntest_constraint()...')
    # time_const = '[{1524614400000 -1 0 0}]'
    time_const = '[{1488592145000 3600000 -1 140740000}]'
    # time_const = '[{1522713600000 3600000 -1 108000000}]'

    start = Time('2018-04-10 00:00:00')
    end = Time('2018-04-12 00:00:00')
    print('time_const', time_const)
    print('start, end of schedule window: ', start, end)

    const = convconstraint(time_const, start, end)
    print('Timing window (start,end):')
    [[print(c.iso) for c in con] for con in const]

    assert const[0][0].iso == Time('2018-04-10 10:08:45.000').iso
    assert const[0][1].iso == Time('2018-04-10 11:08:45.000').iso
    print('Test successful!')
    return


def test_instrument():
    from astropy.table import Table

    print('\ntest_instrument()...')

    inst = np.array(['GMOS'])
    disperser = np.array(['B'])
    fpu = np.array(['A'])
    i_obs = [np.array([0], dtype=int)]
    instcal = Table(np.array(['2018-01-01', 'GMOS-S', 'A,F', 'B,G', 'C']),
                    names=['date', 'insts', 'gmos_fpu', 'gmos_disp', 'f2_fpu'])

    print('i_obs', i_obs)
    print('inst', inst)
    print('disperser', disperser)
    print('fpu', fpu)
    print(instcal)

    window = instrument(i_obs=i_obs,
                        obs_inst=inst,
                        obs_disp=disperser,
                        obs_fpu=fpu,
                        insts=instcal['insts'].data[0],
                        gmos_disp=instcal['gmos_disp'].data[0],
                        gmos_fpu=instcal['gmos_fpu'].data[0],
                        f2_fpu=instcal['f2_fpu'].data[0])

    print('Valid indices:', window)
    assert window[0] == 0
    print('Test successful!')
    return


if __name__ == '__main__':
    test_constraint()
    test_checkwindow()
    test_i_time()
    test_instrument()
    print('\nCOMPLETE!')
