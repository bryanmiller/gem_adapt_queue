# Matt Bonnyman 23 July 2018
# This module holds a variety of functions that are used for printing info to the terminal or to a log file.

# astroconda packages
import os
import time as t
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy import coordinates

#gemini packages
from dt import deltat

def shortid(obsid):
    """
    Compact format for the observation id, like QPT

    Parameters
    ----------
    obsid : string
        Program id string

    Returns
    -------
    shortid : string
        Compact format
    """

    idvals = obsid.split('-')

    shortid = idvals[0][-1] + idvals[1][2:] + '-' + idvals[2] + '-' + idvals[3] + '[' + idvals[4] + ']'

    return shortid

def _get_order(plan):
    """
    For observations scheduled in plan, get the observation indices in the order they appear, and return the
    array indices of schedule period boundaries observation.

    Example
    -------
    >>> plan = [2, 2, 2, 2, 1, 1, 1, 1, 5, 5, 4, 4, 4, 4]
    >>> ind_order, i_start, i_end = _get_order(plan)
    >>> print(ind_order)
    >>> print(i_start)
    >>> print(i_end)
    [2, 1, 5, 4]
    [0, 4, 8, 10]
    [3, 7, 9, 13]

    Parameters
    ----------
    plan : numpy integer array
        Observation indices throughout night.

    Returns
    -------
    order : list of ints
        order that indices appear in plan.

    i_start : list of ints
        indices of time block beginnings corresponding to plan.

    i_end : list of ints
        indices of time block endings corresponding to plan.
    """

    ind_order = [plan[0]]
    i_start = [0]
    i_end = []
    for i in range(1, len(plan)):
        prev = plan[i-1]
        if plan[i] != prev:
            ind_order.append(plan[i])
            i_end.append(i-1)
            i_start.append(i)
        if i == len(plan)-1:
            i_end.append(i)
    return ind_order, i_start, i_end


def plantable(plan, obs, targets, timetable, description=''):
    """
    Get a table of the current plan as a list of strings.

    Parameters
    ----------
    plan : numpy integer array
        indices of targets from 'targets' at times throughout night.

    obs : '~astropy.table.Table'
        Observation data table or subset of table created by obstable.py.

    targets : '~astropy.table.Table'
        target information table created by targettable.py.

    timetable : '~astropy.table.Table'
        Row of time information table created by timetable.py corresponding to plan period.

    Returns
    -------
    lines : list of strings
        Rows of table stored as a list of strings.
    """

    verbose = False

    dt = deltat(time_strings=timetable['utc'][0][0:2])

    obs_order, i_start, i_end = _get_order(plan=plan)

    if verbose:
        print('len(plan)', len(plan))
        print('plan', plan)
        print('obs_order', obs_order)
        print('i_start', i_start)
        print('i_end', i_end)

    sprint = '{0:17.15s}{1:12.10s}{2:7.5s}{3:7.5s}{4:7.5s}{5:7.5s}{6:7.5s}{7:7.5s}{8:7.5s}{9:7.5s}{10:6.4s}' \
             '{11:7.5s}{12:10.10}'

    lines = ['', '', '{:>60s}'.format(str('-- ' + timetable['date'][0] + ' schedule '+description+'--')), '',
             str(sprint.format('Obs. ID', 'Target', 'RA', 'Dec.', 'Instr.', 'UTC', 'LST', 'Start', 'End', 'Dur.', 'AM',
                               'HA', 'Complete')),
             str(sprint.format('-------', '------', '--', '---', '-----', '---', '---', '-----', '---', '---', '--',
                               '--', '--------')),
             str(sprint.format('12 deg.twi.', '', '', '', '', timetable['utc'][0][0].iso[11:16],
                               str('{:.2f}'.format(timetable['lst'][0][0])), timetable['local'][0][0].iso[11:16],
                               '', '', '', '', ''))]

    for i in range(len(obs_order)):
        if obs_order[i] >= 0:

            if verbose:
                print(shortid(targets['id'][obs_order[i]]))
                print(obs['target'][obs_order[i]])
                print(obs['inst'][obs_order[i]])
                print(timetable['utc'][0][i_start[i]][11:16])
                print(str('{:.2f}'.format(timetable['lst'].quantity[0][i_start[i]])))
                print(timetable['local'][0][i_start[i]].iso)
                print(timetable['local'][0][i_end[i]].iso)
                print((Time(timetable['local'][0][i_end[i]]) -
                       Time(timetable['local'][0][i_start[i]]) + dt).to(u.hr).round(2))
                print(str('{:.2f}'.format(targets['AM'][obs_order[i]][i_start[i]])))
                print(str('{:.2f}'.format(targets['HA'][obs_order[i]][i_start[i]])))
                print(str(obs['obs_comp'][obs_order[i]] >= 1.))

            obs_name = shortid(targets['id'][obs_order[i]])
            ra = obs['ra'].quantity[obs_order[i]].round(2)
            dec = obs['dec'].quantity[obs_order[i]].round(2)
            targ_name = obs['target'][obs_order[i]]
            inst_name = obs['inst'][obs_order[i]]
            utc_start = timetable['utc'][0][i_start[i]].iso
            lst_start = str('{:.2f}'.format(timetable['lst'][0][i_start[i]]))
            local_start = timetable['local'][0][i_start[i]].iso
            local_end = (timetable['local'][0][i_end[i]] + dt).iso
            duration = (timetable['local'][0][i_end[i]] - timetable['local'][0][i_start[i]] + dt).to(u.hr).round(2)
            airmass = str('{:.2f}'.format(targets['AM'][obs_order[i]][i_start[i]]))
            HA = str('{:.2f}'.format(targets['HA'][obs_order[i]][i_start[i]]))
            cplt = str(obs['obs_comp'][obs_order[i]] >= 1.)
            lines.append(str(sprint.format(obs_name, targ_name, ra, dec, inst_name, utc_start[11:16], lst_start,
                                           local_start[11:16], local_end[11:16], duration, airmass, HA, cplt)))

    lines.append(str(sprint.format('12 deg. twi.', '', '', '', '',
                                   timetable['utc'][0][-1].iso[11:16],
                                   str('{:.2f}'.format(timetable['lst'][0][-1])),
                                   timetable['local'][0][-1].iso[11:16], '',
                                   '', '', '', '')))
    return lines


def listobs(obs):
    """
    Get a table of observation, program, and group names for a set of observations in 'obs' in the form of a list of
    strings.

    Parameters
    ----------
    obs : '~astropy.table.Table'
        Observation data table or subset of table created by obstable.py.

    Returns
    -------
    lines : list of strings
            Rows of table stored as a list of strings.
    """

    sprint = '\t{0:23s}{1:21s}{2:21.15s}{3:40.40}'

    lines = ['', sprint.format('Observation', 'Program', 'Target', 'Group'),
             sprint.format('-----------', '-------', '------', '-----')]

    for i in range(len(obs)):
        j = np.where(obs['obs_id'] == obs['obs_id'][i])[0][0]
        lines.append(sprint.format(obs['obs_id'][j], obs['prog_ref'][j], obs['target'][j], obs['group'][i]))

    return lines


def timeinfo(solar_midnight, utc_to_local):
    """
    Get table of time info for current scheduling period.

    Parameters
    ----------
    solar_midnight : '~astropy.time.core.Time'
        UTC solar midnight time.

    utc_to_local : 'astropy.units'
        hour difference to convert from UTC to local time.

    Returns
    -------
    lines : list of strings
            Output info as a list of lines.
    """
    fprint = '{0:<35s}{1}'
    local_solar_mid = solar_midnight + utc_to_local
    return ['', fprint.format('Solar midnight (UTC):', solar_midnight.iso[:16]),
            fprint.format('Solar midnight (local):', local_solar_mid.iso[:16])]


def suninfo(ra, dec):
    """
    Get table of sun info for current scheduling period.

    Parameters
    ----------
    ra : 'astropy.units'
        Sun right ascension at solar midnight

    dec : 'astropy.units'
        Sun declination at solar midnight

    Returns
    -------
    lines : list of strings
            Output info as a list of lines.
    """
    fprint = '{0:<35s}{1}'
    return ['', fprint.format('Sun ra: ', ra.round(2)),
            fprint.format('Sun dec: ', dec.round(2))]


def mooninfo(ra, dec, frac, phase):
    """
    Get table of moon info for current scheduling period.

    Parameters
    ----------
    ra : 'astropy.units'
        Moon right ascension at solar midnight

    dec : 'astropy.units'
        Moon declination at solar midnight

    frac : float
        Fraction of Moon illuminated at solar midnight

    phase : 'astropy.units'
        Moon phase angle at solar midnight

    Returns
    -------
    lines : list of strings
            Output info as a list of lines.
    """
    fprint = '{0:<35s}{1}'
    return ['', fprint.format('Moon ra: ', ra.round(2)),
            fprint.format('Moon dec: ', dec.round(2)),
            fprint.format('Moon fraction: ', frac.round(2)),
            fprint.format('Moon phase: ', phase.round(2))]


def skyinfo(iq, cc, wv, conddist, skycond):
    """
    Get string of sky conditions info for current scheduling period.

    Parameters
    ----------
    iq : string
        Image quality

    cc : string
        Cloud condition

    wv : string
        Water vapor

    conddist : string or None
        Condition distribution type if generated randomly.

    skycond : 'astropy.table.Table'
        Sky conditions table with columns iq, cc, wv.

    Returns
    -------
    lines : list of strings
        Output info as a list of lines.
    """
    if conddist is None:
        fprint = '{0:<35s}({1}, {2}, {3})'
        return ['', fprint.format('Sky conditions (iq, cc, wv): ', iq, cc, wv)]
    else:
        fprint = '{0:<35s}{1} (iq={2}, cc={3}, wv={4})'
        iq = skycond['iq'].data[0]
        cc = skycond['cc'].data[0]
        wv = skycond['wv'].data[0]
        return ['', fprint.format('Sky conditions:', conddist, iq, cc, wv)]


def too_event(type, time):
    """
    Print type and time of incoming target of opportunity.

    Example
    -------


    Parameters
    ----------
    type : str
        Rapid or Standard type ToO

    time : 'astropy.time.Time'
        local time of conditions change event (local time)

    Returns
    -------
    lines : list of strings
        Output info as a list of lines.
    """

    fprint = '\n\tAt {0} local time, {1} added to queue.'
    return [fprint.format(time.iso[11:16], type)]


def conditions_event(iq, cc, wv, time):
    """
    Print new sky conditions and local time of conditions change

    Example
    -------


    Parameters
    ----------
    iq : float
        Image quality

    cc : float
        Cloud condition

    wv : float
        Water vapor

    time : 'astropy.time.Time'
        local time of conditions change event (local time)

    Returns
    -------
    lines : list of strings
        Output info as a list of lines.
    """
    if iq < 1:
        iq = str(int(round(iq*100, 0))) + '%'
    else:
        iq = 'Any'

    if cc < 1:
        cc = str(int(round(cc*100, 0))) + '%'
    else:
        cc = 'Any'

    if wv < 1:
        wv = str(int(round(wv*100, 0))) + '%'
    else:
        wv = 'Any'

    fprint = '\n\tAt {0} local time, sky conditions change to iq={1}, cc={2}, wv={3}.'
    return [fprint.format(time.iso[11:16], iq, cc, wv)]


def windinfo(dir, vel):
    """
    Get string of wind direction and velocity for current scheduling night.

    Parameters
    ----------
    dir : 'astropy.units'
        wind direction

    vel : 'astropy.units'
        wind velocity

    Returns
    -------
    lines : list of strings
        Output info as a list of lines.
    """
    fprint = '{0:<35s}{1}, {2}'
    return [fprint.format('Wind conditions (dir., vel.): ', str(dir.round(2)), str(vel.round(2)))]


def programinfo(progname, cwd):
    """
    Get program and directory info as list of strings.

    Example
    -------
    >>> info = printer.programinfo('myprogram.py', '/Users/mbonnyman/github_repository/jul_26')
    >>> [print(line) for line in info]
    Program: gqpt.py
	Run from directory: /Users/mbonnyman/github_repository/jul_26

    Parameters
    ----------
    progname : str
        name of program

    cwd : str
        current working directory

    Returns
    -------
    lines : list of strings
        lines of program parameters text
    """
    return ['File created at '+ t.strftime('%d%b%y-%H:%M:%S', t.gmtime()),
            'Program: ' + str(progname),
            'Run from directory: ' + str(cwd)]


def inputfiles(otfile, instfile, toofile, prfile):
    """
    Get input file info as list of strings.

    Example
    -------
    >>> info = printer.inputfiles('observations.txt', 'instruments.txt', 'too.txt')
    >>> [print(line) for line in info]
    Observation information retrieved from observations.txt
	Instrument calendar retrieved from instruments.txt
    ToO information retrieved from too.txt

    Parameters
    ----------
    otfile : str
        observation catalog file name

    instfile : str
        instrument calendar file name

    prfile : str
        program status file name

    toofile : str or None
        target of opportunity file name

    Returns
    -------
    lines : list of strings
        lines of program parameters text

    """

    lines = ['',
             'Observation information retrieved from ' + otfile,
             'Program status information retrieved from ' + prfile,
             'Instrument calendar retrieved from ' + instfile]

    if toofile is not None:
        lines.append('Target of Opportunity information retrieved from ' + toofile)

    return lines


def parameters(site, dates, daylightsavings):
    """
    Get scheduling dates and observatory site info as list of strings.

    Parameters
    ----------
    site : '~astroplan.Observer'
        observer site

    dates : array or list of strings
        dates for current simulation period

    daylightsavings : bool

    Returns
    -------
    lines : list of strings
        lines of program parameters text
    """

    aprint = '\t{0:<15s}{1}'  # print two strings
    bprint = '\t{0:<15s}{1:<.4f}'  # print string and number
    return ['', 'Dates: ' + dates[0] + ' to ' + dates[-1], 'Number of nights: ' + str(len(dates)),
            'Daylight savings time: ' + str(daylightsavings), '', 'Observatory: ',
            aprint.format('Site: ', site.name), bprint.format('Height: ', site.location.height),
            bprint.format('Longitude: ', coordinates.Angle(site.location.lon)),
            bprint.format('Latitude: ', coordinates.Angle(site.location.lat)),]


def planoptions(too_prob, too_max, cond_change_prob, cond_change_max, iq, cc, wv, conddist, direction, velocity,
                random):
    """
    Print simulation plan options.

    Parameters
    ----------
    too_prob : float
        Probability of each potential ToO arriving during the night

    too_max : int
        Number of potential ToOs per night

    cond_change_prob : float
        Probability of conditions changing during the night

    cond_change_max : int
        Number of potential conditions changing during per night

    iq : string
        Image quaility percentile

    cc : string
        Cloud conditions percentile

    wv : string
        Water vapour percentile

    conddist : string or None
        Condition distribution type if generating random viewing conditions

    direction : astropy.units degrees
        Wind direction

    velocity : astropy.units meters/second
        Wind velocity

    random : boolean
        Generate wind randomly from location mean and standard deviation

    Return
    ------
    lines of strings
    """
    return ['', 'ToO probability: ' + str(too_prob),
            'Max. number of ToOs per night: ' + str(too_max),
            'Condition change probability: ' + str(cond_change_prob),
            'Max. number of condition changes per night: ' + str(cond_change_max),
            '',
            'Condition inputs (iq,cc,wv): ({0}, {1}, {2})'.format(iq, cc, wv),
            'Generate conditions from distribution: ' + str(conddist),
            'Wind conditions: ({}deg, {}m/s)'.format(direction, velocity),
            'Random wind conditions: {}'.format(random)]


def queuestatus(progs, simstats, description):
    """
    Get observation statistics as lines of text.

    Example
    -------
    >>> text = queuestatus(progs=progs, simstats=simstats, description='This is the state of the current queue')
    >>> [print(line) for line in text]

     This is the state of the current queue...

     Program             Completion     Total time
     -------             ----------     ----------
     GS-2018A-A-1        100%           1.02 h
     GS-2018A-B-1        2.8%           2.68 h
     GS-2018A-B-2        0.03%          6.68 h
     GS-2018A-C-1        1.02%          12.7 h
     GS-2018A-C-2        100%           0.4 h
     GS-2018A-C-3        30.32%         14.84 h
     GS-2018A-C-4        43.74%         2.51 h
     GS-2018A-C-5        44.96%         0.63 h
     GS-2018A-C-6        0.96%          21.91 h
     GS-2018A-C-7        83.34%         1.82 h
     GS-2018A-C-8        93.1%          12.03 h
     GS-2018A-C-9        65.69%         14.61 h
     GS-2018A-C-10       9.68%          4.14 h
     GS-2018A-C-11       51.86%         9.17 h

     Number of programs: 28
     Total program time: 341.63 h
     Total time completion: 10.35%

     Completed: 2
     Observed time: 1.43 h

     Partially completed: 12
     Observed time: 33.77 h
     Remaining time: 69.95 h

     Not started: 14
     Remaining time: 236.48 h

    Parameters
    ----------
    obs : '~astropy.table.Table'
        observation information table created by obstable.py

    simstats : dict
        simulation statistics for total observable time and total observed time.
         (i.e. simstats = {'night_time': float * u.h, 'used_time': float * u.h})

    description: string
        A description of when the queue statistics were obtained

    Returns
    -------
    lines : list of strings
        lines of text
    """

    prog_comp_time = 0. * u.h  # total time of completed programs
    prog_part_alloc = 0. * u.h  # allocated time of partially completed programs
    prog_part_time = 0. * u.h  # total observed time of partially completed programs
    prog_notstart_time = 0. * u.h  # allocated time of not observed programs
    num_prog_comp = 0  # number of completed programs
    num_prog_started = 0  # number of partially completed programs
    num_prog_notstart = 0  # number of not yet started programs

    aprint = '{0:<20}{1:<15}{2:<15}'

    lines = ['', '', '\t-- ' + description + ' --', '', aprint.format('Program', 'Completion', 'Observed time'),
             aprint.format('-------', '----------', '-------------')]

    for i in range(len(progs)):
        if progs['prog_comp'][i] >= 1.:
            perc_comp = '100%'
            prog_comp_time = prog_comp_time + progs['prog_time'].quantity[i]
            num_prog_comp = num_prog_comp + 1
        elif 0. < progs['prog_comp'][i] < 1.:
            perc_comp = str((100*progs['prog_comp'][i]).round(2))+'%'
            prog_part_alloc = prog_part_alloc + progs['alloc_time'].quantity[i]
            prog_part_time = prog_part_time + progs['prog_time'].quantity[i]
            num_prog_started = num_prog_started + 1
        else:
            perc_comp = '0.00%'
            prog_notstart_time = prog_notstart_time + progs['alloc_time'].quantity[i]
            num_prog_notstart = num_prog_notstart + 1

        # append program info to list of strings.
        lines.append(aprint.format(progs['gemprgid'][i], perc_comp, str(progs['prog_time'][i].round(2))))

    tot_prog_time = sum(progs['alloc_time'].quantity)
    tot_obs_time = sum(progs['prog_time'].quantity)

    lines.append('')
    lines.append('Number of programs: ' + str(len(progs)))
    lines.append('Total program time: ' + str(tot_prog_time.round(2)))
    lines.append('Total time completion: ' + str((100 * tot_obs_time / tot_prog_time).round(2)) + '%')

    lines.append('')
    lines.append('Total available time: ' + str(simstats['tot_time'].round(2)))
    lines.append('Total scheduled time: ' + str(simstats['used_time'].round(2)))

    lines.append('')
    lines.append('Completed: ' + str(num_prog_comp))
    lines.append('Observed time: '+str(prog_comp_time.round(2)))

    lines.append('')
    lines.append('Partially completed: ' + str(num_prog_started))
    lines.append('Observed time: ' + str(prog_part_time.round(2)))
    lines.append('Remaining time: ' + str((prog_part_alloc-prog_part_time).round(2)))

    lines.append('')
    lines.append('Not started: ' + str(num_prog_notstart))
    lines.append('Remaining time: ' + str(prog_notstart_time.round(2)))
    return lines


def overwrite_log(filename, lines):
    """
    Create a new log file and write a list of strings.
    A '\n' newline character is automatically placed at the start of
    each line.

    Parameters
    ----------
    filename : string
        file name including extension type (eg. 'logfile26Jul18.log').

    lines : list of strings
        lines of text to append to log file
    """

    fprint = '\n {}'  # write each element of list on own line

    with open(filename, 'w') as log:
        [log.write(fprint.format(line)) for line in lines]
        log.close()
    return


def append_to_file(filename, lines):
    """
    Append a list of strings to log file. A '\n' newline character is automatically placed at the start of
    each line.

    Parameters
    ----------
    filename : string
        file name including extension type (eg. 'logfile26Jul18.log').

    lines : list of strings
        lines of text to append to log file
    """

    fprint = '\n {}'  # write each element of list on own line

    with open(filename, 'a') as log:
        [log.write(fprint.format(line)) for line in lines]
        log.close()
    return


def def_filename(name, ext):
    """
    Return a timestamped .log file name of the form

    'programnameDDMMMYY-hh:mm:ss.log'

    Example
    -------
    >>> from printer import def_filename
    >>> filename = def_filename('myprogram', 'log')
    >>> print(filename)
     myprogram26Jul18-20:20:55.log

    Parameters
    ----------
    name : string
        name of program or other identifier to be part of timestamp string.

    ext : string
        extension type of file.
    """
    return name + t.strftime('%d%b%y-%H:%M:%S', t.gmtime()) + '.' + ext
