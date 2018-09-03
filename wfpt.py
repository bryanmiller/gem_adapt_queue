# Matt Bonnyman 24 July 2018

# astroconda packages
import argparse
import astropy.units as u
from astroplan import download_IERS_A
from astropy.table import Table, Column
from astropy.time import Time
import importlib
import numpy as np
import textwrap

# gemini packages
from sb import sb
import convert_conditions
from condition_table import condition_table
from dates import getdates
import make_plot
import printer
import program_table
from time_table import time_table
import timing_windows
import weights
from wind_table import wind_table
from observation_table import observation_table
from sun_table import sun_table
from moon_table import moon_table
from instrument_table import instrument_table
import select_obs as select_obs
from observer_site import getsite
import convert_conditions as convertcond


def getobsindex(obs_id, obs):
    """
    Retrieve the index for the row of an observation the observation data table 'obs'.

    Parameters
    ----------
    obs_id : string
        Uniqie observation identifier

    obs : '~astropy.table.Table'
        Observation data table or subset of table.

    Returns
    -------
    integer of row index corresponding to selected observation.
    """

    ii = np.where(obs['obs_id'] == obs_id)[0][:]
    if len(ii) == 1:
        return ii[0]
    elif len(ii) == 0:
        raise ValueError(' \'{}\' not found.'.format(obs_id))
    elif len(ii) > 1:
        raise ValueError('Found ' + str(len(ii)) + ' observations with identifier ' + obs_id)
    else:
        raise ValueError('Something went wrong.')


def weightplotmode(site, timetable, sun, moon, obs, progs, targets, skycond, wind, wra):
    """
    Select and plot the weighting function of any observation on the current date.  When this
    function is run separately from the Gemini Queue Planning Tool, it will not generate a plan,
    and will therefore not be able to print the plan table, and will not require the 'plan' or 'targets'
    variables.

    Parameters
    ----------
    site : '~astroplan.Observer'
        Observatory site information.

    timetable : '~astropy.table.Table'
        time information table created by time_table.py

    sun : '~astropy.table.Table'
        sun information table created by sun_table.py

    moon : '~astropy.table.Table'
        moon information table created by moon_table.py

    obs : '~astropy.table.Table'
        observation information table created by observation_table.py

    progs : '~astropy.table.Table'
        program status information table created by observation_table.py

    targets : '~astropy.table.Table'
        target position data table for observing period created by target_table.py

    skycond : '~astropy.table.Table'
        sky conditions information table created by condition_table.py

    wind : '~astropy.table.Table'
        wind conditions information table created by wind_table.py

    wra : numpy float array
        Right ascension distribution weights computed from full set of observations.s

    """
    verbose = False

    aprint = '\t{0:<20s}{1}'  # print two strings
    fprint = '\t{0:5s}{1:40}{2:40s}'  # format options menu

    print('\n\n\t---------- Weight function plotting mode ----------\n')
    print(aprint.format('Plan date:', timetable['date'][0]))
    while True:
        # -- Weight function plotting mode menu --
        print('\n\tOptions:')
        print('\t--------')
        print(fprint.format('1.', 'See list of available observations', '-'))
        print(fprint.format('2.', 'Conditions (iq,cc,wv)',
                            '({}, {}, {})'.format(str(skycond['iq'][0]), str(skycond['cc'][0]), str(skycond['wv'][0]))))
        print(fprint.format('3.', 'Wind conditions (dir, vel)',
                            '({}, {})'.format(wind['dir'].quantity[0], wind['vel'].quantity[0])))

        userinput = input('\n Select option or provide an observation identifier: ')

        if userinput == '1':
            [print(line) for line in printer.listobs(obs=Table(obs['obs_id', 'prog_ref', 'group', 'target']))]
            continue

        elif userinput == '2':
            condinput = input(' Input new iq, cc, wv percentiles(eg. \'20 50 Any\'): ')
            tempconds = condinput.split(' ')
            if len(tempconds) != 3:
                print(' Did not receive 3 values. No changes were made.')
                continue
            else:
                try:
                    iq, cc, wv = convertcond.inputcond(iq=tempconds[0], cc=tempconds[1], wv=tempconds[2])
                    skycond = condition_table(size=len(timetable['utc'].data[0]), iq=iq, cc=cc, wv=wv)
                except ValueError:
                    print(' ValueError: Could not set new conditions. Changes not made.')
                    continue

        elif userinput == '3':
            windinput = input(' Input new direction(deg), velocity(m/s) (eg. \'330 5\'): ')
            tempwind = windinput.split(' ')
            if len(tempwind) != 2:
                print(' Did not receive 2 values. No changes were made.')
                continue
            else:
                try:
                    wind = wind_table(size=len(timetable['utc'][0]), direction=float(tempwind[0]),
                                      velocity=float(tempwind[1]), site_name=site.name)
                except ValueError:
                    print(' ValueError: Could not set new wind conditions.')
                    continue

        elif userinput.lower() == 'quit' or userinput.lower() == 'exit' or userinput.lower() == 'q':
            break

        else:
            try:  # retrieve table row number
                i = getobsindex(obs_id=userinput, obs=Table(obs))
            except ValueError as e:
                print(e)
                continue

            importlib.reload(make_plot)
            importlib.reload(weights)

            target = Table(targets[i])

            # ====== Compute visible sky brightnesses at targets ======
            target['vsb'] = Column([sb(mpa=moon['phase'].quantity[0],
                                       mdist=targets['mdist'].quantity[i],
                                       mZD=moon['ZD'].data[0] * u.deg,
                                       ZD=targets['ZD'].quantity[i],
                                       sZD=sun['ZD'].data[0] * u.deg,
                                       cc=skycond['cc'].data)])


            # ====== Convert vsb to sky background percentiles ======
            target['bg'] = Column([convert_conditions.sb_to_cond(sb=target['vsb'][0])])

            target['weight'] = Column(
                [weights.obsweight(
                    obs_id=obs['obs_id'][i],
                    ra=obs['ra'].quantity[i],
                    dec=obs['dec'].quantity[i],
                    iq=obs['iq'].data[i],
                    cc=obs['cc'].data[i],
                    bg=obs['bg'].data[i],
                    wv=obs['wv'].data[i],
                    elev_const=obs['elev_const'][i],
                    i_wins=target['i_wins'][0],
                    band=obs['band'].data[i],
                    user_prior=obs['user_prior'][i],
                    AM=target['AM'].data[0],
                    HA=target['HA'].data[0] * u.hourangle,
                    AZ=target['AZ'].quantity[0],
                    latitude=site.location.lat,
                    prog_comp=progs['prog_comp'].data[obs['i_prog'].data[i]],
                    obs_comp=obs['obs_comp'].quantity[i],
                    skyiq=skycond['iq'].data,
                    skycc=skycond['cc'].data,
                    skywv=skycond['wv'].data,
                    skybg=target['bg'].data[0],
                    winddir=wind['dir'].quantity,
                    windvel=wind['vel'].quantity,
                    wra=wra[i])])


            make_plot.weightcomponents(
                    obs_id=obs['obs_id'][i],
                    ra=obs['ra'].quantity[i],
                    dec=obs['dec'].quantity[i],
                    iq=obs['iq'].data[i],
                    cc=obs['cc'].data[i],
                    bg=obs['bg'].data[i],
                    wv=obs['wv'].data[i],
                    elev_const=obs['elev_const'][i],
                    i_wins=target['i_wins'][0],
                    band=obs['band'].data[i],
                    user_prior=obs['user_prior'][i],
                    AM=target['AM'].data[0],
                    HA=target['HA'].data[0] * u.hourangle,
                    AZ=target['AZ'].quantity[0],
                    latitude=site.location.lat,
                    prog_comp=progs['prog_comp'].data[obs['i_prog'].data[i]],
                    obs_comp=obs['obs_comp'].quantity[i],
                    skyiq=skycond['iq'].data,
                    skycc=skycond['cc'].data,
                    skywv=skycond['wv'].data,
                    skybg=target['bg'].data[0],
                    winddir=wind['dir'].quantity,
                    windvel=wind['vel'].quantity,
                    wra=wra[i],
                    localtimes = timetable['local'].data[0])

            make_plot.weightfunction(obs_id=userinput,
                                     local_time=timetable['local'].data[0],
                                     date=timetable['date'].data[0],
                                     weight=target['weight'].quantity[0])

            make_plot.skyconditions(skycond=skycond,
                                    local_time=timetable['local'].data[0],
                                    date=timetable['date'].data[0],
                                    bg=target['bg'].quantity[0])

            make_plot.windconditions(wind=wind,
                                     local_time=timetable['local'].data[0],
                                     date=timetable['date'].data[0])
    return


def wfpt():

    parser = argparse.ArgumentParser(prog='wfpt.py',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent('''                                                             
                                        Weight function plotting tool
        *****************************************************************************************************               
    
            otfile                  OT catalog file name.
            
            prfile                  Gemini exechours program status file name.
            
            instcal                 Instrument calendar filename.
    
            -o   --observatory      Observatory site [DEFAULT='gemini_south']. Accepts the following:
                                    1. 'gemini_north' (or 'MK' for Mauna Kea)
                                    2. 'gemini_south' (or 'CP' for Cerro Pachon)
    
            -d   --date             Date 'YYYY-MM-DD' [DEFAULT=current].
    
            -dst --daylightsavings  Toggle daylight savings time [DEFAULT=False].
            
            -dt  --gridsize         Size of time-grid spacing [DEFAULT=0.1hr].
    
                                    Sky conditions:
            -i   --iq               Image quality constraint [DEFAULT=70].
            -c   --cc               Cloud cover constraint   [DEFAULT=50].
            -w   --wv               Water vapor constraint   [DEFAULT=Any].
                                    
                                    Wind conditions:
            -dir --direction        Wind direction [DEFAULT=270deg].
            -vel --velocity         Wind velocity [DEFAULT=10deg].
            
            -rw  --randwind         Random wind conditions (use mean and standard deviation of site):
                                        Cerro Pachon : dir=(330 +/- 20)deg, vel=(5 +/- 3)m/s
                                        Mauna Kea    : dir=(330 +/- 20)deg, vel=(5 +/- 3)m/s
            
            -u   --update           Download up-to-date IERS(International Earth Rotation and Reference Systems).
    
        *****************************************************************************************************                        
                                        '''))

    parser.add_argument(action='store',
                        dest='otfile')

    parser.add_argument(action='store',
                        dest='prfile')

    parser.add_argument(action='store',
                        dest='instfile')

    parser.add_argument('-o', '--observatory',
                        action='store',
                        default='gemini_south')

    parser.add_argument('-d', '--date',
                        action='store',
                        default=None)

    parser.add_argument('-dst', '--daylightsavings',
                        action='store_true',
                        dest='dst',
                        default=False)

    parser.add_argument('-dt', '--gridsize',
                        action='store',
                        default=0.1)

    parser.add_argument('-iq', '--iq',
                        default='70')

    parser.add_argument('-cc', '--cc',
                        default='50')

    parser.add_argument('-wv', '--wv',
                        default='Any')

    parser.add_argument('-dir', '--direction',
                        default=330)

    parser.add_argument('-vel', '--velocity',
                        default=5)

    parser.add_argument('-rw', '--randwind',
                        action='store_true',
                        default=False)

    parser.add_argument('-u', '--update',
                        action='store_true',
                        default=False)


    parse = parser.parse_args()
    otfile = parse.otfile
    prfile = parse.prfile
    instfile = parse.instfile
    iq, cc, wv = convertcond.inputcond(parse.iq, parse.cc, parse.wv)
    dst = parse.dst
    dir = parse.direction
    vel = parse.velocity
    randwind = parse.randwind

    if parse.update:  # download most recent International Earth Rotation and Reference Systems data
        download_IERS_A()

    verbose = False
    verbose_progress = True

    # Time grid spacing size in hours
    dt = float(parse.gridsize) * u.h
    if dt <= 0 * u.h:
        raise ValueError('Time grid spacing must be greater than 0.')


    # ====== Get Site Info ======
    if verbose_progress:
        print('...observatory site, time_zone, utc_to_local')
    # Create 'astroplan.Observer' object for observing site.
    # Get time-zone name (for use by pytz) and utc_to_local time difference.
    site, timezone_name, utc_to_local = getsite(site_name=parse.observatory,
                                                daylightsavings=dst)


    # ====== Plan start/end dates ======
    if verbose_progress:
        print('...scheduling period dates')
    # Check format of command line input dates.
    # Create 'astropy.time.core.Time' objects for start and end of plan period
    start, end = getdates(startdate=parse.date,
                          enddate=None,
                          utc_to_local=utc_to_local)

    # ====== Time data table for scheduling period ======
    if verbose_progress:
        print('...time data and grids')
    # Create 'astropy.table.Table' (one row for each day in plan period)
    # Stores time grids for UTC, local, lst.
    # Stores solar midnight, evening/morning nautical twilights
    timetable = time_table(site=site,
                           utc_to_local=utc_to_local,
                           dt=dt,
                           start=start,
                           end=end)

    # set start and end times as boundaries of scheduling period
    start = Time(timetable[0]['utc'][0])
    end = Time(timetable[-1]['utc'][-1])

    # ====== Sun data table for scheduling period ======
    if verbose_progress:
        print('...Sun data')
    # Create 'astropy.table.Table' (one row for each day in plan period)
    # Stores ra, dec at midnight on each night.
    # Stores azimuth angle, zenith distance angle, and hour angle
    # throughout scheduling period.
    sun = sun_table(latitude=site.location.lat,
                    solar_midnight=timetable['solar_midnight'].data,
                    lst=timetable['lst'].data)

    # ====== Moon data table for scheduling period ======
    if verbose_progress:
        print('...Moon data')
    # Create 'astropy.table.Table' (one row for each day in plan period)
    # Stores fraction illuminated, phase angle, ra, and dec at solar midnight
    # on each night.
    # Stores ra, dec, azimuth angle, zenith distance angle, hour angle, airmass
    # throughout scheduling period.
    moon = moon_table(site=site,
                      solar_midnight=timetable['solar_midnight'].data,
                      utc=timetable['utc'].data,
                      lst=timetable['lst'].data)

    # ====== Assemble Observation Table ======
    if verbose_progress:
        print('...observations')
    # Create 'astropy.table.Table' (one observation per row)
    obs = observation_table(filename=otfile)

    # ====== Assemble Program Table ======
    if verbose_progress:
        print('...programs')
    # read columns from exechours_YYYYL.txt file into 'astropy.table.Table' object
    exechourtable = program_table.read_exechours(filename=prfile)

    # retrieve additional program information from the observation table (if available)
    proginfo = program_table.get_proginfo(exechourtable=exechourtable,
                                          prog_ref_obs=obs['prog_ref'].data,
                                          obs_id=obs['obs_id'].data,
                                          pi=obs['pi'].data,
                                          partner=obs['partner'].data,
                                          band=obs['band'].data,
                                          too_status=obs['too_status'].data)

    # For now, program activation(prog_start) and
    # deactivation(prog_end) times are are set to the scheduling period
    # boundaries.
    # All programs are set to active.
    # These inputs can be changed once the information is available.
    progs = program_table.programtable(gemprgid=proginfo['prog_ref'].data,
                                       partner=proginfo['partner'].data,
                                       pi=proginfo['pi'].data,
                                       prog_time=proginfo['prog_time'].data,
                                       alloc_time=proginfo['alloc_time'].data,
                                       partner_time=proginfo['partner_time'].data,
                                       active=np.full(len(proginfo), True),
                                       prog_start=np.full(len(proginfo), start),
                                       prog_end=np.full(len(proginfo), end),
                                       too_status=proginfo['too_status'].data,
                                       scirank=proginfo['scirank'].data,
                                       observations=proginfo['obs'].data)

    # Add an additional column in the observation table to hold the
    # program table (progs) row index of each observation's
    # corresponding program.
    obs['i_prog'] = select_obs.i_progs(gemprgid=progs['gemprgid'].data,
                                       prog_ref=obs['prog_ref'].data)

    # ====== Instrument configuration calendar table ======
    # Create 'astropy.table.Table' (one row per night in plan period)
    # Store date, available instruments, GMOS-FPU, GMOS-Disperser, F2-FPU
    if verbose_progress:
        print('...instrument calendar')
    instcal = instrument_table(filename=instfile,
                               dates=timetable['date'].data)

    # ====== Timing windows and target calendar ======
    if verbose_progress:
        print('...target calendar')
    targetcal = timing_windows.get_timing_windows(site=site,
                                                  timetable=timetable,
                                                  moon=moon,
                                                  obs=obs,
                                                  progs=progs,
                                                  instcal=instcal)


    # -- Conditions tables --
    skycond = condition_table(size=len(timetable['utc'][0]), iq=iq, cc=cc, wv=wv)
    wind = wind_table(size=len(timetable['utc'][0]), direction=dir, velocity=vel, site_name=site.name)
    if verbose:
        print(skycond)
        print(wind)

    # ====== Get remaining available observations in nightly queue ======
    # target_cal[i_day] observations with remaining program time
    i_queue_cal = np.where(progs['prog_comp'].data[obs['i_prog'].data[targetcal[0]['i'].data]] < 1)[0][:]

    # target_cal[i_day] observations with remaining observation time
    i_queue_cal = i_queue_cal[np.where(obs['obs_comp'].data[targetcal[0]['i'].data[i_queue_cal]] < 1)[0][:]]

    # obs table rows of observations with remaining observation and program time
    i_queue_obs = targetcal[0]['i'].data[i_queue_cal]

    # all observations with remaining observation and program time in full queue.
    # Required for computing the distribution of remaining observation time.
    i_obs = np.where(progs['prog_comp'].data[obs['i_prog'].data] < 1)[0][:]
    i_obs = i_obs[np.where(obs['obs_comp'].data[i_obs] < 1)[0][:]]

    # ====== Compute observation time distribution (wra = right ascension weight) ======
    wra_all = weights.radist(ra=obs['ra'].quantity[i_obs],
                             tot_time=obs['tot_time'].quantity[i_obs],
                             obs_time=obs['obs_time'].quantity[i_obs])  # wra of all obs
    wra = wra_all[np.where([(i in i_obs) for i in i_queue_obs])[0][:]]  # wra of obs in tonight's queue

    # ====== tonight's queue from targetcal table ======
    # Create 'astropy.table.Table' for the active observations in tonight's queue
    targets = Table(targetcal[0][i_queue_cal])

    weightplotmode(site=site, timetable=timetable, sun=sun, moon=moon, obs=Table(obs[i_queue_obs]), progs=progs,
                   targets=targets, skycond=skycond, wind=wind, wra=wra)

    return


if __name__ == '__main__':
    wfpt()
