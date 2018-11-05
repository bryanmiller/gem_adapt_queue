#!/usr/bin/env python

# -- astroconda modules --
import argparse
from astroplan import download_IERS_A
import astropy.units as u
from astropy.table import Table, Column
from astropy.time import Time
import copy
import importlib
# from joblib import Parallel, delayed
import numpy as np
import os
import random
import textwrap

# -- Gemini modules --
from condition_table import checkdist
import convert_conditions
from convert_index import convindex
from dates import getdates
from dt import deltat
from events import generate_events
import make_plot
from observer_site import getsite
import printer
from sb import sb
import schedule
import select_obs
import timing_windows
import weights
# from wfpt import weightplotmode

# -- Data structure modules --
from condition_table import condition_table
from instrument_table import instrument_table
from moon_table import moon_table
from observation_table import observation_table
import program_table
from sun_table import sun_table
from time_table import time_table
from wind_table import wind_table

# # If astroplan download_IERS_A download link times out, try using one of these mirror URLs
# from astropy.utils import iers
# iers.conf.auto_download = False  # stop astroplan from reverting to default IERS_A link
# iers.IERS_A_URL = 'http://toshi.nofs.navy.mil/ser7/finals2000A.all'  # mirror URL
# iers.IERS_A_URL = 'https://datacenter.iers.org/eop/-/somos/5Rgv/latest/9'  # mirror URL

# # If Observer site registry URL is not accessible, try forcing download.
# from astropy.coordinates.earth import EarthLocation
# EarthLocation._get_site_registry(force_download=True)
# print(EarthLocation.get_site_names())

def takeSecond(row):
    return row[1]

parser = argparse.ArgumentParser(prog='gaqpt.py',
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=textwrap.dedent('''  
                                    Gemini Adaptive Queue Planning Tool
    *****************************************************************************************************               
        
        REQUIRED
        --------
        otfile                  OT catalog filename.
        
        prfile                  Program status filename.

        instcal                 Instrument calendar filename.
        
        
        OPTIONAL
        --------
        -s   --startdate        Start date 'YYYY-MM-DD' [DEFAULT=current].

        -e   --enddate          End date 'YYYY-MM-DD' [DEFAULT=startdate]. End date must be before
                                start date.  If no end date is provided, the scheduling period will
                                default to a single night.

        -dst --daylightsavings  Toggle daylight savings time [DEFAULT=False].
        
        -dt  --gridsize         Size of time-grid spacing [DEFAULT=0.1hr].
        
        -o   --observatory      Observatory site [DEFAULT='gemini_south']. Accepts the following:
                                1. 'gemini_north' (or 'MK' for Mauna Kea)
                                2. 'gemini_south' (or 'CP' for Cerro Pachon)
        
        -l   --logfile          Logfile name [DEFAULT='gaqptDDMMYY-hh:mm:ss.log'].
        
        -t  --toofile           Target of opportunity observation models filename [DEFAULT=None].
        
        -tp  --tooprob          Probability of incoming ToOs during the night [DEFAULT=0].
        
        -tm  --toomax           Maximum number of potential ToOs during the night [DEFAULT=4].
        
        -cp  --condprob         Probability of a sky conditions changing during the night [DEFAULT=0].
        
        -cm  --condmax          Maximum number of potential sky conditions changes during the night [DEFAULT=4].  
        
        -p   --plantype         Scheduling algorithm type [DEFAULT='Priority']. 
        
                                Conditions (if distribution=False):
        -iq  --iq               Image quality constraint [DEFAULT=70%].
        -cc  --cc               Cloud cover constraint   [DEFAULT=50%].
        -wv  --wv               Water vapor constraint   [DEFAULT=Any].
        
        -d   --distribution     Random viewing conditions from distribution [DEFAULT=False]. Accepts the following:
                                1. 'random' (or 'r').  Generate conditions from uniform distribution.
                                2. 'variant' (or 'v').  Randomly select one of several variants.
                                
                                Wind conditions:
        -dir --direction        Wind direction [DEFAULT=270deg].
        -vel --velocity         Wind velocity [DEFAULT=10deg].
        
        -rw  --randwind         Random wind conditions [DEFAULT=False]. 
                                Means and standard deviations at sites:
                                    Cerro Pachon : dir=(330 +/- 20)deg, vel=(5 +/- 3)m/s
                                    Mauna Kea    : dir=(330 +/- 20)deg, vel=(5 +/- 3)m/s                            

        -pp  --planplots        Show airmass plot of nightly plan [DEFAULT=False].
        
        -ip  --iterplots        Show airmass plot after each iteration of the plan (when simulating 
                                incoming ToO and changing sky conditions) [DEFAULT=False].
                                
        -bp  --buildupplots     Show airmass plot after each time an observation is added 
                                to the plan [DEFAULT=False].
        
        -sp  --skyplots         Show sky conditions plot [DEFAULT=False]. 
        
        -wp  --windplots        Show wind condition plot [DEFAULT=False]. 

        -u   --update           Download up-to-date IERS(International Earth Rotation and Reference Systems) data.

        -rs  --seed             Random seed number for random number generation [DEFAULT=1000].

        -v   --verbose          Print important variables [DEFAULT=False].

        -dg  --debug            Print additional outputs (intended for trouble-shooting) [DEFAULT=False].

    *****************************************************************************************************                        
                                    '''))

#       -we  --weightplot       Weight function plotting mode [DEFAULT=False].  After each plan is constructed,
#                               the user may select and view the weighting functions used to generate the
#                               plan.

parser.add_argument(action='store',
                    dest='otfile')

parser.add_argument(action='store',
                    dest='prfile')

parser.add_argument(action='store',
                    dest='instcal')

parser.add_argument('-s', '--startdate',
                    action='store',
                    default=None)

parser.add_argument('-e', '--enddate',
                    action='store',
                    default=None)

parser.add_argument('-dst', '--daylightsavings',
                    action='store_true',
                    dest='dst',
                    default=False)

parser.add_argument('-dt', '--gridsize',
                    action='store',
                    default=0.1)

parser.add_argument('-l', '--logfile',
                    action='store',
                    default=None)

parser.add_argument('-o', '--observatory',
                    action='store',
                    default='gemini_south')

parser.add_argument('-t', '--toofile',
                    action='store',
                    default=None)

parser.add_argument('-tp', '--tooprob',
                    action='store',
                    default=0)

parser.add_argument('-tm', '--toomax',
                    action='store',
                    default=4)

parser.add_argument('-cp', '--condprob',
                    action='store',
                    default=0)

parser.add_argument('-cm', '--condmax',
                    action='store',
                    default=4)

parser.add_argument('-p', '--plantype',
                    default='Priority')

parser.add_argument('-iq', '--iq',
                    default='70')

parser.add_argument('-cc', '--cc',
                    default='50')

parser.add_argument('-wv', '--wv',
                    default='Any')

parser.add_argument('-d', '--distribution',
                    default=None)

parser.add_argument('-dir', '--direction',
                    default=330)

parser.add_argument('-vel', '--velocity',
                    default=5)

parser.add_argument('-rw', '--randwind',
                    action='store_true',
                    default=False)

parser.add_argument('-pp', '--planplots',
                    action='store_true',
                    default=False)

parser.add_argument('-ip', '--iterplots',
                    action='store_true',
                    default=False)

parser.add_argument('-bp', '--buildupplots',
                    action='store_true',
                    default=False)

parser.add_argument('-sp', '--skyplots',
                    action='store_true',
                    default=False)

parser.add_argument('-wp', '--windplots',
                    action='store_true',
                    default=False)

parser.add_argument('-u', '--update',
                    action='store_true',
                    default=False)

parser.add_argument('-rs', '--seed',
                    dest='seed',
                    default=1000)

parser.add_argument('-v', '--verbose',
                    action='store_true',
                    default=False)

parser.add_argument('-dg', '--debug',
                    action='store_true',
                    default=False)

parse = parser.parse_args()

# Download updated International Earth Rotation and Reference Systems
# This will need to be done every few days.
if parse.update:
    download_IERS_A()

# -- input file names --
instfile = parse.instcal
otfile = parse.otfile
prfile = parse.prfile
too_file = parse.toofile

# -- check for input files --
cwdfiles = [f for f in os.listdir('.') if os.path.isfile(f)]
if otfile not in cwdfiles:
    raise ValueError(otfile + ' not found in current directory.')
if prfile not in cwdfiles:
    raise ValueError(prfile + ' not found in current directory.')
if instfile not in cwdfiles:
    raise ValueError(instfile + ' not found in current directory.')
if too_file is not None and too_file not in cwdfiles:
    raise ValueError(too_file + ' not found in current directory.')

# -- Schedule parameters --
dst = parse.dst
logfilename = parse.logfile

iq, cc, wv = convert_conditions.inputcond(parse.iq, parse.cc, parse.wv)
conddist = checkdist(parse.distribution)

dir = abs(float(parse.direction))
vel = abs(float(parse.velocity))
randwind = parse.randwind

showskyplots = parse.skyplots
showwindplots = parse.windplots
showplanplots = parse.planplots
showiterationplots = parse.iterplots
showplanbuildup = parse.buildupplots
seednum = parse.seed

# Time grid spacing size in hours
dt = float(parse.gridsize) * u.h
if dt <= 0*u.h:
    raise ValueError('Time grid spacing must be greater than 0.')

# Maximum number of incoming ToOs per night
too_max = int(parse.toomax)
if too_max < 0.:
    too_max = 0

# ToO probability
too_prob = round(float(parse.tooprob), 2)
if too_prob > 1.:
    too_prob = 1.
elif too_prob < 0.:
    too_prob = 0.

# Maximum number of possible sky condition changes per night
cond_change_max = int(parse.condmax)
if cond_change_max < 0.:
    cond_change_max = 0

# Probability of sky conditions change to occur
cond_change_prob = round(float(parse.condprob), 2)
if cond_change_prob > 1.:
    cond_change_prob = 1.
elif cond_change_prob < 0.:
    cond_change_prob = 0.

# Type of scheduling algorithm
plantype = parse.plantype
if plantype.lower() == 'priority':
    plantype = 'Priority'
else:
    raise ValueError('Plan type \'' + plantype + '\' not recognized.')

# ==============================================
#            Compute universal quantities
# ==============================================

verbose_progress = True  # print important variable/function names as they are completed
verbose = parse.verbose  # print important variables
verbose2 = parse.debug  # print additional outputs (intended for trouble-shooting)

# ====== Get Site Info ======
if verbose_progress:
    print('...observatory site, time_zone, utc_to_local')
# Create 'astroplan.Observer' object for observing site.
# Get time-zone name (for use by pytz) and utc_to_local time difference.
site, timezone_name, utc_to_local = getsite(site_name=parse.observatory,
                                            daylightsavings=dst)
if verbose or verbose2:
    print('\nSite', site)
    print('timezone_name', timezone_name)
    print('utc_to_local', utc_to_local)

# ====== Plan start/end dates ======
if verbose_progress:
    print('...scheduling period dates')
# Check format of command line input dates.
# Create 'astropy.time.core.Time' objects for start and end of plan period
start, end = getdates(startdate=parse.startdate,
                      enddate=parse.enddate,
                      utc_to_local=utc_to_local)
if verbose or verbose2:
    print('startdate, enddate', start, end)

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
if verbose or verbose2:
    print('\nTimetable:')
    print(timetable)

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
if verbose or verbose2:
    print('\nSun table:')
    print(sun)

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
if verbose or verbose2:
    print('\nMoon table:')
    print(moon)

# ====== Instrument configuration calendar table ======
if verbose_progress:
    print('...instrument calendar')
# Create 'astropy.table.Table' (one row per night in plan period)
# Store date, available instruments, GMOS-FPU, GMOS-Disperser, F2-FPU
instcal = instrument_table(filename=instfile,
                           dates=timetable['date'].data)
if verbose or verbose2:
    print('\nInstrument calendar:')
    print(instcal)

# ====== Assemble Observation Table ======
if verbose_progress:
    print('...observations')
# Create 'astropy.table.Table' (one observation per row)
obs = observation_table(filename=otfile, verbose = verbose)
if verbose or verbose2:
    print('\nObservation table:')
    print(obs)

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
if verbose or verbose2:
    print('\nProgram table:')
    print(progs)
    if verbose2:
        print('\nproginfo:')
        print(proginfo)

# ====== Timing windows and target calendar ======
if verbose_progress:
    print('...target calendar')
targetcal = timing_windows.get_timing_windows(site=site,
                                              timetable=timetable,
                                              moon=moon,
                                              obs=obs,
                                              progs=progs,
                                              instcal=instcal,
                                              verbose=verbose_progress,debug=verbose2)

obs_original = copy.deepcopy(obs)
progs_original = copy.deepcopy(progs)
targetcal_original = copy.deepcopy(targetcal)

while True:

    # ====== Log file name ======
    # Use custom log file name or make a time stamped file name
    if logfilename is not None:
        logfile = logfilename
    else:
        logfile = printer.def_filename('gqpt', 'log')

    # ==============================================
    #              Simulation menu
    # ==============================================
    print()
    print('\t---------------------------------------------------------------------')
    print('\t                  Gemini Adaptive Queue Planning Tool')
    print('\t---------------------------------------------------------------------')
    [print('\t' + line) for line in printer.parameters(dates=timetable['date'], site=site, daylightsavings=dst)]

    while True:
        fprint = '\t{0:10s}{1:45s}{2:40}'
        print('\n\tOptions:')
        print('\t--------')
        print(fprint.format('1.', 'Log file', logfile))
        print(fprint.format('2.', 'ToO file', str(too_file)))
        print(fprint.format('3.', 'ToO probability', str(too_prob)))
        print(fprint.format('4.', 'Max. number of ToOs per night', str(too_max)))
        print(fprint.format('5.', 'Conditions (iq,cc,wv)', '(' + iq + ', ' + cc + ', ' + wv + ')'))
        print(fprint.format('6.', 'Conditions from distribution type', str(conddist)))
        print(fprint.format('7.', 'Wind conditions (dir, vel)', '({}deg, {}m/s)'.format(dir, vel)))
        print(fprint.format('8.', 'Generate random wind conditions', str(randwind)))
        print(fprint.format('9.', 'Probability of condition change', str(cond_change_prob)))
        print(fprint.format('10.', 'Max. number of condition changes per night', str(cond_change_max)))
        print(fprint.format('11.', 'Show plan plots', str(showplanplots)))
        print(fprint.format('12.', 'Show airmass plot of each plan iteration', str(showiterationplots)))
        print(fprint.format('13.', 'Show airmass plots of plan building up', str(showplanbuildup)))
        print(fprint.format('14.', 'Show sky conditions plots', str(showskyplots)))
        print(fprint.format('15.', 'Show wind condition plots', str(showwindplots)))
        print()
        print(fprint.format('dir', 'Show files in current directory', ''))
        print(fprint.format('x,q', 'Exit', ''))

        userinput = input('\n Press enter to run or select an option: ')
        if userinput == '':
            break

        else:  # edit menu options
            while True:

                if userinput == '':
                    break

                elif userinput.lower() == 'x' or userinput.lower() == 'q':
                	exit()
                	
                elif userinput == 'dir':  # show files in curent directory
                    files = [f for f in os.listdir('.') if os.path.isfile(f)]
                    files.sort()
                    print('\n Current directory:\n -----------------')
                    for file in files:
                        print(' ' + file)

                elif userinput == '1':  #
                    templogfile = input(' Choose log file name (or \'d\' for DEFAULT): ')
                    if templogfile == '':
                        print(' No input received. No changes were made.')
                    elif templogfile.lower() == 'd':
                        logfile = printer.def_filename('gaqpt', 'log')
                        logfilename = None
                        print(' Log file name changed to ' + logfile)
                    else:
                        files = [f for f in os.listdir('.') if os.path.isfile(f)]
                        if templogfile in files:
                            userinput = input(' ' + templogfile + ' already exists. Overwrite? [y/n] ')
                            if userinput.lower() == 'y' or userinput == '':
                                logfile = templogfile
                            else:
                                print(' No changes were made.')
                        else:
                            logfile = templogfile
                            print(' Log file name changed to ' + logfile)

                elif userinput == '2':
                    tempfile = input(' Enter file name: ')
                    if tempfile == '':
                        print(' No input received. To remove the current ToO file use command \'None\'.')
                    elif tempfile.lower() == 'none':
                        too_file = None
                    else:
                        files = [f for f in os.listdir('.') if os.path.isfile(f)]
                        if tempfile in files:
                            too_file = tempfile
                            print(' File name changed to ' + tempfile)
                        else:
                            print('\n Current directory:\n -----------------')
                            # for file in files:
                            #     print(' ' + file)
                            print('\n File \'' + tempfile + '\' not found in current directory.')

                elif userinput == '3':
                    tempprob = input(' Enter a value between 0 and 1: ')
                    try:
                        tempprob = round(float(tempprob), 2)
                        if tempprob == '' or tempprob <= 0.:
                            too_prob = 0.
                        elif tempprob >= 1.:
                            too_prob = 1.
                        else:
                            too_prob = tempprob
                        print(' ToO probability set to ' + str(too_prob) + '.')
                    except ValueError:
                        print(' Recieved type ' + str(type(tempprob)) + '.  Expected a float between 0 and 1.')

                elif userinput == '4':
                    tempint = input(' Enter an integer: ')
                    try:
                        tempint = int(tempint)
                        if tempint == '' or tempint <= 0.:
                            too_max = 0.
                        else:
                            too_max = tempint
                        print(' Number of potential ToOs set to ' + str(too_max) + '.')
                    except ValueError:
                        print(' Recieved type ' + str(type(tempint)) + '.  Expected integer type.')

                elif userinput == '5':
                    condinput = input(' Input iq, cc, wv  percentiles separated by spaces (eg. \'20 50 Any\'): ')
                    tempconds = condinput.split(' ')
                    if len(tempconds) != 3:
                        print(' Did not receive 3 values. No changes were made.')
                    else:
                        try:
                            iq, cc, wv = convert_conditions.inputcond(iq=tempconds[0].strip("'"), cc=tempconds[1], wv=tempconds[2].strip("'"))
                            print(' New sky conditions: iq=' + iq + ', cc=' + cc + ', wv=' + wv + '')
                        except ValueError:
                            print(' ValueError: Could not set new conditions. Changes not made.')

                elif userinput == '6':
                    tempdist = input(' Input condition generator distribution type (or \'None\'): ')
                    try:
                        conddist = checkdist(tempdist)
                    except ValueError:
                        print(' No changes were made.')

                elif userinput == '7':
                    windinput = input(' Input new wind direction(deg), velocity(m/s) (eg. \'330 5\'): ')
                    tempwind = windinput.split(' ')
                    if len(tempwind) != 2:
                        print(' Did not receive 2 values. No changes were made.')
                        continue
                    else:
                        try:
                            wind = wind_table(size=len(timetable['utc'][0]), direction=float(tempwind[0]),
                                              velocity=float(tempwind[1]), site_name=site.name)
                            dir = tempwind[0]
                            vel = tempwind[1]
                        except ValueError:
                            print(' ValueError: Could not set new wind conditions.')
                            continue

                elif userinput == '8':
                    if randwind:
                        randwind = False
                        print(' Turned off random wind conditions.')
                    else:
                        randwind = True
                        print(' Turned on random wind conditions.')

                elif userinput == '9':
                    tempprob = input(' Enter a value between 0 and 1: ')
                    try:
                        tempprob = round(float(tempprob), 2)
                        if tempprob == '' or tempprob <= 0.:
                            cond_change_prob = 0.
                        elif tempprob >= 1.:
                            cond_change_prob = 1.
                        else:
                            cond_change_prob = tempprob
                        print(' Condition change probability set to ' + str(cond_change_prob) + '.')
                    except ValueError:
                        print(' Recieved type ' + str(type(tempprob)) + '.  Expected a float between 0 and 1.')

                elif userinput == '10':
                    tempint = input(' Enter an integer: ')
                    try:
                        tempint = int(tempint)
                        if tempint == '' or tempint <= 0.:
                            cond_change_max = 0.
                        else:
                            cond_change_max = tempint
                        print(' Number of potential condition changes set to ' + str(cond_change_max) + '.')
                    except ValueError:
                        print(' Recieved type ' + str(type(tempint)) + '.  Expected integer type.')

                elif userinput == '11':
                    if showplanplots:
                        showplanplots = False
                        print(' Turned off air mass plots.')
                    else:
                        showplanplots = True
                        print(' Turned on air mass plots.')

                elif userinput == '12':
                    if showiterationplots:
                        showiterationplots = False
                        print(' Turned off plan iteration airmass plots.')
                    else:
                        showiterationplots = True
                        print(' Turned on plan iteration airmass plots.')

                elif userinput == '13':
                    if showplanbuildup:
                        showplanbuildup = False
                        print(' Turned off airmass build-up plots.')
                    else:
                        showplanbuildup = True
                        print(' Turned on airmass build-up plots.')

                elif userinput == '14':
                    if showskyplots:
                        showskyplots = False
                        print(' Turned off viewing conditions plots.')
                    else:
                        showskyplots = True
                        print(' Turned on viewing conditions plots.')

                elif userinput == '15':
                    if showwindplots:
                        showwindplots = False
                        print(' Turned off wind condition plots.')
                    else:
                        showwindplots = True
                        print(' Turned on wind condition plots.')

                else:
                    print(' Did not recognize input. No changes made.')

                userinput = input(' Press enter to return to menu or select another option: ')

    # ====== Re-load libraries ======
    importlib.reload(schedule)
    importlib.reload(make_plot)
    importlib.reload(select_obs)
    importlib.reload(printer)
    importlib.reload(weights)
    importlib.reload(timing_windows)

    # ====== Reset random number seed ======
    random.seed(seednum)

    # ====== initialize simulation stats ======
    sim_stats = {'tot_time': 0. * u.h, 'used_time': 0. * u.h}
    obs = copy.deepcopy(obs_original)
    progs = copy.deepcopy(progs_original)
    targetcal = copy.deepcopy(targetcal_original)

    # ====== Record inputs and parameters in log file ======
    printer.overwrite_log(filename=logfile,
                          lines=printer.programinfo(progname=__file__,
                                                    cwd=os.getcwd()))
    printer.append_to_file(filename=logfile,
                           lines=printer.inputfiles(otfile=otfile,
                                                    instfile=instfile,
                                                    toofile=too_file,
                                                    prfile=prfile))

    printer.append_to_file(filename=logfile,
                           lines=printer.planoptions(too_prob=too_prob,
                                                     too_max=too_max,
                                                     cond_change_prob=cond_change_prob,
                                                     cond_change_max=cond_change_max,
                                                     iq=iq,
                                                     cc=cc,
                                                     wv=wv,
                                                     conddist=conddist,
                                                     direction=dir,
                                                     velocity=vel,
                                                     random=randwind))

    printer.append_to_file(filename=logfile,
                           lines=printer.parameters(dates=timetable['date'],
                                                    site=site,
                                                    daylightsavings=dst))

    # Record status of programs in log file
    printer.append_to_file(filename=logfile,
                           lines=printer.queuestatus(progs=progs,
                                                     simstats=sim_stats,
                                                     description='Initial queue status'))

    # ====== ToO observation models =======
    toonum = 0  # count ToOs
    if too_file is not None:
        too_models = observation_table(filename=too_file)

    # ==============================================
    #              Begin Scheduling
    # ==============================================

    # -- Cycle through nights --
    for i_day in np.arange(len(timetable['date'])):

        # ====== Generate sky and wind condition tables ======
        # Create one 'astropy.table.Table' for each.
        # Rows in tables corresponding to times in time grid for current night.
        skycond = condition_table(size=len(timetable['utc'][i_day]),
                                  iq=iq,
                                  cc=cc,
                                  wv=wv,
                                  conddist=conddist)

        wind = wind_table(size=len(timetable['utc'][i_day]),
                          direction=dir,
                          velocity=vel,
                          site_name=site.name,
                          random=randwind)

        if verbose:
            print('\nSky conditions:\n', skycond)
            print('\nWind conditions:\n', wind)

        # ====== Generate ToO and sky condition change events ======
        events = []  # list of event type and indices in plan that they will occur.

        if too_file is not None:
            too_events = generate_events(grid_size=len(timetable[i_day]['utc']),
                                         event_type='Target of Opportunity',
                                         probability=too_prob,
                                         event_max=too_max)  # generate ToO events
            events.extend(too_events)

        conditions_events = generate_events(grid_size=len(timetable[i_day]['utc']),
                                            event_type='Condition change',
                                            probability=cond_change_prob,
                                            event_max=cond_change_max)  # generate condition changes
        events.extend(conditions_events)

        # sort events into order of sequence and store in a table.
        events.sort(key=takeSecond)
        if len(events) != 0:
            events = Table(rows=events, names=('type', 'i'), dtype=(str, int))
        else:
            events = Table()
        n_events = len(events)
        event_num = 0  # iterate through events in list
        if verbose:
            pass
            print('\nevents')
            print(events)

        # ====== Print plan parameters/conditions to console window ======
        [print('\t' + line)
         for line in ['\n', '\t\t\t-- Generating plan for night of ' + timetable['date'][i_day] + ' --']]

        [print('\t' + line)
         for line in printer.skyinfo(iq=iq,
                                     cc=cc,
                                     wv=wv,
                                     conddist=conddist,
                                     skycond=skycond)]
        [print('\t' + line)
         for line in printer.windinfo(dir=wind['dir'].quantity[0],
                                      vel=wind['vel'].quantity[0])]
        [print('\t' + line)
         for line in printer.timeinfo(solar_midnight=timetable['solar_midnight'][i_day],
                                      utc_to_local=utc_to_local)]
        [print('\t' + line)
         for line in printer.suninfo(ra=sun['ra'].quantity[i_day],
                                     dec=sun['dec'].quantity[i_day])]
        [print('\t' + line)
         for line in printer.mooninfo(ra=moon['ra_mid'].quantity[i_day],
                                      dec=moon['dec_mid'].quantity[i_day],
                                      frac=moon['fraction'].quantity[i_day],
                                      phase=moon['phase'].quantity[i_day])]

        # ====== Print plan parameters/conditions to log file ======
        printer.append_to_file(filename=logfile,
                               lines=['\n',
                                      '\t\t\t-- Generating plan for night of ' + timetable['date'][i_day] + ' --'])
        printer.append_to_file(filename=logfile,
                               lines=printer.skyinfo(iq=iq,
                                                     cc=cc,
                                                     wv=wv,
                                                     conddist=conddist,
                                                     skycond=skycond))
        printer.append_to_file(filename=logfile,
                               lines=printer.windinfo(dir=wind['dir'].quantity[0],
                                                      vel=wind['vel'].quantity[0]))
        printer.append_to_file(filename=logfile,
                               lines=printer.timeinfo(solar_midnight=timetable['solar_midnight'][i_day],
                                                      utc_to_local=utc_to_local))
        printer.append_to_file(filename=logfile,
                               lines=printer.suninfo(ra=sun['ra'].quantity[i_day],
                                                     dec=sun['dec'].quantity[i_day]))
        printer.append_to_file(filename=logfile,
                               lines=printer.mooninfo(ra=moon['ra_mid'].quantity[i_day],
                                                      dec=moon['dec_mid'].quantity[i_day],
                                                      frac=moon['fraction'].quantity[i_day],
                                                      phase=moon['phase'].quantity[i_day]))

        # ====== Get remaining available observations in nightly queue ======
        # target_cal[i_day] observations with remaining program time
        i_queue_cal = np.where(progs['prog_comp'].data[obs['i_prog'].data[targetcal[i_day]['i'].data]] < 1)[0][:]

        # target_cal[i_day] observations with remaining observation time
        i_queue_cal = i_queue_cal[np.where(obs['obs_comp'].data[targetcal[i_day]['i'].data[i_queue_cal]] < 1)[0][:]]

        # obs table rows of observations with remaining observation and program time
        i_queue_obs = targetcal[i_day]['i'].data[i_queue_cal]

        # all observations with remaining observation and program time in full queue.
        # Required for computing the distribution of remaining observation time.
        i_obs = np.where(progs['prog_comp'].data[obs['i_prog'].data] < 1)[0][:]
        i_obs = i_obs[np.where(obs['obs_comp'].data[i_obs] < 1)[0][:]]
        i_obs_current = 0

        # ==============================================
        #           Generate Plan for Tonight
        # ==============================================

        # ====== Initialize plan parameters ======
        plan = np.full(len(timetable['utc'][i_day]), -1)  # Empty plan
        s1 = 0  # Start index of current schedule section.
        s2 = len(plan)  # End index of current schedule section.
        itnum = 1  # Plan iteration number

        ii = np.where(plan == -1)[0][:]  # unscheduled time slots
        while len(ii) != 0:

            # ====== Compute observation time distribution (wra = right ascension weight) ======
            wra_all = weights.radist(ra=obs['ra'].quantity[i_obs],
                                     tot_time=obs['tot_time'].quantity[i_obs],
                                     obs_time=obs['obs_time'].quantity[i_obs], verbose=verbose)  # wra of all obs
            wra = wra_all[np.where([(i in i_obs) for i in i_queue_obs])[0][:]]  # wra of obs in tonight's queue

            if verbose2:
                print('Tonight\'s queue (i_queue):', i_queue_obs)
                print('RA weights (wra):', wra)

            # ====== tonight's queue from targetcal table ======
            # Create 'astropy.table.Table' for the active observations in tonight's queue
            targets = Table(targetcal[i_day][i_queue_cal])

            # ====== Compute visible sky brightnesses at targets ======
            targets['vsb'] = Column([sb(mpa=moon['phase'].quantity[i_day],
                                        mdist=targets['mdist'].quantity[i],
                                        mZD=moon['ZD'].data[i_day] * u.deg,
                                        ZD=targets['ZD'].quantity[i],
                                        sZD=sun['ZD'].data[i_day] * u.deg,
                                        cc=skycond['cc'].data)
                                     for i in range(len(targets))])

            # ====== Convert vsb to sky background percentiles ======
            targets['bg'] = Column([convert_conditions.sb_to_cond(sb=targets['vsb'][i]) for i in range(len(targets))])

            # ====== Compute weights ======
            targets['weight'] = Column(
                [weights.obsweight(
                    obs_id=obs['obs_id'][i_queue_obs[i]],
                    ra=obs['ra'].quantity[i_queue_obs[i]],
                    dec=obs['dec'].quantity[i_queue_obs[i]],
                    iq=obs['iq'].data[i_queue_obs[i]],
                    cc=obs['cc'].data[i_queue_obs[i]],
                    bg=obs['bg'].data[i_queue_obs[i]],
                    wv=obs['wv'].data[i_queue_obs[i]],
                    elev_const=obs['elev_const'][i_queue_obs[i]],
                    i_wins=targets['i_wins'][i],
                    band=obs['band'].data[i_queue_obs[i]],
                    user_prior=obs['user_prior'][i_queue_obs[i]],
                    AM=targets['AM'].data[i],
                    HA=targets['HA'].data[i] * u.hourangle,
                    AZ=targets['AZ'].quantity[i],
                    latitude=site.location.lat,
                    prog_comp=progs['prog_comp'].data[obs['i_prog'][i_queue_obs[i]]],
                    obs_comp=obs['obs_comp'].quantity[i_queue_obs[i]],
                    skyiq=skycond['iq'].data,
                    skycc=skycond['cc'].data,
                    skywv=skycond['wv'].data,
                    skybg=targets['bg'].data[i],
                    winddir=wind['dir'].quantity,
                    windvel=wind['vel'].quantity,
                    wra=wra[i], verbose = verbose, debug=verbose2)
                    for i in range(len(targets))])

            # ====== Schedule remaining spaces in plan ======
            if verbose:
                print('current plan', plan)

            plan_temp = copy.deepcopy(plan)  # make copy of current plan before completing next iteration
            ii = np.where(plan == -1)[0][:]
            obs_temp = Table(obs[i_queue_obs])  # copy observation table before running scheduling algorthm

            # ====== Complete current plan iteration ======
            # Fill nightly plan one observation at a time.
            while len(ii) != 0:

                # add an observation to the plan
                if plantype == 'Priority':
                    plan_temp = schedule.priority(plan=plan_temp,
                                                  obs=obs_temp,
                                                  targets=targets,
                                                  dt=deltat(time_strings=timetable['local'][0][0:2]))

                if showplanbuildup:
                    make_plot.airmass(plan=plan_temp,
                                      obs_id=targets['id'].data,
                                      am=targets['AM'].data,
                                      date=timetable[i_day]['date'],
                                      local=timetable['local'].data[i_day],
                                      moonam=moon['AM'].data[i_day],
                                      description=' (build-up)',
                                      obslabels=True)

                ii = np.where(plan_temp == -1)[0][:]

            # ====== Optimize current section of plan (from s1 to end of plan) ======
            plan_opt = schedule.optimize(plan=plan_temp,
                                         targets=targets,
                                         jj=np.arange(s1, len(plan)))
            if verbose:
                print('new plan', plan_opt)

            # ==============================================
            #       ToO or viewing conditions change
            # ==============================================
            addevent = False
            # Check for future ToO or condition change events occuring between
            # current point in night (s1) and end of night
            if len(events) != 0 and event_num < len(events) and (s1 <= events['i'].data[event_num]):
                addevent = True

                # ====== Print current plan iteration ======
                [print('\t' + line)
                 for line in printer.plantable(plan=plan_opt,
                                               targets=targets,
                                               obs=Table(obs['obs_id', 'inst', 'target', 'obs_comp', 'ra', 'dec']
                                                         [targets['i'].data]),
                                               timetable=Table(timetable['date', 'utc', 'local', 'lst'][i_day]),
                                               description='(iteration ' + str(itnum) + ') '
                                               )
                 ]

                # ====== Write plan iteration to log file ======
                printer.append_to_file(
                    filename=logfile,
                    lines=printer.plantable(plan=plan_opt,
                                            targets=targets,
                                            timetable=Table(timetable['date', 'utc', 'local', 'lst'][i_day]),
                                            obs=Table(obs['obs_id', 'inst', 'target', 'obs_comp', 'ra', 'dec']
                                                      [targets['i'].data]),
                                            description='(iteration ' + str(itnum) + ') '
                                            )
                )

                # ====== Air mass plot ======
                if showiterationplots:
                    make_plot.airmass(plan=plan_opt,
                                      obs_id=targets['id'].data,
                                      am=targets['AM'].data,
                                      date=timetable[i_day]['date'],
                                      local=timetable['local'].data[i_day],
                                      moonam=moon['AM'].data[i_day],
                                      description=' (iteration ' + str(itnum) + ')'
                                      )

                # print('s1, s2', s1, s2)
                # print(events['i'].data)
                # print(s1 <= events['i'].data)
                # print(events['i'].data < s2)
                # print(np.logical_and(s1 <= events['i'].data, events['i'].data < s2))

                k1 = s1  # index of current progress through night
                k2 = s2  # index of next plan interruption

                # Handle ToO or Sky condition event before making next plan iteration.
                # If next event is a Rapid ToO, generate a new plan immediately.
                # If next event is a Standard ToO, generate a new plan at the end of the current observation or at
                # the next event that interrupts the plan. Whichever occurs first.
                # Similarly, if next event is a change of sky condition, generate a new plan where appropriate if the
                # conditions worsen and the current observation can't be finished.  Otherwise, generate a new plan at
                # the end of the current observation or the next event that interrupts the plan. Whichever occurs first.

                # while any remaining events occur before next plan interruption...
                while s1 <= events['i'].data[event_num] <= s2:

                    next_event = events[event_num]
                    i_event = next_event['i'].data[0]  # time grid index when event occurs

                    i_target_current = plan_opt[i_event]  # target table index of current obs
                    if i_target_current != -2:
                        i_obs_current = targets['i'].data[i_target_current]  # obs table index of current obs
                        # print('i_obs_current', i_obs_current)

                    # print('i_target_current', i_target_current)
                    # print('k1', k1)
                    # print('nextevent')
                    # print(next_event)
                    # print('i_event', i_event)

                    # ====== Remaining observation boundaries ======
                    k1 = i_event  # time grid when event occurs
                    k2 = i_event + 1  # end of current observation
                    if i_target_current == -2:  # if no observation at current time. set plan to be interrupted.
                        s2 = k2
                    else:
                        while True:  # otherwise, get end of current observation
                            if k2 <= len(plan_temp) - 2:
                                if plan_opt[k2 + 1] == i_target_current:
                                    k2 = k2 + 1
                                else:
                                    break
                            else:
                                break

                    # print('plan_opt', plan_opt)
                    # print('k1, k2', k1, k2)

                    if next_event[0] == 'Target of Opportunity':

                        toonum = toonum + 1  # increment too observation number
                        i_too_model = np.random.randint(len(too_models))  # select random ToO type from file

                        # Observation table of new ToO
                        new_too = Table(too_models[i_too_model])

                        # assign observation identifier
                        new_too['obs_id'][0] = new_too['prog_ref'][0] + '-' + str(toonum)

                        # find corresponding program in table
                        new_too['i_prog'] = select_obs.i_progs(gemprgid=progs['gemprgid'].data,
                                                               prog_ref=new_too['prog_ref'].data)

                        # append ToO to bottom of observation table
                        obs.add_row(new_too[0])

                        # add observation identifier to program table
                        progs[new_too['i_prog'].data[0]]['observations'].append(new_too['obs_id'][0])

                        # print event info to console window
                        [print('\t' + line)
                         for line in printer.too_event(type=new_too[0]['user_prior'],
                                                       time=Time(timetable['local'].data[i_day][i_event])
                                                       )
                         ]

                        # print event info to log file
                        printer.append_to_file(filename=logfile,
                                               lines=printer.too_event(type=new_too[0]['user_prior'],
                                                                       time=Time(timetable['local']
                                                                                 .data[i_day][i_event])
                                                                       )
                                               )

                        # generate target tables for scheduling period
                        toocal = timing_windows.get_timing_windows(site=site,
                                                                   timetable=timetable,
                                                                   moon=moon,
                                                                   obs=new_too,
                                                                   progs=progs,
                                                                   instcal=instcal,
                                                                   current_time=Time(timetable['utc'].data
                                                                                     [i_day][i_event]))

                        # print('i_event', i_event)
                        # print(obs[-1])

                        # append ToO target tables to existing target tables for remainder of scheduling period.
                        for i in range(i_day, len(targetcal)):
                            # print(targetcal[i].colnames)
                            # print(toocal[i].colnames)
                            # print('too window', toocal[i]['i_wins'].data)
                            if len(toocal[i]) != 0:
                                toocal[i]['i'][0] = len(obs) - 1  # assign row index at end of each target table
                                targetcal[i].add_row(toocal[i][0])  # add to target table

                        # Append index number to current queue lists (observation table subset and target table subset)
                        # print(i_queue_cal)
                        # print(len(targets))
                        i_queue_cal = np.append(i_queue_cal, [len(targetcal[i_day]) - 1])
                        i_queue_obs = np.append(i_queue_obs, [len(obs)-1])
                        i_obs = np.append(i_obs, [len(obs)-1])
                        # print(i_queue_cal)

                        if len(toocal[i_day]) != 0:
                            # print(obs[-1]['user_prior'])

                            # If currently observing interrupt ToO, do not interrupt plan again
                            if 'Interrupt' in obs[i_obs_current]['user_prior']:
                                print('Could not schedule. Currently observing Interrupt type ToO.')
                                printer.append_to_file(filename=logfile,
                                                       lines=['Could not schedule. '
                                                              'Currently observing Interrupt type ToO.'])

                            # If Interrupt type ToO, whether or not to interrupt plan immediately.
                            elif 'Interrupt' in obs[-1]['user_prior']:

                                # print(obs['tot_time'].quantity[-1], dt)
                                # print(obs['tot_time'].quantity[-1] / dt)
                                # print((obs['tot_time'].quantity[-1] / dt).round())
                                # print(int((obs['tot_time'].quantity[-1] / dt).round()))
                                # print(toocal[i_day]['i_wins'].data[0][0][0])
                                # print(toocal[i_day]['i_wins'].data[0][0][1])

                                # number of grid spaces required to complete observation
                                nt = int((obs['tot_time'].quantity[-1] / dt).round())

                                # if ToO time window is long enough to perform whole observation...
                                if toocal[i_day]['i_wins'].data[0][0][1] - toocal[i_day]['i_wins'].data[0][0][0] >=\
                                        nt - 1:
                                    # If current iq, cc, wv are good enough to observe,
                                    # compute and check that all weights are positive for remainder of
                                    # observation window.
                                    if skycond['iq'].data[i_event] <= obs['iq'].data[-1] \
                                            or skycond['cc'].data[i_event] <= obs['cc'].data[-1] \
                                            or skycond['wv'].data[i_event] <= obs['wv'].data[-1]:

                                        temp_vsb = sb(mpa=moon['phase'].quantity[i_day],
                                                      mdist=targets['mdist'].quantity[-1],
                                                      mZD=moon['ZD'].data[i_day] * u.deg,
                                                      ZD=targets['ZD'].quantity[-1],
                                                      sZD=sun['ZD'].data[i_day] * u.deg,
                                                      cc=skycond['cc'].data)

                                        temp_bg = convert_conditions.sb_to_cond(sb=temp_vsb)

                                        temp_weights = weights.obsweight(
                                            obs_id=obs['obs_id'][-1],
                                            ra=obs['ra'].quantity[-1],
                                            dec=obs['dec'].quantity[-1],
                                            iq=obs['iq'].data[-1],
                                            cc=obs['cc'].data[-1],
                                            bg=obs['bg'].data[-1],
                                            wv=obs['wv'].data[-1],
                                            elev_const=obs['elev_const'][-1],
                                            i_wins=toocal[i_day]['i_wins'][0],
                                            band=obs['band'].data[-1],
                                            user_prior=obs['user_prior'][-1],
                                            AM=toocal[i_day]['AM'].data[0],
                                            HA=toocal[i_day]['HA'].data[0] * u.hourangle,
                                            AZ=toocal[i_day]['AZ'].quantity[0],
                                            latitude=site.location.lat,
                                            prog_comp=progs['prog_comp'].data[obs['i_prog'][-1]],
                                            obs_comp=obs['obs_comp'].quantity[-1],
                                            skyiq=skycond['iq'].data,
                                            skycc=skycond['cc'].data,
                                            skywv=skycond['wv'].data,
                                            skybg=temp_bg,
                                            winddir=wind['dir'].quantity,
                                            windvel=wind['vel'].quantity,
                                            wra=1)


                                        # print('i_event, nt', i_event, nt)
                                        # print(temp_weights)
                                        # print(temp_weights[i_event+1:i_event+nt+2])

                                        # check if observation can be started at the time grid index and
                                        # completed for the current conditions
                                        ii_bad_weights = np.where(temp_weights[i_event+1:i_event+nt+2] == 0)[0][:]
                                        # print('temp_bg', temp_bg)
                                        # print('ii_bad_weights', ii_bad_weights)
                                        if len(ii_bad_weights) == 0:
                                            k2 = k1 + 1
                                        else:
                                            print('\tCould not schedule ToO.')
                                            printer.append_to_file(filename=logfile,
                                                                   lines=['Could not schedule.'])




                        # print(obs[-1])
                        # print(toocal[i_day][-1])

                    elif next_event[0] == 'Condition change':
                        # generate new sky conditions at random
                        new_skycond = condition_table(size=len(timetable['utc'][i_day]),
                                                      iq=iq,
                                                      cc=cc,
                                                      wv=wv,
                                                      conddist='random')

                        # Replace sky conditions in table from time of event to end of night
                        skycond['iq'].data[i_event:] = new_skycond['iq'].data[i_event:]
                        skycond['cc'].data[i_event:] = new_skycond['cc'].data[i_event:]
                        skycond['wv'].data[i_event:] = new_skycond['wv'].data[i_event:]

                        # print event info to console window
                        [print('\t' + line)
                         for line in printer.conditions_event(iq=skycond['iq'][i_event],
                                                              cc=skycond['cc'][i_event],
                                                              wv=skycond['wv'][i_event],
                                                              time=Time(timetable['local'].data[i_day][i_event])
                                                              )
                         ]
                        # print event info to log file
                        printer.append_to_file(filename=logfile,
                                               lines=printer.conditions_event(iq=skycond['iq'][i_event],
                                                                              cc=skycond['cc'][i_event],
                                                                              wv=skycond['wv'][i_event],
                                                                              time=Time(timetable['local']
                                                                                        .data[i_day][i_event])
                                                                              )
                                               )

                        # If event occurs part way through an observation. Determine whether or not
                        # to interrupt the observation to generate a new plan.
                        # if plan is set to be interrupted at next time grid spacing, this section should be skipped.
                        if k2 - k1 > 1:

                            # print(obs['bg'].data[i_obs_current])
                            # print(moon['phase'].quantity[i_day])
                            # print(targets['mdist'].quantity[i_target_current][k1:k2 + 1])
                            # print(moon['ZD'].data[i_day][k1:k2 + 1] * u.deg)
                            # print(targets['ZD'].quantity[i_target_current][k1:k2 + 1])
                            # print(sun['ZD'].data[i_day][k1:k2 + 1] * u.deg)
                            # print(skycond['cc'].data[k1:k2 + 1])

                            # If current obs can be continued for new IQ, CC, and WV, re-compute and check BG
                            # otherwise, interrupt plan immediately.
                            if new_skycond['iq'].data[i_event] > obs['iq'].data[i_obs_current] \
                                    or new_skycond['cc'].data[i_event] > obs['cc'].data[i_obs_current] \
                                    or new_skycond['wv'].data[i_event] > obs['wv'].data[i_obs_current]:
                                # print('bad sky conditions')
                                k2 = k1 + 1
                            else:
                                i_target = None
                                temp_vsb = sb(mpa=moon['phase'].quantity[i_day],
                                              mdist=targets['mdist'].quantity[i_target_current][k1+1:k2+1],
                                              mZD=moon['ZD'].data[i_day][k1+1:k2+1] * u.deg,
                                              ZD=targets['ZD'].quantity[i_target_current][k1+1:k2+1],
                                              sZD=sun['ZD'].data[i_day][k1+1:k2+1] * u.deg,
                                              cc=skycond['cc'].data[k1+1:k2+1])
                                temp_bg = convert_conditions.sb_to_cond(sb=temp_vsb)
                                ii_bad_bg = np.where(temp_bg > obs['bg'].data[i_obs_current])[0][:]

                                # print('temp_bg', temp_bg)
                                # print('ii_bad_bg', ii_bad_bg)

                                # If observation can be finished, set end of current plan iteration (s2) to the end of
                                # the observation (k1).  Otherwise, set the plan to be interrupted when the conditions
                                # no longer satisfy the observation constraints.
                                if len(ii_bad_bg) != 0:
                                    k2 = ii_bad_bg[0] + k1 + 1

                    s2 = k2  # set next interruption of plan
                    # print('s1, s2', s1, s2)
                    event_num = event_num + 1
                    if event_num == len(events):
                        break

            # ====== Finalize plan preceding ToO/condition event ======
            plan[s1:s2+1] = plan_opt[s1:s2+1]
            # print(plan)

            # ====== Update observation and program times ======
            i_plan = convindex(plan=plan, i_obs=targets['i'].data)  # 'obs' rows corresponding to plan indices

            schedule.update_obs_progs(plan=i_plan[s1:s2+1],
                                      obs=obs,
                                      progs=progs,
                                      dt=deltat(time_strings=timetable['utc'][0][0:2]))

            # ====== Prepare next iteration of plan ======
            # Adjust plan parameters for next iteration of scheduling.
            s1 = s2 + 1
            s2 = len(plan) - 1
            itnum = itnum + 1
            ii = np.where(plan == -1)[0][:]

            if s1 == len(plan) - 1:  # break loop if current iteration starts at end of plan.
                break

        # ==============================================
        #                Plan Results
        # ==============================================

        # ====== Print plan to console window ======
        [print('\t' + line)
         for line in printer.plantable(plan=plan,
                                       targets=targets,
                                       obs=Table(obs['obs_id', 'inst', 'target', 'obs_comp', 'ra', 'dec']
                                                 [targets['i'].data]),
                                       timetable=Table(timetable['date', 'utc', 'local', 'lst'][i_day])
                                       )
         ]

        # ====== Write plan to log file ======
        printer.append_to_file(
            filename=logfile,
            lines=printer.plantable(plan=plan,
                                    targets=targets,
                                    timetable=Table(timetable['date', 'utc', 'local', 'lst'][i_day]),
                                    obs=Table(obs['obs_id', 'inst', 'target', 'obs_comp', 'ra', 'dec']
                                              [targets['i'].data])
                                    )
        )

        if showplanplots:
            # ====== Air mass plot ======
            make_plot.airmass(plan=plan,
                              obs_id=targets['id'].data,
                              am=targets['AM'].data,
                              date=timetable[i_day]['date'],
                              local=timetable['local'].data[i_day],
                              moonam=moon['AM'].data[i_day],
                              description=''
                              )

            # ====== Altitude azimuth plot ======
            make_plot.altaz(date=timetable['date'][i_day],
                            plan=plan,
                            obs_id=targets['id'].data,
                            az=targets['AZ'].quantity,
                            zd=targets['ZD'].quantity,
                            moonaz=moon['AZ'].data[i_day] * u.rad,
                            moonzd=moon['ZD'].data[i_day] * u.deg)

        if showskyplots:
            make_plot.skyconditions(skycond=skycond,
                                    local_time=timetable['local'][i_day],
                                    date=timetable['date'][i_day])

        if showwindplots:
            make_plot.windconditions(wind=wind,
                                     local_time=timetable['local'][i_day],
                                     date=timetable['date'][i_day])

        # -- Update simulation stats --
        sim_stats = schedule.nightstats(stats=sim_stats,
                                        plan=plan,
                                        timetable=Table(timetable[i_day]))

    # -- Write simulation results to log file --
    printer.append_to_file(filename=logfile,
                           lines=printer.queuestatus(progs=progs,
                                                     simstats=sim_stats,
                                                     description='Final queue status'))

    print('\nSimulation complete!\n')
