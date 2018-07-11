ttimer = False
if ttimer: import time as t
if ttimer: timer = t.time()

# astroconda packages
import os
import re
import copy
import random
import textwrap
import argparse
import importlib
import numpy as np
import astropy.units as u
from joblib import Parallel, delayed
from astropy import (coordinates, time)
from astroplan import (download_IERS_A, Observer)

# gqpt packages
import logger
import select_obs
from sb import sb
import weightplotmode
from gcirc import gcirc
from amplot import amplot
from condgen import condgen
from printplan import printPlanTable
from gemini_schedulers import PlanInfo
from timing_windows import timingwindows
from conversions import actual_conditions
from gemini_instruments import getinstruments
from gemini_observations import Gobservations, Gcatfile
from calc_weight import (calc_weight, weight_ra, getprstatus)
from gemini_classes import (TimeInfo, SunInfo, MoonInfo, TargetInfo)

if ttimer: print('\n\tTimer imports = ', t.time() - timer)

#   ======================================= Read and command line inputs ===============================================

parser = argparse.ArgumentParser(prog='gqpt.py',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description = textwrap.dedent('''                                                             
                                                        Guide
            *****************************************************************************************************               

                otfile                  OT catalog filename.
                
                instcal                 Instrument calendar filename.

                -o   --observatory      Observatory site [DEFAULT='gemini_south']. Accepts the following:
                                        1. 'gemini_north' (or 'MK' for Mauna Kea)
                                        2. 'gemini_south' (or 'CP' for Cerro Pachon)
                
                -s   --startdate        Start date 'YYYY-MM-DD' [DEFAULT=tonight].
                
                -e   --enddate          End date 'YYYY-MM-DD' [DEFAULT=startdate]. End date must be before
                                        the start date.  If no enddate is provided, a single night will be
                                        scheduled.

                -dst --daylightsavings  Toggle daylight savings time [DEFAULT=False].
                                        
                                        Conditions (if distribution=False):
                -i   --iq               Image quality constraint [DEFAULT=70]
                -c   --cc               Cloud cover constraint   [DEFAULT=50]
                -w   --wv               Water vapor constraint   [DEFAULT=Any]

                -d   --distribution     Generate conditions from distribution [DEFAULT=False]. Accepts the following:
                                        1. 'gaussian' (or 'g')  
                
                -b   --buildplots       Create and save airmass plots at stages of plan construction [DEFAULT=False].
                
                -a   --amplots          Create and save airmass plot of plan [DEFAULT=False].
                
                -wp  --weightplot       Weight function plotting mode [DEFAULT=False].  After plan is constructed, the 
                                        user may select and view weighting functions for observations in queue.
                
                -u   --update           Download up-to-date IERS(International Earth Rotation and Reference Systems).
                   
            *****************************************************************************************************                        
                                    '''))

parser.add_argument(action='store',
                    dest='otfile')

parser.add_argument(action='store',
                    dest='instcal')

parser.add_argument('-l','--logfile',
                    action='store',
                    default=None)

parser.add_argument('-o','--observatory',
                    action='store',
                    default='gemini_south')

parser.add_argument('-s','--startdate',
                    action='store',
                    default=None)

parser.add_argument('-e','--enddate',
                    action='store',
                    default=None)

parser.add_argument('-dst','--daylightsavings',
                    action='store_true',
                    dest='dst',
                    default=False)

parser.add_argument('-iq', '--iq',
                    default='70')

parser.add_argument('-cc', '--cc',
                    default='50')

parser.add_argument('-wv', '--wv',
                    default='Any')

parser.add_argument('-d', '--distribution',
                    default=None)

parser.add_argument('-b', '--buildplots',
                    action='store_true',
                    default=False)

parser.add_argument('-a', '--amplots',
                    action='store_true',
                    default=False)

parser.add_argument('-w','--weightplot',
                    action='store_true',
                    default=False)

parser.add_argument('-u', '--update',
                    action='store_true',
                    default=False)

parse = parser.parse_args()
otfile = parse.otfile
instfile = parse.instcal
logfilename = parse.logfile
site_name = parse.observatory
startdate = parse.startdate
enddate = parse.enddate
daylightsavings = parse.dst
distribution = parse.distribution
update = parse.update
iqtemp = parse.iq
cctemp = parse.cc
wvtemp = parse.wv
planbuildplot = parse.buildplots
amplots = parse.amplots
weightplots = parse.weightplot

if update:  # download most recent International Earth Rotation and Reference Systems data
    download_IERS_A()

def getobserver(site_name, daylightsavings):
    """
    Initialize '~astroplan.Observer' for Cerro Pachon or Mauna Kea.

    Parameters
    ----------
    site_name : string
        Observatory site name. Accepts 'gemini_north', 'MK' (Mauna Kea), 'gemini_south', and 'CP' (Cerro Pachon).

    daylightsavings : boolean
        Toggle daylight savings time

    Returns
    -------
    site : '~astroplan.Observer'
        Observer site object

    timezone_name : string
        Pytz timezone name for observer site

    utc_to_local : '~astropy.units.Quantity'
        Hour difference between utc and local time.
    """

    if np.logical_or(site_name=='gemini_south',site_name=='CP'):
        site_name = 'gemini_south'
        timezone_name = 'Chile/Continental'
        if daylightsavings:
            utc_to_local = -3.*u.h
        else:
            utc_to_local = -4.*u.h
    elif np.logical_or(site_name=='gemini_north',site_name=='MK'):
        site_name = 'gemini_north'
        timezone_name = 'US/Hawaii'
        utc_to_local = -10.*u.h
    else:
        print('Input error: Could not determine observer location and timezone. '
              'Allowed inputs are \'gemini_south\', \'CP\'(Cerro Pachon), \'gemini_north\', and \'MK\'(Mauna Kea).')
        exit()

    site = Observer.at_site(site_name) #create Observer object for observatory site
    # can add timezone=timezone_name later if desired (useful if pytz objects are ever used)

    return site, timezone_name, utc_to_local

def getstarttimes(startdate, enddate, utc_to_local):
    """
    Create '~astropy.time.Time' object for 16:00 local time on start date of schedule period and
    get number of nights in schedule.

    Parameters
    ----------
    startdate : str or None
        Start date (eg. 'YYYY-MM-DD').  If None, defaults to current date.

    enddate : str or None
        End date (eg. 'YYYY-MM-DD'). If None, defaults to day after start date.

    utc_to_local : '~astropy.unit.Quantity'
        Hour difference between utc and local time.

    Returns
    -------
    local_start : '~astropy.time.Time' objects
        16:00 local time of start date

    n_nights : int
        number of nights between start date and end date
    """

    dform = re.compile('\d{4}-\d{2}-\d{2}') #yyyy-mm-dd format

    current_utc = time.Time.now()  # current UTC
    current_local = current_utc + utc_to_local  # current local time
    if startdate == None:
        local_start = time.Time(str(current_local)[0:10] + ' 16:00:00.00')  # 16:00:00 local tonight
    else:
        if dform.match(startdate): #check startdate format
            local_start = time.Time(startdate + ' 16:00:00.00')  # 16:00:00 local on startdate
        else:
            print('\nInput error: \"'+startdate+'\" not a valid start date.  Must be in the form \'YYYY-MM-DD\'')
            exit()
    if enddate == None: #default number of observation nights is 1
        n_nights = 1
    else:
        if dform.match(enddate):  # check enddate format
            local_end = time.Time(enddate + ' 16:00:00.00')  # 16:00:00 local on enddate
            d = int((local_end - local_start).value)  # days between startdate and enddate
            if d >= 0:
                n_nights = d  # count days e.g. [0*u.d,1*u.d,2*u.d,...]
            else:
                print('\nInput error: Selected end date \"'+enddate+'\" is prior to the start date.')
                exit()
        else:
            print('\nInput error: \"'+enddate+'\" not a valid end date.  Must be in the form \'YYYY-MM-DD\'')
            exit()

    return local_start, n_nights

def gettimeinfo(site, utc_night_starts):
    """
    Manage the computation and organization of all required time data, Sun data, and Moon data for the schedule
    period.

    Parameters
    ----------
    site : '~astroplan.Observer'
        Observer site object

    utc_night_starts : array of '~astropy.time.Time' objects
        Time object for 16:00 local time on each night in the scheduling period

    Returns
    -------
    timeinfo : list of '~gemini_classes.TimeInfo' objects
        TimeInfo objects containing time data for each night

    suninfo : list of '~gemini_classes.SunInfo' objects
        SunInfo objects containing time dependent Sun data for each night

    mooninfo : list of '~gemini_classes.MoonInfo' objects
        MoonInfo objects containing time dependent Moon data for each night
    """

    ttimer = False
    if ttimer:
        import time as t
        timer = t.time()

    dt = 0.1
    timeinfo = Parallel(n_jobs=10)(delayed(TimeInfo)(site=site, starttime=utc_night_start, dt=dt,
                                                         utc_to_local=utc_to_local) for utc_night_start in utc_night_starts)
    if ttimer:
        print('\n\tTime to gather time data: ', t.time() - timer)
        timer = t.time()

    suninfo = Parallel(n_jobs=10)(delayed(SunInfo)(site=site, utc_times=times.utc) for times in timeinfo)
    if ttimer:
        print('\n\tTime to gather sun data: ', t.time() - timer)
        timer = t.time()

    mooninfo = Parallel(n_jobs=10)(delayed(MoonInfo)(site=site, utc_times=times.utc) for times in timeinfo)
    if ttimer: print('\n\tTime to gather moon data: ', t.time() - timer)

    return timeinfo, suninfo, mooninfo

def utc_times(local_time, utc_to_local, n):
    """
    Convert local time to utc and copy for n consecutive days. Return array of '~astropy.time.Time' objects.

    Parameters
    ----------
    local_time : '~astropy.time.Time'

    utc_to_local : '~astropy.unit.Quantity'
        Hour difference between utc and local time.

    n : int
        Number of consecutive days

    Returns
    -------
    array of '~astropy.time.Time' objects
        utc time for several consecutive days
    """


def setconddist(distribution):
    """
    Set condition generator function.

    Parameters
    ----------
    distribution : str or None
        distribution type for generating conditions.

    Returns
    -------
    None, 'None' : NoneType, str
        Returns NoneType and string 'None' if distribution is None

    function, 'distribution type' : function, str
        Returns distribution generator function and name of distribution if distribution not None.

    ValueError
        Returns ValueError if distribution is neither None nor an accepted distribution type.

    """
    if distribution == 'None': distribution = None
    if distribution is not None:

        if distribution == 'gaussian' or distribution == 'g':
            distribution = 'gaussian'
            return condgen.gauss, distribution
        else:
            print('\n \'' + str(distribution) + '\' is not an accepted distribution type.'
                                               ' Refer to program help using \'-h\' and try again.')
            raise ValueError
    else:
        return None, None

def setcond(iq, cc, wv):
    """
    Format input condition constraints to be 'Any' or a 2-digit percentage.
    If input is not 'any', 'Any', or a number, a ValueError is raised.

    Example
    -------
    'any' --> 'Any'
    '10'  --> '10%'
    '85%' --> '85%'
    '08'  --> '08%'
    '808' --> '80%'
    '8'   --> '8%'
    '8%'  --> ValueError

    Parameters
    ----------
    iq : str
        image quality percentile

    cc : str
        cloud condition percentile

    wv : str
        water vapor percentile

    Returns
    -------
    newiq, newcc, newwv : str, str, str
        Formatted condition constraints strings
    """

    if iq.lower() == 'any':
        newiq = 'Any'
    else:
        try:
            float(iq[0:2])
            newiq = iq[0:2] + '%'
        except ValueError:
            raise ValueError
    if cc.lower() == 'any':
        newcc = 'Any'
    else:
        try:
            float(cc[0:2])
            newcc = cc[0:2] + '%'
        except ValueError:
            raise ValueError
    if wv.lower() == 'any':
        newwv = 'Any'
    else:
        try:
            float(wv[0:2])
            newwv = wv[0:2] + '%'
        except ValueError:
            raise ValueError
    return newiq, newcc, newwv

def getobservations(otfile):
    """
    Manage input catalog and initialization of 'Gobservations' object.

    Parameters
    ----------
    otfile : str
        ot catalog file name

    Returns
    -------
    obs : '~gemini_otcat.GObservation' object.
        Information for observations in queue
    """

    otcat = Gcatfile(otfile)  # class containing catalog data under column header attribute name as strings
    obs = Gobservations(otcat, epoch=local_start) # class containing sorted and converted catalog data (as in IDL version)
    n_obs = len(obs.obs_id)
    print('\n\t'+str(n_obs)+' observations selected for queue...')
    return obs, n_obs


# **********************************************************************************************************************
#                                                   GQPT PROTOTYPE
# **********************************************************************************************************************

if ttimer: timer = t.time()
cond_func, distribution = setconddist(distribution)
if ttimer: print('\n\tTimer setconddist = ', t.time() - timer)

if ttimer: timer = t.time()
iq, cc, wv = setcond(iq=iqtemp, cc=cctemp, wv=wvtemp)
if ttimer: print('\n\tTimer setcond = ', t.time() - timer)

if ttimer: timer = t.time()
current_local = time.Time.now()
if ttimer: print('\n\tTimer current_local = ', t.time() - timer)

if ttimer: timer = t.time()
site, timezone_name, utc_to_local = getobserver(site_name=site_name, daylightsavings=daylightsavings)
if ttimer: print('\n\tTimer getobserver = ', t.time() - timer)

if ttimer: timer = t.time()
local_start, n_nights = getstarttimes(startdate=startdate, enddate=enddate, utc_to_local=utc_to_local)
if ttimer: print('\n\tTimer getstarttimes = ', t.time() - timer)

if ttimer: timer = t.time()
local_starts = local_start + np.arange(n_nights) * u.d
utc_starts =  local_starts - utc_to_local
if ttimer: print('\n\tTimer utc_starts = ', t.time() - timer)

if ttimer: timer = t.time()
timeinfo, suninfo, mooninfo = gettimeinfo(site=site, utc_night_starts=utc_starts)
if ttimer: print('\n\tTimer gettimeinfo = ', t.time() - timer)

verbose = False
customfilename = False
while True:

    if customfilename:
        pass
    elif logfilename is not None:
        statfilename = logfilename
    else:
        statfilename = 'logfile' + time.Time.now().isot[:-4] + '.log'

    seed = random.seed(1000)

    aprint = '\t{0:<35s}{1}'  # print two strings
    bprint = '\t{0:<35s}{1:<.4f}'  # print string and number

    print('\n' + aprint.format('Site: ', site.name))
    print(bprint.format('Height: ', site.location.height))
    print(bprint.format('Longitude: ', coordinates.Angle(site.location.lon)))
    print(bprint.format('Latitude: ', coordinates.Angle(site.location.lat)))

    print('\n' + aprint.format('Start/end date: ',
                               local_start.iso[0:10] + ', ' + (local_start + n_nights * u.d).iso[0:10]))

    print('\n' + aprint.format('Schedule start (local): ', local_start.iso[0:19]))
    print(aprint.format('Schedule start (UTC): ', utc_starts[0].iso[0:19]))
    print(bprint.format('Julian date (UTC): ', utc_starts[0].jd))
    print(bprint.format('Julian year (UTC): ', utc_starts[0].jyear))
    print(aprint.format('Number of nights: ', n_nights))

    # ================================================ Menu ============================================================
    while True:

        fsel = '\t{0:10s}{1:40s}{2:40}'
        print('\n\tParameters:')
        print('\t---------------------------------------------------------------------')
        print(fsel.format('1.', 'Catalog file', otfile))
        print(fsel.format('2.', 'Inst. calendar file', instfile))
        print(fsel.format('3.', 'Log file', statfilename))
        print(fsel.format('4.', 'Conditions (iq,cc,wv)', '('+iq+', '+cc+', '+wv+')'))
        print(fsel.format('5.', 'Condition generator distribution', str(distribution)))
        print(fsel.format('6.', 'Create airmass plots', str(amplots)))
        print(fsel.format('7.', 'Create airmass buildup plots', str(planbuildplot)))
        print(fsel.format('8.', 'Weighting function plot mode', str(weightplots)))
        print()
        print(fsel.format('dir', 'Show files in current directory', ''))

        userinput = input('\n Press enter to run or select number to make changes: ')
        if userinput == '':
            break

        elif userinput == 'dir':
            files = [f for f in os.listdir('.') if os.path.isfile(f)]
            files.sort()
            print('\n Current directory:\n -----------------')
            for file in files:
                print(' ' + file)

        elif userinput == '1':
            tempotfile = input(' Input OT catalog file name: ')
            if tempotfile=='':
                print(' No input received. Changes not made.')
                continue
            else:
                files = [f for f in os.listdir('.') if os.path.isfile(f)]
                if tempotfile in files:
                    obs, n_obs = getobservations(tempotfile)
                    otfile = tempotfile
                else:
                    print('\n Current directory:\n -----------------')
                    for file in files: print(' '+file)
                    print('\n File \''+tempotfile+'\' not found in current directory.')
                    continue

        elif userinput == '2':
            tempinstfile = input(' Input instrument calendar file name: ')
            if tempinstfile == '':
                print(' No input received. Changes not made.')
                continue
            else:
                files = [f for f in os.listdir('.') if os.path.isfile(f)]
                if tempinstfile in files:
                    instcal = getinstruments(tempinstfile)
                    instfile = tempinstfile
                else:
                    print('\n File \'' + tempinstfile + '\' not found in current directory.')
                    continue

        elif userinput == '3':
            tempstatfile = input(' Choose log file name: ')
            if tempstatfile == '':
                print(' No input received. Changes not made.')
                continue
            else:
                files = [f for f in os.listdir('.') if os.path.isfile(f)]
                if tempstatfile in files:
                    userinput = input(' ' + tempstatfile + ' already exists. Overwrite? [y/n] ')
                    if userinput.lower() == 'y' or userinput == '':
                        statfilename = tempstatfile
                        continue
                    else:
                        print(' Changes not made.')
                        continue
                else:
                    statfilename = tempstatfile
                    continue

        elif userinput == '4':
            condinput = input(' Input iq, cc, wv  percentiles separated by spaces (eg. \'20 50 Any\'): ')
            tempconds = condinput.split(' ')
            if len(tempconds)!=3:
                print(' Did not receive 3 values. Changes not made.')
                continue
            else:
                try:
                    iq, cc, wv = setcond(iq=tempconds[0], cc=tempconds[1], wv=tempconds[2])
                except ValueError:
                    print(' ValueError! Could not set new conditions. Changes not made.')
                    continue

        elif userinput == '5':
            tempdist = input(' Input condition generator distribution type (or \'None\'): ')
            if tempdist=='':
                print(' No input received. Changes not made.')
                continue
            else:
                try:
                    cond_func, distribution = setconddist(tempdist)
                except ValueError:
                    continue

        elif userinput == '6':
            if amplots == True:
                amplots = False
                print(' Turned off airmass plots.')
                continue
            elif amplots == False:
                amplots = True
                print(' Turned on airmass plots.')

        elif userinput == '7':
            if planbuildplot == True:
                planbuildplot = False
                print(' Turned off airmass build-up plots.')
                continue
            if planbuildplot == False:
                planbuildplot = True
                print(' Turned on airmass build-up plots.')

        elif userinput == '8':
            if weightplots == True:
                weightplots = False
                print(' Turned off weight function plotting mode.')
                continue
            if weightplots == False:
                weightplots = True
                print(' Turned on weight function plotting mode.')

        else:
            print(' Did not recognize input. Changes not made.')
            continue

    # ========================================= Reload packages ========================================================
    if ttimer: timer = t.time()
    importlib.reload(select_obs)
    importlib.reload(weightplotmode)
    if ttimer: print('\n\tTimer packages = ', t.time() - timer)

    # ============================= Retrieve observation and instrument information ====================================
    if ttimer: timer = t.time()
    obs, n_obs = getobservations(otfile)
    if ttimer: print('\n\tTimer getobservations = ', t.time() - timer)

    if ttimer: timer = t.time()
    instcal = getinstruments(instfile)
    if ttimer: print('\n\tTimer instcal = ', t.time() - timer)

    # ==================================== Initialize statistics logfile ===============================================
    semesterstats = {'night_time': 0. * u.h, 'used_time': 0. * u.h}

    logger.initLogFile(filename=statfilename, catalogfile=otfile, site=site,
                       start=local_start, n_nights=n_nights, dst=daylightsavings)

    logger.logProgStats(filename=statfilename, obs=obs, semesterinfo=semesterstats,
                        description='Initial completion status...')

    print('\n\tOutput log file: '+statfilename)

    # ========================================== Timing windows ========================================================
    if ttimer: timer = t.time()
    # testconst = [{'start': local_start-1.*u.day, 'duration': -1, 'repeat': 0, 'period': None}]
    # testconst = [{'start': local_start-1.*u.day, 'duration': 5*u.h, 'repeat': 100, 'period': 8*u.h}]
    timewins = Parallel(n_jobs=10)(delayed(timingwindows)(time_consts=obs.time_const[i], t1=timeinfo[0].start,
                                                              t2=timeinfo[-1].end) for i in range(n_obs))
    if ttimer: print('\n\tTimer timewindows = ', t.time() - timer)
    # for night in timeinfo:
    #     print(night.start.iso,night.end.iso)
    # for i in range(n_obs):
    #     print('\n Observation:',obs.obs_id[i])
    #     print(' Time constraints...')
    #     if obs.time_const[i] is None:
    #         print('\t',obs.time_const[i])
    #     else:
    #         for time_const in obs.time_const[i]:
    #             print('\t',time_const['start'].iso, time_const['duration'], time_const['repeat'], time_const['period'])
    #
    #     print(' Time windows...')
    #     if timewins[i] is not None:
    #         for win in timewins[i]:
    #             print('\t',win[0].iso,win[1].iso)
    #     else:
    #         print('\t',timewins[i])

    # ======================================== Schedule night conditions ===============================================

    for i_day in range(len(timeinfo)):  # cycle through schedule days
        date = str(timeinfo[i_day].utc[0].iso)[0:10]  # date as string
        print('\n\n\t_______________________ Night of '+date+' _______________________')

        # ====================================== Set/generate conditions ===============================================
        if ttimer: timer = t.time()
        if distribution is not None: # generate random sky conditions from selected distribution
            randcond = cond_func()
            iq = randcond.iq
            cc = randcond.cc
            wv = randcond.wv
        acond = actual_conditions(iq, cc, 'Any', wv) # convert actual conditions to decimal values
        if ttimer: print('\n\tTimer actual_conditions = ', t.time() - timer)
        print('\n\tSky conditions (iq,cc,wv): {0} , {1} , {2}'.format(acond[0], acond[1], acond[3]))

        # ====================== Get weighting factor for RA distribution of observation hours =========================
        if ttimer: timer = t.time()
        wra = weight_ra(ras=obs.ra, tot_time=obs.tot_time, obs_time=obs.obs_time)
        if verbose: print('wra (ra distribution weight)', wra)
        if ttimer: print('\n\tTimer weight_ra = ', t.time() - timer)

        # =============================== Select schedule candidates from queue ========================================
        i_obs = np.arange(n_obs)

        if ttimer: timer = t.time()
        try:
            i_obs = i_obs[select_obs.selectinst(i_obs=i_obs, obs=obs, instcal=instcal, datestring=date)]
        except ValueError:
            userinput = input(' Continue anyways? [y/n]: ')
            if userinput.lower()=='y' or userinput=='':
                pass
            else:
                break
        if ttimer: print('\n\tTimer selectinst = ', t.time() - timer)

        if ttimer: timer = t.time()
        # for i in i_obs:
            # print(select_obs.selecttimewindow(time_consts=obs.time_const[i], timeinfo=timeinfo[i]))
        i_obs = i_obs[np.where(Parallel(n_jobs=10)(delayed(select_obs.selecttimewindow)(time_consts=obs.time_const[j], timeinfo=timeinfo[i_day]) for j in i_obs))[0][:]]
        if ttimer: print('\n\tTimer selecttimewindow = ', t.time() - timer)

        if ttimer: timer = t.time()
        # for i in i_obs:
            # print(select_obs.selecttimewindow(time_consts=obs.time_const[i], timeinfo=timeinfo[i]))
        i_obs = i_obs[np.where(select_obs.selectincomplete(tot_time=obs.tot_time[i_obs], obs_time=obs.obs_time[i_obs]))[0][:]]
        if ttimer: print('\n\tTimer selectincomplete = ', t.time() - timer)

        # =============================== Compute data for selected observations =======================================

        if ttimer: timer = t.time()
        targetinfo = Parallel(n_jobs=10)(delayed(TargetInfo)(ra=obs.ra[j], dec=obs.dec[j], name=obs.obs_id[j],
                                         site=site, utc_times=timeinfo[i_day].utc) for j in i_obs)
        if ttimer: print('\n\tTimer TargetInfo = ', t.time() - timer)

        if ttimer: timer = t.time()
        mdists = Parallel(n_jobs=10)(delayed(gcirc)(mooninfo[i_day].ra, mooninfo[i_day].dec, target.ra,
                                       target.dec) for target in targetinfo)
        if ttimer: print('\n\tTimer mdists = ', t.time() - timer)

        if ttimer: timer = t.time()
        vsbs = Parallel(n_jobs=10)(delayed(sb)(mpa=mooninfo[i_day].phase, mdist=mdists[i], mZD=mooninfo[i_day].ZD,
                                               ZD=targetinfo[i].ZD, sZD=suninfo[i_day].ZD, cc=acond[1]) for i in range(len(targetinfo)))
        if ttimer: print('\n\tTimer sb = ', t.time() - timer)

        if ttimer: timer = t.time()
        # ttime = np.round((obs.tot_time[i_obs] - obs.obs_time[i_obs]) * 10) / 10  # remaining time in observations
        prstatus = getprstatus(prog_ref=obs.prog_ref[i_obs], obs_time=obs.obs_time[i_obs])
        if ttimer: print('\n\tTimer prstatus = ', t.time() - timer)

        if ttimer: timer = t.time()
        weightinfo = calc_weight(site=site, i_obs=i_obs, obs=obs, targetinfo=targetinfo, acond=acond, vsbs=vsbs,
                                 wra=wra, prstatus=prstatus)
        if ttimer: print('\n\tTimer calc_weight = ', t.time() - timer)

        if ttimer: timer = t.time()
        # generate PlanInfo plan object and update Gobservation object
        plan, obs_updated = PlanInfo.priority(i_obs=i_obs ,obs=copy.deepcopy(obs), timeinfo=timeinfo[i_day], targetinfo=targetinfo, showbuildup=planbuildplot)
        if ttimer: print('\n\tTimer PlanInfo = ', t.time() - timer)

        if ttimer: timer = t.time()
        # print plan
        [print(line) for line in printPlanTable(plan=plan, i_obs=i_obs, obs=obs_updated, timeinfo=timeinfo[i_day], targetinfo=targetinfo)]
        if ttimer: print('\n\tTimer printPlanTable = ', t.time() - timer)

        logger.logPlanStats(filename=statfilename, i_obs=i_obs, obs=obs_updated, plan=plan, timeinfo=timeinfo[i_day], suninfo=suninfo[i_day],
                        mooninfo=mooninfo[i_day], targetinfo=targetinfo, acond=acond)
        if ttimer: print('\n\tTimer logPlanStats = ', t.time() - timer)

        semesterstats['night_time'] = semesterstats['night_time'] + plan.night_length
        semesterstats['used_time'] = semesterstats['used_time'] + plan.used_time

        if amplots:
            if ttimer: timer = t.time()
            amplot(plan=plan, timeinfo=timeinfo[i_day], mooninfo=mooninfo[i_day], targetinfo=targetinfo)
            if ttimer: print('\n\tTimer amplot = ', t.time() - timer)

        if weightplots:
            weightplotmode.weightplotmode(site=site, timeinfo=timeinfo[i_day], suninfo=suninfo[i_day],
                                          mooninfo=mooninfo[i_day], targetinfo=targetinfo, obs=obs, i_obs=i_obs,
                                          plan=plan, acond=acond)

        obs=obs_updated  # updated obs structure

    logger.logProgStats(filename=statfilename, obs=obs, semesterinfo=semesterstats, description='Schedule results...')

    print('\n\n Simulation complete!\n\n')