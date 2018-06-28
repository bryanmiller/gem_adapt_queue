import argparse
import random
import re
import numpy as np
import textwrap
import astropy.units as u
from joblib import Parallel, delayed
from astroplan import (download_IERS_A, Observer)
from astropy import (coordinates, time)

# gqpt packages
import logger
from sb import sb
from gcirc import gcirc
from amplot import amplot
from condgen import condgen
from calc_weight import calc_weight
from printplan import printPlanTable
from gemini_schedulers import PlanInfo
from conversions import actual_conditions
from gemini_otcat import Gobservations, Gcatfile
from gemini_classes import TimeInfo,SunInfo,MoonInfo,TargetInfo




seed = random.seed(1000)


#   ======================================= Read and command line inputs ===============================================

verbose = False

aprint = '\t{0:<35s}{1}'  # print two strings
bprint = '\t{0:<35s}{1:<.4f}'  # print string and number

parser = argparse.ArgumentParser(prog='gqpt.py',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description = textwrap.dedent('''                                                             
                                                                Guide
                    *****************************************************************************************************               

                        otfile                  OT catalog filename.

                        -o   --observatory      Observatory site [DEFAULT='gemini_south']. Accepts the following:
                                                1. 'gemini_north' (or 'MK' for Mauna Kea)
                                                2. 'gemini_south' (or 'CP' for Cerro Pachon)
                        
                        -s   --startdate        Start date 'YYYY-MM-DD' [DEFAULT=tonight]
                        
                        -e   --enddate          End date 'YYYY-MM-DD' [DEFAULT=startdate]. End date must be before
                                                the start date.  If no enddate is provided, a single night will be
                                                scheduled.

                        -dst --daylightsavings  Toggle daylight savings time [DEFAULT=False]
                                                
                                                Conditions (if distribution=False):
                        -i   --iq               Image quality constraint [DEFAULT=70]
                        -c   --cc               Cloud cover constraint   [DEFAULT=50]
                        -w   --wv               Water vapor constraint   [DEFAULT=Any]

                        -d   --distribution     Generate conditions from distribution [DEFAULT=False]. Accepts the following:
                                                1. 'gaussian' (or 'g')  

                        -u   --update           Download up-to-date IERS(International Earth Rotation and Reference Systems).

                    *****************************************************************************************************                        
                                            '''))

parser.add_argument(action='store',
                    dest='otfile')

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

parser.add_argument('-i', '--iq',
                    default='70')

parser.add_argument('-c', '--cc',
                    default='50')

parser.add_argument('-w', '--wv',
                    default='Any')

parser.add_argument('-d', '--distribution',
                    default=None)

parser.add_argument('-u', '--update',
                    action='store_true',
                    default=False)

parse = parser.parse_args()
otfile = parse.otfile
site_name = parse.observatory
startdate = parse.startdate
enddate = parse.enddate
dst = parse.dst
distribution = parse.distribution
update = parse.update

if update:  # download up to date International Earth Rotation and Reference Systems data
    download_IERS_A()

# ========================================= Handle input conditions ====================================================

# Assign condition distribution generator function if selected
conddist = False
if distribution is not None:
    conddist = True
    if distribution=='gaussian' or distribution=='g':
        cond_func = condgen.gauss
    else:
        print('\n\''+str(distribution)+'\' is not an accepted distribution type.'
                                       ' Refer to program help using \'-h\' and try again.')
        exit()
else:
    # Format condition constraint input if random distribution not specified.
    if parse.iq == 'Any':
        iq = parse.iq
    else:
        iq = parse.iq[0:2] + '%'
    if parse.cc == 'Any':
        cc = parse.cc
    else:
        cc = parse.cc[0:2] + '%'
    if parse.wv == 'Any':
        wv = parse.wv
    else:
        wv = parse.wv[0:2] + '%'

# =========================================== Handle input site name ===================================================
# Set site name for Cerro Pachon or Mauna Kea and initialize astroplan.Observer object

# Check observatory input. return timezone, utc-local time difference
if np.logical_or(site_name=='gemini_south',site_name=='CP'):
    site_name = 'gemini_south'
    timezone_name = 'Chile/Continental'
    if dst == True:
        utc_to_local = -3.*u.h
    else:
        utc_to_local = -4.*u.h 
elif np.logical_or(site_name=='gemini_north',site_name=='MK'):
    site_name = 'gemini_north'
    timezone_name = 'US/Hawaii'
    utc_to_local = -10.*u.h
else:
    print('Input error: Could not determine observer location and timezone. Allowed inputs are \'gemini_south\', \'CP\'(Cerro Pachon),\'gemini_north\', or \'MK\'(Mauna Kea).')
    exit()

site = Observer.at_site(site_name) #create Observer object for observatory site
# can add timezone=timezone_name later if desired (useful if pytz objects are ever used)

print('\n'+aprint.format('Site: ',site.name))
print(bprint.format('Height: ',site.location.height))
print(bprint.format('Longitude: ',coordinates.Angle(site.location.lon)))
print(bprint.format('Latitude: ',coordinates.Angle(site.location.lat)))

# ======================================== Handle input start/end dates  ===============================================
# Set start and end dates. Create time object for current local and utc time

dform = re.compile('\d{4}-\d{2}-\d{2}') #yyyy-mm-dd format

current_utc = time.Time.now()  # current UTC
current_local = current_utc + utc_to_local  # current local time
if startdate == None:
    local_start = time.Time(str(current_local)[0:10] + ' 16:00:00.00')  # 16:00:00 local tonight
else:
    if dform.match(startdate): #check startdate format
        local_start = time.Time(startdate + ' 16:00:00.00')  # 16:00:00 local on startdate
    else:
        print('\nInput error: \"'+startdate+'\" not a valid start date.  Must be in the form \'yyyy-mm-dd\'')
        exit()
if enddate == None: #default number of observation nights is 1
    count_day = [0] * u.d
else:
    if dform.match(enddate):  # check enddate format
        local_end = time.Time(enddate + ' 16:00:00.00')  # 16:00:00 local on enddate
        d = int((local_end - local_start).value + 1)  # days between startdate and enddate
        if d >= 0: 
            count_day = np.arange(d) * u.d  # count days e.g. [0*u.d,1*u.d,2*u.d,...]
        else: 
            print('\nInput error: Selected end date \"'+enddate+'\" is prior to the start date.')
            exit()
    else:
        print('\nInput error: \"'+enddate+'\" not a valid end date.  Must be in the form \'yyyy-mm-dd\'')
        exit()

utc_start = local_start-utc_to_local
utc_night_starts = utc_start + count_day

print('\n'+aprint.format('Schedule start (local): ',local_start.iso))
print(aprint.format('Schedule start (UTC): ',utc_start.iso))
print(bprint.format('Julian date (UTC): ',utc_start.jd))
print(bprint.format('Julian year (UTC): ',utc_start.jyear))
print(aprint.format('Number of nights: ', len(count_day)))

# ========================================= Handle input catalog file ==================================================
# Read appropriate file into GObservation class and select observations for queue

otcat = Gcatfile(otfile)  # class containing catalog data under column header attribute name as strings
obs = Gobservations(otcat, epoch=local_start) # class containing sorted and converted catalog data (as in IDL version)

print('\n\t'+str(len(obs.obs_id))+' observations selected for queue...')

# ======================================== Initialize statistics logfile ===============================================
semesterstats = {'night_time': 0. * u.h, 'used_time': 0. * u.h}

statfilename = 'logfile'+current_local.isot[:-5]+'.log'
statfilename = 'testlogfile.log'

logger.initLogFile(filename=statfilename, catalogfile=otfile, site=site,
                   start=local_start, n_nights=len(count_day), dst=dst)

logger.logProgStats(filename=statfilename, obs=obs, semesterinfo=semesterstats,
                    description='Initial completion status...')

print('\n\tOuput file: '+statfilename)

# =============================== Compute times, sun data, and moon data for all dates =================================

ttimer = True
if ttimer:
    import time as t
    timer = t.time()
dt = 0.1
print()
timeinfo = Parallel(n_jobs=10)(delayed(TimeInfo)(site=site, starttime=utc_night_start, dt=dt,
                                                     utc_to_local=utc_to_local) for utc_night_start in utc_night_starts)
if ttimer: print('\n\tTime to gather time data: ', t.time() - timer)
timer = t.time()

suninfo = Parallel(n_jobs=10)(delayed(SunInfo)(site=site, utc_times=times.utc) for times in timeinfo)
if ttimer: print('\n\tTime to gather sun data: ', t.time() - timer)
timer = t.time()

mooninfo = Parallel(n_jobs=10)(delayed(MoonInfo)(site=site, utc_times=times.utc) for times in timeinfo)
if ttimer: print('\n\tTime to gather moon data: ', t.time() - timer)

# ============================================== Begin Queueing ========================================================

for i in range(len(timeinfo)):  # cycle through observation days

    # night_start = local_start + i_day # time object for 18:00 local on current night
    # night_start_utc = night_start - utc_to_local # time object for 18:00 UTC on current night
    date = str(timeinfo[i].utc[0].iso)[0:10]  # date as string

    print('\n\n\t_______________________ Night of '+date+' _______________________')

    if conddist: # generate random sky conditions from selected distribution
        randcond = cond_func()
        iq = randcond.iq
        cc = randcond.cc
        wv = randcond.wv

    acond = actual_conditions(iq, cc, 'Any', wv) # convert actual conditions to decimal values
    print('\n\tSky conditions (iq,cc,wv): {0} , {1} , {2}'.format(acond[0], acond[1], acond[3]))

    # calculate time dependent parameters for observing window
    # timeinfo, suninfo, mooninfo, targetinfo = calc_night(obs=obs, site=site, starttime=night_start_utc,
    #                                                      utc_to_local=utc_to_local)

    if ttimer:
        timer = t.time()

    targetinfo = Parallel(n_jobs=10)(delayed(TargetInfo)(ra=obs.ra[j], dec=obs.dec[j], name=obs.obs_id[j],
                                     site=site, utc_times=timeinfo[i].utc) for j in range(len(obs.obs_id)))
    if ttimer:
        print('\n\tTime to gather target data: ', t.time() - timer)
        timer = t.time()

    # compute mdist and visible sky brightness using parallel processes.
    mdists = Parallel(n_jobs=10)(delayed(gcirc)(mooninfo[i].ra, mooninfo[i].dec, target.ra,
                                   target.dec) for target in targetinfo)
    for k in range(len(mdists)):
        targetinfo[k].mdist=mdists[k]
    if ttimer:
        print('\n\tTime to calc mdist: ', t.time() - timer)
        timer = t.time()

    vsbs = Parallel(n_jobs=10)(delayed(sb)(mpa=mooninfo[i].phase, mdist=target.mdist, mZD=mooninfo[i].ZD,
                                           ZD=target.ZD, sZD=suninfo[i].ZD, cc=acond[1]) for target in targetinfo)
    for k in range(len(vsbs)):
        targetinfo[k].vsb = vsbs[k]
    if ttimer:
        print('\n\tTime to calc vsb: ', t.time() - timer)
        timer = t.time()

    # calculate weights
    weightinfo = calc_weight(site=site, obs=obs, timeinfo=timeinfo[i], targetinfo=targetinfo, acond=acond)

    # generate PlanInfo plan object and update Gobservation object
    plan, obs = PlanInfo.priority(obs=obs, timeinfo=timeinfo[i], targetinfo=targetinfo)

    # print plan
    [print(line) for line in printPlanTable(plan=plan, obs=obs, timeinfo=timeinfo[i], targetinfo=targetinfo)]

    logger.logPlanStats(filename=statfilename, obs=obs, plan=plan, timeinfo=timeinfo[i], suninfo=suninfo[i],
                    mooninfo=mooninfo[i], targetinfo=targetinfo, acond=acond)
    semesterstats['night_time'] = semesterstats['night_time'] + plan.night_length
    semesterstats['used_time'] = semesterstats['used_time'] + plan.used_time

    # airmass plot to png file
    amplot(plan=plan, timeinfo=timeinfo[i], mooninfo=mooninfo[i], targetinfo=targetinfo)

logger.logProgStats(filename=statfilename, obs=obs, semesterinfo=semesterstats, description='Schedule results...')
exit()
