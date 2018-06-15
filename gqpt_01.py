# import time as t
# timer = t.time()

import argparse
import random
import re
import numpy as np
import textwrap
import astropy.units as u
from astroplan import (download_IERS_A, Observer)
from astropy import (coordinates, time)

# gqpt packages
import condgen
import gcirc
import sb
from conversions import actual_conditions
from calc_night import calc_night
from calc_weight import calc_weight
from gemini_classes import Gobservations, Gcatfile
from amplot import amplot
from printplan import printplan
from gemini_schedulers import Gschedule

# print('\nTime to import packages: ',t.time()-timer)
# timer = t.time() #reset timer

seed = random.seed(1000)


#   =============================================== Read and command line inputs =============================================================

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

# ========================================= Input conditions =====================================================

# Assign condition distribution generator function if selected
conddist = False
if distribution is not None:
    conddist = True
    if distribution=='gaussian' or distribution=='g':
        cond_func = condgen.gauss_cond
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

# ========================== Create astroplan.Observer object for observatory site ====================================
# Set site name for Cerro Pachon or Mauna Kea and initialize Observer object

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

print('')
print(aprint.format('Site: ',site.name))
print(bprint.format('Height: ',site.location.height))
print(bprint.format('Longitude: ',coordinates.Angle(site.location.lon)))
print(bprint.format('Latitude: ',coordinates.Angle(site.location.lat)))

# ================= Read input start/end dates. Create time object for current local and utc time ===================

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
        end_time = time.Time(enddate+' 16:00:00.00')  # 16:00:00 local on enddate
        d = int((end_time - local_start).value + 1)  # days between startdate and enddate
        if d >= 0: 
            count_day = np.arange(d) * u.d  # count days e.g. [0*u.d,1*u.d,2*u.d,...]
        else: 
            print('\nInput error: Selected end date \"'+enddate+'\" is prior to the start date.')
            exit()
    else:
        print('\nInput error: \"'+enddate+'\" not a valid end date.  Must be in the form \'yyyy-mm-dd\'')
        exit()

print('')
print(aprint.format('Schedule start (local): ',local_start.iso))
print(aprint.format('Schedule start (UTC): ',(local_start-utc_to_local).iso))
print(bprint.format('Julian date (UTC): ',(local_start-utc_to_local).jd))
print(bprint.format('Julian year (UTC): ',(local_start-utc_to_local).jyear))
print(aprint.format('Number of nights: ', len(count_day)))

# ====================== Read catalog and select observations for queue ========================
otcat = Gcatfile(otfile)  # object containing catalog data under column headers.

# Select observations from catalog to queue
i_obs = np.where(np.logical_and(np.logical_or(otcat.obs_status=='Ready',otcat.obs_status=='Ongoing'),
    np.logical_or(otcat.obs_class=='Science',
    np.logical_and(np.logical_or(otcat.inst=='GMOS',otcat.inst=='bHROS'),
    np.logical_or(otcat.obs_class=='Nighttime Partner Calibration',
                  otcat.obs_class=='Nighttime Program Calibration')))))[0][:] #get indeces of observation to queue

print('\n\t'+str(len(i_obs))+' observations selected for queue...')

# Interpret and sort catalog data into appropriate columns(as done in Bryan Miller's IDL version)
obs = Gobservations(catinfo=otcat,i_obs=i_obs) 

# =================================== Begin Queueing ==============================================

# print('\nTime to prepare catalog data: ',t.time()-timer)

for i_day in count_day: #cycle through observation days

    night_start = local_start + i_day # time object for 18:00 local on current night
    night_start_utc = night_start - utc_to_local # time object for 18:00 UTC on current night
    date = str(night_start)[0:10]  # date as string

    print('\n\n\t_______________________ Night of '+date+' _______________________')

    if conddist: # generate random sky conditions from selected distribution
        iq,cc,wv = cond_func()
    acond = actual_conditions(iq,cc,'Any',wv) # convert actual conditions to decimal values
    print('\n\tSky conditions (iq,cc,wv): {0} , {1} , {2}'.format(iq, cc, wv))

    # calculate time dependent parameters for observing window
    timeinfo, suninfo, mooninfo, targetinfo = calc_night(obs=obs, site=site, starttime=night_start_utc,
                                                         utc_to_local=utc_to_local)

    for target in targetinfo: # compute additional time dependent info and store in TargetInfo objects
        target.mdist = gcirc.gcirc(mooninfo.ra, mooninfo.dec, target.ra,
                                   target.dec)  # angular distance between target and moon at utc_times
        target.vsb = sb.sb(mpa=mooninfo.phase, mdist=target.mdist, mZD=mooninfo.ZD, ZD=target.ZD, sZD=suninfo.ZD,
                           cc=acond[1])  # visible sky background magnitude at utc_times

    # calculate weights
    weightinfo = calc_weight(site=site, obs=obs, timeinfo=timeinfo, targetinfo=targetinfo, acond=acond)

    # generate Gschedule plan object and update Gobservation object
    prior_plan, obs = Gschedule.priority(obs=obs, timeinfo=timeinfo, targetinfo=targetinfo)

    # print plan
    printplan(plan=prior_plan, obs=obs, timeinfo=timeinfo, targetinfo=targetinfo)

    # airmass plot to png file
    amplot(plan=prior_plan, timeinfo=timeinfo, mooninfo=mooninfo, targetinfo=targetinfo)