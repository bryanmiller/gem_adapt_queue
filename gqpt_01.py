import argparse
import random
import re
import os
import numpy as np
import textwrap
import astropy.units as u
from joblib import Parallel, delayed
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
from printplan import plantable
from gemini_schedulers import PlanInfo


def _init_logfile(filename, catalogfile, site, start, n_nights, dst):
    aprint = '\n\t{0:<15s}{1}'  # print two strings
    bprint = '\n\t{0:<15s}{1:<.4f}'  # print string and number
    with open(filename, 'w') as log:
        log.write('\n Program: gqpt.py')
        log.write('\n Run from directory: ' + os.getcwd())
        log.write('\n Observation information retrieved from '+catalogfile)
        log.write('\n\n Dates: '+str(start)[0:10]+' to '+str(start+(n_nights-1)*u.d)[0:10])
        log.write('\n Number of nights: '+str(n_nights))
        log.write('\n Daylight savings time: ' + str(dst))
        log.write('\n\n Observing site: ')
        log.write(aprint.format('Site: ', site.name))
        log.write(bprint.format('Height: ', site.location.height))
        log.write(bprint.format('Longitude: ', coordinates.Angle(site.location.lon)))
        log.write(bprint.format('Latitude: ', coordinates.Angle(site.location.lat)))
        log.close()
    return

def _log_planstats(filename, obs, plan, timeinfo, suninfo, mooninfo, targetinfo, acond):

    with open(filename, 'a') as log:
        log.write('\n\n -----------------------------------------------------------')
        log.write('\n\n Night schedule:')
        log.write('\n\tSky conditions (iq,cc,wv): {0} , {1} , {2}'.format(acond[0], acond[1], acond[3]))
        [log.write('\n'+line) for line in timeinfo.table()]
        [log.write('\n' + line) for line in suninfo.table()]
        [log.write('\n' + line) for line in mooninfo.table()]
        [log.write('\n' + line) for line in plan.table()]
        [log.write('\n' + line) for line in plantable(plan=plan, obs=obs, timeinfo=timeinfo, targetinfo=targetinfo)]
        log.close()

    return

def _log_progstats(filename, obs, semesterinfo, description):
    """
    Computes and appends observations statistics to a file.

    Example
    -------

    >>> _log_progstat(filename='myfile.log', obs=Gobservations(catinfo=Gcatfile(otfile),i_obs=i_obs))

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

     Parameter
     ---------
     filename : string
         log file name including extension type (eg. 'logfile2018-01-01.log')

     obs : 'gemini_classes.Gobservations'
         Observation information object
    """

    progname, rr, ri, rc = np.unique(obs.prog_ref, return_index=True, return_inverse=True, return_counts=True)
    prog_comp_time = 0. * u.h  # total time of completed programs
    prog_start_time = 0. * u.h  # total time of partially completed programs
    obs_start_time = 0. *u.h  # total observed time for partially completed program
    prog_unstart_time = 0. * u.h  # total time of unstarted programs
    num_prog_comp = 0  # number of completed programs
    num_prog_started = 0  # number of partially completed programs
    num_prog_unstarted = 0  # number of not yet started programs

    aprint = '\n {0:<20}{1:<15}{2:<15}'
    with open(filename, 'a') as log:

        log.write('\n\n -----------------------------------------------------------')
        log.write('\n\n ' + description)
        log.write('\n\n'+aprint.format('Program', 'Completion', 'Total time'))
        log.write(aprint.format('-------', '----------', '----------'))
        for i in range(len(rr)):
            jj = np.where(obs.obs_comp[rr[i]:rr[i]+rc[i]]>0.)[0][:]
            if len(jj) > 0: # started or completed obs in program
                prog_time = sum(obs.tot_time[rr[i]:rr[i]+rc[i]])
                sum_obs_time = sum(obs.obs_time[rr[i]:rr[i]+rc[i]])
                frac_comp = sum_obs_time/prog_time
                if frac_comp >=1.:
                    perc_comp = '100%'
                    prog_comp_time = prog_comp_time + prog_time
                    num_prog_comp = num_prog_comp + 1
                else:
                    perc_comp = str((100*frac_comp).round(2))+'%'
                    obs_start_time = obs_start_time + sum_obs_time
                    prog_start_time = prog_start_time + prog_time
                    num_prog_started = num_prog_started + 1
                log.write(aprint.format(progname[i], perc_comp, str(prog_time.round(2))))
            else:
                prog_unstart_time = prog_unstart_time + sum(obs.tot_time[rr[i]:rr[i] + rc[i]])
                num_prog_unstarted = num_prog_unstarted + 1

        tot_prog_time = sum(obs.tot_time)
        tot_obs_time = sum(obs.obs_time)
        log.write('\n\n Number of programs: ' + str(len(progname)))
        log.write('\n Total program time: ' + str(tot_prog_time.round(2)))
        log.write('\n Total time completion: ' + str((100 * tot_obs_time / tot_prog_time).round(2)) + '%')

        log.write('\n\n Total observable time: '+str(semesterinfo['night_time'].round(2)))
        log.write('\n Total scheduled time: ' + str(semesterinfo['used_time'].round(2)))

        log.write('\n\n Completed: ' + str(num_prog_comp))
        log.write('\n Observed time: '+str(prog_comp_time.round(2)))

        log.write('\n\n Partially completed: ' + str(num_prog_started))
        log.write('\n Observed time: ' + str(obs_start_time.round(2)))
        log.write('\n Remaining time: ' + str((prog_start_time-obs_start_time).round(2)))

        log.write('\n\n Not started: ' + str(num_prog_unstarted))
        log.write('\n Remaining time: ' + str(prog_unstart_time.round(2)))

        log.close()
    return

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
        cond_func = condgen.condgen.gauss
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

# ====================== Initialize statistics logfile ========================
statfilename = 'logfile'+current_local.isot+'.log'
# statfilename = 'logfile.log'
_init_logfile(filename=statfilename, catalogfile=otfile, site=site, start=local_start, n_nights=len(count_day),
              dst=dst)

semesterinfo = {'night_time':0.*u.h,'used_time':0.*u.h}
_log_progstats(filename=statfilename, obs=obs, semesterinfo=semesterinfo, description='Initial completion status...')



print('\n\tOuput file: '+statfilename)
# =================================== Begin Queueing ==============================================

for i_day in count_day: #cycle through observation days

    night_start = local_start + i_day # time object for 18:00 local on current night
    night_start_utc = night_start - utc_to_local # time object for 18:00 UTC on current night
    date = str(night_start)[0:10]  # date as string

    print('\n\n\t_______________________ Night of '+date+' _______________________')

    if conddist: # generate random sky conditions from selected distribution
        randcond = cond_func()
    acond = actual_conditions(randcond.iq,randcond.cc,'Any',randcond.wv) # convert actual conditions to decimal values
    print('\n\tSky conditions (iq,cc,wv): {0} , {1} , {2}'.format(acond[0], acond[1], acond[3]))

    # calculate time dependent parameters for observing window
    timeinfo, suninfo, mooninfo, targetinfo = calc_night(obs=obs, site=site, starttime=night_start_utc,
                                                         utc_to_local=utc_to_local)

    ttimer = False
    if ttimer:
        import time as t
        timer = t.time()

    # compute mdist and visible sky brightness using parallel processes.
    mdists = Parallel(n_jobs=10)(delayed(gcirc.gcirc)(mooninfo.ra, mooninfo.dec, target.ra,
                                   target.dec) for target in targetinfo)
    for i in range(len(mdists)):
        targetinfo[i].mdist=mdists[i]
    if ttimer: print('\n\tTime to calc mdist: ', t.time() - timer)

    vsbs = Parallel(n_jobs=10)(delayed(sb.sb)(mpa=mooninfo.phase, mdist=target.mdist, mZD=mooninfo.ZD, ZD=target.ZD, sZD=suninfo.ZD,
                           cc=acond[1]) for target in targetinfo)
    for i in range(len(vsbs)):
        targetinfo[i].vsb = vsbs[i]
    if ttimer: print('\n\tTime to calc vsb: ', t.time() - timer)

    # calculate weights
    weightinfo = calc_weight(site=site, obs=obs, timeinfo=timeinfo, targetinfo=targetinfo, acond=acond)

    # generate PlanInfo plan object and update Gobservation object
    plan, obs = PlanInfo.priority(obs=obs, timeinfo=timeinfo, targetinfo=targetinfo)

    # print plan
    [print(line) for line in plantable(plan=plan, obs=obs, timeinfo=timeinfo, targetinfo=targetinfo)]

    _log_planstats(filename=statfilename, obs=obs, plan=plan, timeinfo=timeinfo, suninfo=suninfo,
                    mooninfo=mooninfo, targetinfo=targetinfo, acond=acond)
    semesterinfo['night_time'] = semesterinfo['night_time'] + plan.night_length
    semesterinfo['used_time'] = semesterinfo['used_time'] + plan.used_time

    # airmass plot to png file
    amplot(plan=plan, timeinfo=timeinfo, mooninfo=mooninfo, targetinfo=targetinfo)

_log_progstats(filename=statfilename, obs=obs, semesterinfo=semesterinfo, description='Schedule results...')


exit()
