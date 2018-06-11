import time as t
timer = t.time()

import argparse
import random
import re
import numpy as np
import textwrap
import astropy.units as u
from astroplan import (download_IERS_A,Observer)
from astropy import (coordinates,time)

# gqpt packages
import condgen
import convconst
from calc_weights import calc_weights
from gemini_classes import Gcatalog2018,Gprogstatus,Gcatfile,Gcondition,Gelevconst
from amplot import amplot
from printplan import printplan
from gemini_schedulers import priority_scheduler

print('\nTime to import packages: ',t.time()-timer)
timer = t.time() #reset timer

seed = random.seed(1000)


#   =============================================== Read and command line inputs =============================================================

verbose = False

aprint = '\t{0:<35s}{1}' #print two strings
bprint = '\t{0:<35s}{1:<.4f}' #print string and number

parser = argparse.ArgumentParser(prog='gqpt.py',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description = textwrap.dedent('''\
                                                                
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
                        
                        -i   --iq               Image quality constraint [DEFAULT=70]
                        -c   --cc               Cloud cover constraint   [DEFAULT=50]
                        -w   --wv               Water vapor constraint   [DEFAULT=Any]

                        -d   --distribution     Generate conditions from distribution [DEFAULT=False]. Accepts the following:
                                                1. 'gaussian' (or 'g')  

                        -u   --update           Download up-to-date IERS(International Earth Rotation and Reference Systems).

                    *****************************************************************************************************                        
                                            '''))

parser.add_argument(action='store',\
                    dest='otfile')

parser.add_argument('-o','--observatory',\
                    action='store',\
                    default='gemini_south')

parser.add_argument('-s','--startdate',\
                    action='store',\
                    default=None)

parser.add_argument('-e','--enddate',\
                    action='store',\
                    default=None)

parser.add_argument('-dst','--daylightsavings',\
                    action='store_true',\
                    dest='dst',\
                    default=False)

parser.add_argument('-i', '--iq',\
                    default='70')

parser.add_argument('-c', '--cc',\
                    default='50')

parser.add_argument('-w', '--wv',\
                    default='Any')

parser.add_argument('-d', '--distribution',\
                    default=None)

parser.add_argument('-u', '--update',\
                    action='store_true',\
                    default=False)

parse = parser.parse_args()
otfile = parse.otfile
site_name = parse.observatory
startdate = parse.startdate
enddate = parse.enddate
dst = parse.dst
distribution = parse.distribution
update = parse.update

if update: #download up to date International Earth Rotation and Reference Systems data
    download_IERS_A()




#   ========================================= Input conditions =====================================================

# Assign condition distribution generator function if selected
conddist = False
if distribution!=None:
    conddist = True
    if distribution=='gaussian' or distribution=='g':
        cond_func = condgen.gauss_cond
    else:
        print('\n\''+str(distribution)+'\' not a valid condition distribution type. Refer to program guide in help using \'-h\' and try again.')
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




#   ======================================== Select observatory =====================================================
#   Set site name for Cerro Pachon or Mauna Kea and initialize Observer object

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
    print('Input error: Could not determine observer location and timezone. Allowed inputs are \'gemini_south\', \'CP\'(Cerro Pachon),\'gemini_north\', or \'MK\'(Mauna Kea).)')
    exit()

site = Observer.at_site(site_name) #create Observer object for observatory site
#can add timezone=timezone_name later if desired (useful if pytz objects are ever used)

# add horizon info to site object
sun_horiz = -.83*u.degree
equat_radius = 6378137.*u.m
site_horiz = -np.sqrt(2.*site.location.height/equat_radius)*(180./np.pi)*u.degree
setattr(site,'horiz',site_horiz)
setattr(site,'sun_horiz',-.83*u.degree)

print('')
print(aprint.format('Site: ',site.name))
print(bprint.format('Height: ',site.location.height))
print(bprint.format('Longitude: ',coordinates.Angle(site.location.lon)))
print(bprint.format('Latitude: ',coordinates.Angle(site.location.lat)))




#   ===================== Read input start/end dates. Create time object for current local and utc time =======================

dform = re.compile('\d{4}-\d{2}-\d{2}') #yyyy-mm-dd format

if startdate == None:
    current_time = time.Time.now() + utc_to_local    
    local_start = time.Time(str(current_time)[0:10] + ' 18:00:00.00') #18:00 local on first night
else:
    if dform.match(startdate): #check startdate format
        local_start = time.Time(startdate + ' 18:00:00.00') #18:00 local on first night
    else:
        print('\nInput error: \"'+startdate+'\" not a valid start date.  Must be in the form \'yyyy-mm-dd\'')
        exit()

if enddate == None: #default number of observation nights is 1
    day_nums = [0]*u.d
else:
    if dform.match(enddate): #check enddate format
        end_time = time.Time(enddate+' 18:00:00.00') #time object of 18:00:00 local time  
        d = int((end_time - local_start).value + 1) #number of days between startdate and enddate
        if d >= 0: 
            day_nums = np.arange(d)*u.d #list of ints with day units e.g. [0*u.d,1*u.d,2*u.d,...]
        else: 
            print('\nInput error: Selected end date \"'+enddate+'\" is prior to the start date.')
            exit()
    else:
        print('\nInput error: \"'+enddate+'\" not a valid end date.  Must be in the form \'yyyy-mm-dd\'')
        exit()

utc_start = local_start - utc_to_local

print('')
print(aprint.format('Schedule start (local): ',local_start.iso))
print(aprint.format('Schedule start (UTC): ',utc_start.iso))
print(bprint.format('Julian date (UTC): ',utc_start.jd))
print(bprint.format('Julian year (UTC): ',utc_start.jyear))
print(aprint.format('Number of nights: ',len(day_nums)))




#   ================================== Read catalog and select observations for queue ================================================

#Simple object containing catalog information and column headers. 
catinfo = Gcatfile(otfile)
 
#Interpret and sort catalog data into appropriate columns(as done in Bryan Miller's IDL version)
otcat = Gcatalog2018(catinfo=catinfo) 

#Get indices of observations for queue
i_obs = np.where(np.logical_and(np.logical_or(otcat.obs_status=='Ready',otcat.obs_status=='Ongoing'),\
    np.logical_or(otcat.obs_class=='Science',\
    np.logical_and(np.logical_or(otcat.inst=='GMOS',otcat.inst=='bHROS'),\
    np.logical_or(otcat.obs_class=='Nighttime Partner Calibration',otcat.obs_class=='Nighttime Program Calibration')))))[0][:] #get indeces of observation to queue

n_obs = len(i_obs)
print('')
print('\t'+str(n_obs)+' observations selected for queue...')




#   ======================== Convert program times to hours, covnert elevation constraints, convert observation conditions.  =====================================
#   Create dictionary structures and empty arrays

elev_const = np.empty(n_obs,dtype={'names':('type','min','max'),'formats':('U20','f8','f8')})

def hms_to_hr(timestring): #convert 'HH:MM:SS' string to hours
    (h, m, s) = timestring.split(':')
    return (np.int(h) + np.int(m)/60 + np.int(s)/3600)

charged_time = np.zeros(n_obs)
planned_exec_time = np.zeros(n_obs)
for i in range(0,n_obs): #cycle through selected observations
    #compute observed/total time, add additional time if necessary
    temp_ct = hms_to_hr(otcat.charged_time[i_obs[i]])
    temp_pet = hms_to_hr(otcat.planned_exec_time[i_obs[i]])
    if (temp_ct>0.):
        if (otcat.disperser[i_obs[i]]=='Mirror'):
            temp_pet = temp_pet + 0.2
        else:
            temp_pet = temp_pet + 0.3
    charged_time[i] = temp_ct
    planned_exec_time[i] = temp_pet

# create elev_const object
elev_const = Gelevconst(otcat.elev_const[i_obs])

# create condition object
cond = Gcondition(iq=otcat.iq[i_obs],cc=otcat.cc[i_obs],bg=otcat.bg[i_obs],wv=otcat.wv[i_obs])

# create program status object
prog_status = Gprogstatus(prog_id=otcat.prog_ref[i_obs], obs_id=otcat.obs_id[i_obs], target=otcat.target[i_obs],\
                        band=otcat.band[i_obs], comp_time=charged_time/planned_exec_time, \
                        tot_time=planned_exec_time*u.h, obs_time=np.zeros(n_obs)*u.h)

if verbose:
    print('charged_time/planned_exec_time',prog_status.comp_time)
    print('charged_time',charged_time*u.h)
    print('planned_exec_time',planned_exec_time*u.h)
    print('elev_const',elev_const)
    print('cond',cond)


    

#   ========================================================= Begin Queueing ======================================================================

print('\nTime to prepare catalog data: ',t.time()-timer)

for i_day in day_nums: #cycle through observation days

    night_start = local_start + i_day # time object for 18:00 local on current night
    night_start_utc = night_start - utc_to_local # time object for 18:00 UTC on current night
    
    print('\n\t___________________________ Night of '+str(night_start)[0:10]+' _______________________________')

    if conddist==True:
        iq,cc,wv = cond_func()

    print('')
    print(aprint.format('Sky conditions (iq,cc,wv): ','{0} , {1} , {2}'.format(iq,cc,wv)))
    actual_cond = [iq,cc,'Any',wv]

    obslist, plan, prog_status = calc_weights(i_day=i_day, i_obs=i_obs,\
                                        n_obs=n_obs, otcat=otcat, site=site, prog_status=prog_status,\
                                        cond=cond, actual_cond=actual_cond, elev_const=elev_const,\
                                        utc_time=night_start_utc,local_time=night_start)
    
    prior_plan, prior_obslist, prior_prog_status = priority_scheduler(i_obs=i_obs, n_obs=n_obs, \
                                                                    obslist=obslist, plan=plan,\
                                                                    prog_status=prog_status, otcat=otcat)

    printplan(i_obs=i_obs, plan=prior_plan, obslist=prior_obslist,\
                prog_status=prior_prog_status,otcat=otcat,time_diff_utc=utc_to_local)

    
    amplot(obslist=prior_obslist,plan=prior_plan,prog_status=prior_prog_status)

    





