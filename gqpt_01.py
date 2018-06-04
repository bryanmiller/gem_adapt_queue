import argparse
import re
import numpy as np
import textwrap
import astropy.units as u
from astroplan import (download_IERS_A,Observer)
from astropy import (coordinates,time)
from read_cat import Catalog
import convcond
import elevconst
import queueplanner as queueplanner  
from gemini_programs import Gprogram
import convert_input
from amplot import amplot
from printplan import printplan

#starttime = time.time()
#print('Time to read: ',time.time()-starttime)

#   =============================================== Read and command line inputs =============================================================

fprint = '\t{0:<30s}{1}' #print format

parser = argparse.ArgumentParser(prog='gqpt.py',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description = textwrap.dedent('''\
                                                                
                                                                Guide
                    *****************************************************************************************************               

                        otfile                  OT catalog filename.

                        -o   --observatory      Observatory site [DEFAULT='gemini_south']. Accepts the following:
                                                'gemini_north','MK'(Mauna Kea),'gemini_south','CP'(Cerro Pachon).
                        
                        -s   --startdate        Start date 'YYYY-MM-DD' [DEFAULT=tonight]
                        
                        -e   --enddate          End date 'YYYY-MM-DD' [DEFAULT=startdate]. End date must be before
                                                the start date.  If no enddate is provided, a single night will be
                                                scheduled.

                        -dst --daylightsavings  Toggle daylight savings time [DEFAULT=False]
                        
                        -i   --iq               Image quality constraint [DEFAULT=70]
                        -c   --cc               Cloud cover constraint   [DEFAULT=50]
                        -w   --wv               Water vapor constraint   [DEFAULT=Any]

                        -u   --update           Download up-to-date IERS(International Earth Rotation and Reference Systems).

                        -v   --verbose          Toggle verbose mode [DEFAULT=False] 

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

parser.add_argument('-u', '--update',\
                    action='store_true',\
                    default=False)

parser.add_argument('-v', '--verbose',\
                    action='store_true',\
                    default=False)

parse = parser.parse_args()
otfile = parse.otfile
site_name = parse.observatory
dst = parse.dst
verbose = parse.verbose
update = parse.update
startdate = parse.startdate
enddate = parse.enddate

if update: #download up to date International Earth Rotation and Reference Systems data
    download_IERS_A()

#Format condition constraint input
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


#   ================================================ Select observatory ==============================================================
#   Set site name for Cerro Pachon or Mauna Kea and initialize Observer object

# Check observatory input. return timezone, utc-local time difference
site_name,timezone_name,utc_to_local = convert_input.site(site_name=site_name,dst=dst)

site = Observer.at_site(site_name) #initialize Observer object
#can add timezone=timezone_name later if desired (useful if pytz objects are ever used)

print('')
print(fprint.format('Site: ',site.name))
print(fprint.format('Height: ',site.location.height))
print(fprint.format('Longitude: ',coordinates.Angle(site.location.lon)))
print(fprint.format('Latitude: ',coordinates.Angle(site.location.lat)))



#   ===================== Read input start/end dates. Create time object for current local and utc time =======================
local_start, day_nums = convert_input.dates(startdate=startdate,enddate=enddate,utc_to_local=utc_to_local)
utc_start = local_start - utc_to_local

print('')
print(fprint.format('Schedule start (local): ',local_start.iso))
print(fprint.format('Schedule start (UTC): ',utc_start.iso))
print(fprint.format('Julian date (UTC): ',utc_start.jd))
print(fprint.format('Julian year (UTC): ',utc_start.jyear))
print(fprint.format('Number of nights: ',len(day_nums)))



#   ================================== Read catalog and select 'ready' observations for queue ================================================

#read ot catalog and create object containing observation information. 
otcat = Catalog(otfile)
 
#Temporary work around as a conditional for (ra!=0 and dec!=0) conditions. 
ratemp = np.float64(otcat.ra)
dectemp = np.float64(otcat.dec)

#Select observations for queue 
i_obs = np.where(np.logical_and(np.logical_or(otcat.obs_status=='Ready',otcat.obs_status=='Ongoing'),\
    np.logical_or(otcat.obs_class=='Science',\
    np.logical_and(np.logical_or(otcat.inst=='GMOS',otcat.inst=='bHROS'),\
    np.logical_or(otcat.obs_class=='Nighttime Partner Calibration',otcat.obs_class=='Nighttime Program Calibration')))))[0][:] #get indeces of observation to queue

n_obs = len(i_obs)
print('\n'+str(n_obs)+' observations added to queue...')




#   ======================== Convert program times to hours, covnert elevation constraints, convert observation conditions.  =====================================
#   Create dictionary structures and empty arrays
cond = np.empty(n_obs,dtype={'names':('iq','cc','bg','wv'),'formats':('f8','f8','f8','f8')})
prog_status = np.empty(n_obs,dtype={'names':('prog_id','obs_id','target','band','comp_time','tot_time','obs_time'),'formats':('U30','U30','U60','i8','f8','f8','f8')})
elev_const = np.empty(n_obs,dtype={'names':('type','min','max'),'formats':('U20','f8','f8')})
comp_time = np.zeros(n_obs)
tot_time = np.zeros(n_obs)
obs_time = np.zeros(n_obs)

def hms_to_hr(timestring):
    (h, m, s) = timestring.split(':')
    return (int(h) + int(m)/60 + int(s)/3600)

for i in range(0,n_obs): #cycle through selected observations
    
    #compute observed/total time, add additional time if necessary
    temp_comp_time = hms_to_hr(otcat.charged_time[i_obs[i]])
    temp_tot_time = hms_to_hr(otcat.planned_exec_time[i_obs[i]])
    
    if (temp_comp_time>0):
        if (otcat.disperser[i_obs[i]]=='Mirror'):
            temp_tot_time = temp_tot_time+0.2
        else:
            temp_tot_time = temp_tot_time+0.3
    comp_time[i] = temp_comp_time
    tot_time[i] = temp_tot_time

    #compute and fill elev_const structure
    elev_const[i]=elevconst.convert(otcat.elev_const[i_obs[i]])

#convert cond and fill cond dictionary
cond['iq'],cond['cc'],cond['bg'],cond['wv']=convcond.convert_array(otcat.iq[i_obs],otcat.cloud[i_obs],otcat.sky_bg[i_obs],otcat.wv[i_obs])

#fill prog_status dictionary with program status information of observations selected for queueing
prog_status['prog_id']=otcat.prog_ref[i_obs]
prog_status['obs_id']=otcat.obs_id[i_obs]
prog_status['target']=otcat.target[i_obs]
prog_status['band']=otcat.band[i_obs]
prog_status['comp_time']=comp_time/tot_time
prog_status['tot_time']=tot_time
prog_status['obs_time']=obs_time

# print(elev_const)
# print(candidates)


#   ========================================================= Begin Queueing ======================================================================

for i_day in day_nums:

    night_start = local_start + i_day
    night_start_utc = night_start - utc_to_local

    print('\n____________________ Night of '+str(night_start)[0:10]+' ________________________')
    
    actual_cond = [iq,cc,'Any',wv]

    #make plan for single night
    obslist,plan = queueplanner.plan_day(i_day=i_day, i_obs=i_obs,\
                                        n_obs=n_obs, otcat=otcat, site=site, prog_status=prog_status,\
                                        cond=cond, actual_cond=actual_cond, elev_const=elev_const,\
                                        utc_time=night_start_utc, local_time=night_start, verbose=verbose)

    #print plan details
    printplan(i_obs,obslist,plan,prog_status,otcat,utc_to_local)

    #plot airmass
    #amplot(obslist,plan,prog_status)

    





