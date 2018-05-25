import argparse
import numpy as np
import astropy.units as u
from astroplan import (download_IERS_A,Observer)
from astropy import (coordinates,time)
from read_cat import Catalog
import convcond
import elevconst
import queueplanner
from gemini_programs import Gemini_programs

#starttime = time.time()
#print('Time to read: ',time.time()-starttime)

#download_IERS_A()

#   =============================================== Read input commands =============================================================
#Accept command line parameters 

parser = argparse.ArgumentParser(description='This is a description of the possible command line parameters, \
    as well as a short blurb on the use of the program.')
parser.add_argument(action='store',dest='otfile',help='ascii OT catalog(required)')
parser.add_argument('-o','--observatory',action='store',dest='site_name',default='gemini_south',help='Observatory site [DEFAULT=\'gemini_south\']')
parser.add_argument('-dst','--daylightsavings',action='store_true',dest='dst',default=False,help='Daylight savings time [DEFAULT=False]')

parse = parser.parse_args()
otfile = parse.otfile
site_name = parse.site_name
dst = parse.dst

#   ================================================ Select observatory ==============================================================
#Set site name for Cerro Pachon or Mauna Kea and initialize Observer object

if np.logical_or(site_name=='gemini_south',site_name=='CP'):
    site_name = 'gemini_south'
    timezone_name = 'Chile/Continental'
    if dst == True:
        time_diff_utc = -3.*u.h
    else:
        time_diff_utc = -4.*u.h 
elif np.logical_or(site_name=='gemini_north',site_name=='MK'):
    site_name = 'gemini_north'
    timezone_name = 'US/Hawaii'
    time_diff_utc = -10.*u.h
else:
    print('Could not determine observer location and timezone. Allowed inputs are \'gemini_south\', \'CP\'(Cerro Pachon),\'gemini_north\', or \'MK\'(Mauna Kea).)')
    exit()

site = Observer.at_site(site_name) #can add timezone=timezone_name later if desired. Doing so would be useful if pytz objects are used. 
print('\nSite: ',site.name)
print('Height: ',site.location.height)
print('Longitude: ',coordinates.Angle(site.location.lon))
print('Latitude: ',coordinates.Angle(site.location.lat))

current_time = time.Time.now()
utc_time = time.Time(current_time,scale='utc',location=(coordinates.Angle(site.location.lon),coordinates.Angle(site.location.lat)))
local_time = utc_time + time_diff_utc
print('\nTime: ',utc_time.iso)
print('Julian date: ',utc_time.jd)
print('Julian year: ',utc_time.jyear)
print('Local time: ',local_time.iso)

#   ================================== Read catalog and select observations to queue ================================================

#read catalog and create object containing observation information. 
#object attributes are named by converting catalog headers to lower case w/ underscores(eg. Obs. ID --> obs_id)
otcat = Catalog(otfile)

#Temporary work around as a conditional for (ra!=0 and dec!=0) conditions. 
ratemp = np.float16(otcat.ra)
dectemp = np.float16(otcat.dec)

#check observations in catalog against criteria and select those ready to queue 
i_obs = np.where(np.logical_and(np.logical_or(otcat.obs_status=='Ready',otcat.obs_status=='Ongoing'),\
    np.logical_or(otcat.obs_class=='Science',\
    np.logical_and(np.logical_or(otcat.inst=='GMOS',otcat.inst=='bHROS'),\
    np.logical_or(otcat.obs_class=='Nighttime Partner Calibration',otcat.obs_class=='Nighttime Program Calibration')))))[0][:] #get indeces of selected observation candidates
n_obs = len(i_obs) #number of ready-for-queue observations
print('\n'+str(n_obs)+' observations readied for queueing...')

#   ======================================================= Make some structures ======================================================================

#intialize dictionaries
conditions = np.empty(n_obs,dtype={'names':('iq','cc','bg','wv'),'formats':('f8','f8','f8','f8')})
prog_status = np.empty(n_obs,dtype={'names':('prog_id','obs_id','target','band','comp_time','tot_time'),'formats':('U30','U30','U60','i8','f8','f8')})
elev_const = np.empty(n_obs,dtype={'names':('type','min','max'),'formats':('U20','f8','f8')})
targets = np.empty(n_obs)
comp_time = np.zeros(n_obs)
tot_time = np.zeros(n_obs)

print('\nConverting elevation constraints...')
for i in range(0,n_obs): #cycle through selected observations
    
    #compute observed/total time, add additional time if necessary
    (h, m, s) = otcat.charged_time[i_obs[i]].split(':')
    temp_comp_time = (int(h) + int(m)/60 + int(s)/3600)
    (h, m, s) = otcat.planned_exec_time[i_obs[i]].split(':')
    temp_tot_time = (int(h) + int(m)/60 + int(s)/3600)
    if (temp_comp_time>0):
        if (otcat.disperser[i_obs[i]]=='Mirror'):
            temp_tot_time = temp_tot_time+0.2
        else:
            temp_tot_time = temp_tot_time+0.3
    comp_time[i] = temp_comp_time
    tot_time[i] = temp_tot_time

    #compute and fill elev_const structure
    elev_const[i]=elevconst.convert(otcat.elev_const[i_obs[i]])

print('\nConverting conditions...')
#convert conditions and fill conditions dictionary
conditions['iq'],conditions['cc'],conditions['bg'],conditions['wv']=convcond.convert_array(otcat.iq[i_obs],otcat.cloud[i_obs],otcat.sky_bg[i_obs],otcat.wv[i_obs])

print('\nStoring program status information...')
#fill prog_status dictionary with program status information of observations selected for queueing
prog_status['prog_id']=otcat.prog_ref[i_obs]
prog_status['obs_id']=otcat.obs_id[i_obs]
prog_status['target']=otcat.target[i_obs]
prog_status['band']=otcat.band[i_obs]
prog_status['comp_time']=comp_time/tot_time
prog_status['tot_time']=tot_time

# print(elev_const)
# print(candidates)
#create elevation constraint structures (use astroplan.AirmassConstraint structure when necessary)
#elev_const[i]=(elevconst.convert(otcat.elev_const[i_obs[i]]))

#   ========================================================= Begin Queueing ======================================================================

print('\nBeginning scheduling...')

for i_day in range(0,1):
    print('\n____________________Night '+str(i_day+1)+'________________________')
    
    #convert actual conditions 
    actual_conditions = ['20%','20%','20%','20%'] 
    actual_conditions = convcond.convert(actual_conditions[0],actual_conditions[1],actual_conditions[2],actual_conditions[3])

    #make plan for single night
    queueplanner.plan_day(i_day,i_obs,n_obs,otcat,site,prog_status,targets,conditions,actual_conditions,elev_const,utc_time,local_time)









