import time as t
import argparse
import numpy as np
import astropy.units as u
from astroplan import (download_IERS_A,Observer)
from astropy import (coordinates,time)
from read_cat import Catalog
from gemini_programs import Gemini_programs

#starttime = time.time()
#print('Time to read: ',time.time()-starttime)

#download_IERS_A()
print(t.strftime('%Y-%m-%d'))

#   ========================================= Read inputs ====================================================
#   Accept command line parameters 

parser = argparse.ArgumentParser(description='This is a description of the possible command line parameters, \
    as well as a short blurb on the use of the program.')

parser.add_argument('otfile',\
                    action='store',\
                    help='ascii OT catalog filename (required)')

parser.add_argument('-o',\
                    dest='observatory',\
                    action='store',\
                    default='gemini_south',\
                    help='Observatory site [DEFAULT=\'gemini_south\']')

parser.add_argument('-s',\
                    dest='startdate',\
                    action='store',\
                    default=t.strftime('%Y-%m-%d'),\
                    help='Scheduling start date YYYY-MM-DD [DEFAULT=current]')

parser.add_argument('-dst',\
                    dest='daylightsavings',\
                    action='store_true',\
                    default=False,\
                    help='Set daylight savings time to TRUE [DEFAULT=False]')

# parser.add_argument('--version', action='version', version='%(prog)s 1.0')

parse = parser.parse_args()

otfile = parse.otfile

site_name = parse.observatory

startdate = parse.startdate

dst = parse.daylightsavings

print('Inputs: ',otfile,site_name,startdate,dst)

#   ================================================ Select observatory ==============================================================
#   initialize Observer object and get UTC time for 6pm local time.

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
    print('Could not determine observer location and timezone. Allowed inputs are \'gemini_south\', \'CP\'(Cerro Pachon),\'gemini_north\', or \'MK\'(Mauna Kea).)')
    exit()

site = Observer.at_site(site_name) #can add timezone=timezone_name later if desired. Doing so would be useful if pytz objects are used. 
print('\nSite: ',site.name)
print('Height: ',site.location.height)
print('Longitude: ',coordinates.Angle(site.location.lon))
print('Latitude: ',coordinates.Angle(site.location.lat))

local_1800 = startdate+' 18:00:00.000' #set reference time to startdate at 18:00 local time 
utc_1800 = time.Time(local_1800,scale='utc')-utc_to_local #get corresponding utc time 
print('18:00 local time: ',utc_1800)

#   ================================== Read catalog and select observations to queue ================================================

otcat = Catalog(otfile) #read catalog and store info in object

# Sort observations into groups according to their respective programs
i_program = []
i_program_obs = []
ii = np.arange(len(otcat.prog_ref))
while True:
    i_program.append(ii[0]) #save index of prog. name
    prog_name = otcat.prog_ref[ii[0]] #get prog. name
    i_obs = np.where(otcat.prog_ref == prog_name)[0][:] #get indeces of obs. w/ prog. name
    i_program_obs.append(i_obs) #save indeces of obs.

    ii_to_remove = np.where(np.isin(ii,i_obs,assume_unique=True))[0][:] #remove indeces before next iteration
    ii = np.delete(ii,ii_to_remove)
    if len(ii)==0:
        break

# Store programs in class object. Currently stores lists of observation names - will change later to store lists of class objects. 
programs = Gemini_programs(gemprgid=otcat.prog_ref[i_program], \
                            partner=otcat.partner[i_program], \
                            pi=otcat.pi[i_program], \
                            allocated_time=otcat.planned_exec_time[i_program], \
                            charged_time=otcat.charged_time[i_program], \
                            partner_time=otcat.planned_exec_time[i_program], \
                            active=False, \
                            progstart=None, \
                            progend=None, \
                            completed=False, \
                            too_status=1, \
                            scirank=otcat.band[i_program], \
                            obs=[otcat.obs_id[ii] for ii in i_program_obs])









