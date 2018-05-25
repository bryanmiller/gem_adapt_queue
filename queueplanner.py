import time as t
import numpy as np
import convcond
import calc_ZDHA
import gcirc
import sb
import astropy.units as u
from astropy import (coordinates,time)

def plan_day(i_day,i_obs,n_obs,otcat,site,prog_status,targets,conditions,actal_conditions,elev_const,utc_time,local_time):
    
    starttime = t.time() #time program runtime

#   =================================== Get required times/quantities for the night ==================================================
    degrad  =   57.2957795130823
    sun_horiz = -.83*u.degree
    equat_radius = 6378137.*u.m

    #site location
    longitude = site.location.lon/u.deg
    latitude = site.location.lat/u.deg
    print('Site: long,lat',longitude,latitude)

    #horizon angle for the observation site
    site_horiz = np.sqrt(2.*site.location.height/equat_radius)*(180./np.pi)*u.degree
    print('Horizon angle: ',site_horiz)

    #sunset/sunrise times
    sunset_tonight = site.sun_set_time(utc_time,which='nearest',horizon=sun_horiz-site_horiz)
    sunrise_tonight = site.sun_rise_time(utc_time,which='next',horizon=sun_horiz-site_horiz)
    print('Tonight\'s sun set/rise times (UTC): ',sunset_tonight.iso,',',sunrise_tonight.iso)

    #nautical twilight times (sun at -12 degrees)
    evening_twilight = site.twilight_evening_nautical(utc_time,which='nearest')
    morning_twilight = site.twilight_morning_nautical(utc_time,which='next')
    print('Tonight\'s evening/morning twilight times (UTC): ',evening_twilight.iso,',',morning_twilight.iso)

    #utc time at local midnight 
    utc_midnight = time.Time(str(utc_time)[0:10]+' 00:00:00.000')+1.*u.d #get utc midnight
    local_midnight = utc_midnight+(utc_time-local_time) #get local midnight in utc time
    print('UTC at midnight tonight: ',utc_midnight)
    print('Local midnight tonight(in UTC): ',local_midnight)    

    #get location of moon at local midnight
    moon = coordinates.get_moon(local_midnight, location=site.location, ephemeris=None)
    moon_ra = moon.ra/u.deg
    moon_dec = moon.dec/u.deg
    moon_fraction = site.moon_illumination(local_midnight)
    moon_phase = site.moon_phase(local_midnight)*degrad/u.rad
    print('Moon location (ra,dec): ',moon.ra,moon.dec)
    print('Fraction of moon illuminated: ',moon_fraction)
    print('Moon phase angle (deg): ',moon_phase)

    #moonrise/moonset times
    moonrise_tonight = site.moon_rise_time(utc_time,which='nearest',horizon=sun_horiz-site_horiz)
    moonset_tonight = site.moon_set_time(utc_time,which='nearest',horizon=sun_horiz-site_horiz)
    print('Moon rise time: ',moonrise_tonight.iso)
    print('Moon set time: ',moonset_tonight.iso)


#   =================================== Compute time dependent quantities ==================================================   

    print('\nCreating array of timesteps between twilights...')
    
    #create array of astropy.time objects at 1/10hr timesteps
    dt = 0.1 #set 1/10 hr time steps
    tot_nighttime = (morning_twilight - evening_twilight)*24./u.d #get number of hours between twilights
    n_timesteps = int(round(float(tot_nighttime)+0.03,1)*10.) #get number of timesteps between twilights
    step_num = np.arange(1,n_timesteps+1) #assign integer to each timestep
    timesteps = (step_num*dt*u.h+evening_twilight) #get time objects for night.

    lst = np.zeros(n_timesteps) #initlize list of lst times at each timestep

    #get position of moon throughout night
    print('\nComputing moon positions...')
    moon_ra = np.zeros(n_timesteps) #initialize array of moon ras 
    moon_dec = np.zeros(n_timesteps) #initialize array of moon decs
    moon = coordinates.get_moon(timesteps, location=site.location) #get moon position at all times between twilights
    moon_ra = moon.ra/u.deg
    moon_dec = moon.dec/u.deg

    #get position of sun throughout night
    print('\nComputing sun positions...')
    sun_ra = np.zeros(n_timesteps) #initialize array of sun ras 
    sun_dec = np.zeros(n_timesteps) #initialize array of sun decs
    sun = coordinates.get_sun(timesteps) #get sun position at all times between twilights
    sun_ra = sun.ra/u.deg
    sun_dec = sun.dec/u.deg
    

#   ============================================ Compute target and distribution information ======================================================================

    #observation structure to hold calculated properties
    obs = np.repeat({'ra':0.0,'dec':0.0,'ZD':np.zeros(n_timesteps),'HA':np.zeros(n_timesteps),\
        'AZ':np.zeros(n_timesteps),'AM':np.zeros(n_timesteps),'mdist':np.zeros(n_timesteps),\
        'sbcond':np.zeros(n_timesteps),'weight':np.zeros(n_timesteps),'iobswin':None,'wmax':0.0},n_obs)

    print('\nComputing local sidereal time throughout night...')
    for i in range(0,n_timesteps): #get lst times throughout night in hours
        temp_lst = timesteps[i].sidereal_time('apparent',coordinates.Angle(longitude*u.deg))
        lst[i]=(float(temp_lst.hour))

    mZD,mHA,mAZ = calc_ZDHA.calc_ZDHA(lst,longitude,latitude,moon_ra,moon_dec) #compute moon zentih distance, hour angle, and azimuth throughout night.
    sZD,sHA,sAZ = calc_ZDHA.calc_ZDHA(lst,longitude,latitude,sun_ra,sun_dec) #compute sun zentih distance, hour angle, and azimuth throughout night.
    print('\nComputing mZD,mHA,mAZ...')
    print('\nComputing sZD,sHA,sAZ...')

    print('\nComputing mAM...')
    ii = np.where(mHA>12.0) #adjust hour angle range
    if len(ii)!=0:
        mHA[ii]=mHA[ii]-24.
    
    #compute moon AMs
    sec_z = 1. / np.cos((mZD < 87.)/degrad)
    mAM = sec_z - 0.0018167 * ( sec_z - 1 ) - 0.002875 * ( sec_z - 1 )**2 - 0.0008083 * ( sec_z - 1 )**3

    print('\nGetting distribution of total observation hours as function of ra...')
    #RA distribution of targets
    for i in range(0,n_obs):
        #correct for current epoch and store coordinates
        coord_j2000 = coordinates.SkyCoord(otcat.ra[i_obs[i]],otcat.dec[i_obs[i]], frame='icrs', unit=(u.deg, u.deg))
        current_epoch = coord_j2000.transform_to(coordinates.FK5(equinox='J'+str(utc_time.jyear)))
        obs[i]['ra']=float(current_epoch.ra/u.deg) #store ra degrees
        obs[i]['dec']=float(current_epoch.dec/u.deg) #store dec degrees
    all_ras = [obs[i]['ra'] for i in range(0,n_obs)]
    bin_edges = [0.,30.,60.,90.,120.,150.,180.,210.,240.,270.,300.,330.,360.]
    bin_nums = np.digitize(all_ras,bins=bin_edges) #get ra bin number of each target

    #Sum number of required hours in each 30deg histogram bin, divide by mean
    hhra = np.zeros(12)
    for i in range(0,12):
        ii = np.where(bin_nums==i)
        hhra = hhra + sum(prog_status['tot_time'][ii])
    hhra = hhra/np.mean(hhra)


#   ============================================ Begin Scheduling ===================================================================
    
    print('\nCycling through observations...')
    #Cycle through observations.
    for i in range(0,n_obs):
        
        print('\nObservation: '+otcat.obs_id[i_obs[i]])

        ra = float(otcat.ra[i_obs[i]]) #target ra
        dec = float(otcat.dec[i_obs[i]]) #target dec
        
        oZD,oHA,oAZ = calc_ZDHA.calc_ZDHA(lst,longitude,latitude,ra,dec) #compute zentih distance, hour angle, and azimuth of target throughout night.
        
        ii = np.where(oHA>12.0) #adjust hour angle range
        if len(ii)!=0:
            oHA[ii]=oHA[ii]-24.

        obs[i]['ZD'] = oZD #store computed values
        obs[i]['HA'] = oHA
        obs[i]['AZ'] = oAZ

        #compute target AMs
        sec_z = 1. / np.cos((obs[i]['ZD'] < 87.)/degrad)
        obs[i]['AM'] = sec_z - 0.0018167 * ( sec_z - 1 ) - 0.002875 * ( sec_z - 1 )**2 - 0.0008083 * ( sec_z - 1 )**3
        
        obs[i]['mdist'] = gcirc.degrees(moon_ra,moon_dec,ra,dec) #get distance of target from moon's location at local midnight

        vsb=sb.sb(moon_phase,obs[i]['mdist'],mZD,obs[i]['ZD'],sZD,conditions[i]['cc']) #get sky background from lunar phase and distance

    print('Runtime = ',t.time()-starttime) 
