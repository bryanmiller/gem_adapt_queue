import time as t
import numpy as np
import convcond
import calc_ZDHA
import gcirc
import sb
from obsweight import obsweight
from gemini_programs import Gobservation
import astropy.units as u
from matplotlib import pyplot as plt
from astropy import (coordinates,time)

def plan_day(i_day,i_obs,n_obs,otcat,site,prog_status,cond,actual_cond,elev_const,utc_time,local_time):
    
    starttime = t.time() #time program runtime

#   =================================== Get required times/quantities for current night ==================================================
    
    degrad = 57.2957795130823
    sun_horiz = -.83*u.degree
    equat_radius = 6378137.*u.m

    #site location
    longitude = site.location.lon/u.deg
    latitude = site.location.lat/u.deg
    # print('Site: long,lat',longitude,latitude)

    #horizon angle for the observation site
    site_horiz = np.sqrt(2.*site.location.height/equat_radius)*(180./np.pi)*u.degree
    # print('Horizon angle: ',site_horiz)

    #sunset/sunrise times
    sunset_tonight = site.sun_set_time(utc_time,which='nearest',horizon=sun_horiz-site_horiz)
    sunrise_tonight = site.sun_rise_time(utc_time,which='next',horizon=sun_horiz-site_horiz)
    # print('Tonight\'s sun set/rise times (UTC): ',sunset_tonight.iso,',',sunrise_tonight.iso)

    #nautical twilight times (sun at -12 degrees)
    evening_twilight = site.twilight_evening_nautical(utc_time,which='nearest')
    morning_twilight = site.twilight_morning_nautical(utc_time,which='next')
    # print('Tonight\'s evening/morning twilight times (UTC): ',evening_twilight.iso,',',morning_twilight.iso)

    #utc time at local midnight 
    utc_midnight = time.Time(str(utc_time)[0:10]+' 00:00:00.000')+1.*u.d #get utc midnight
    local_midnight = utc_midnight+(utc_time-local_time) #get local midnight in utc time
    # print('UTC at midnight tonight: ',utc_midnight)
    # print('Local midnight tonight(in UTC): ',local_midnight)    

    #get location of moon at local midnight
    moon = coordinates.get_moon(local_midnight, location=site.location, ephemeris=None)
    moon_ra = moon.ra/u.deg
    moon_dec = moon.dec/u.deg
    moon_fraction = site.moon_illumination(local_midnight)
    moon_phase = site.moon_phase(local_midnight)*degrad/u.rad
    # print('Moon location (ra,dec): ',moon.ra,moon.dec)
    # print('Fraction of moon illuminated: ',moon_fraction)
    # print('Moon phase angle (deg): ',moon_phase)

    #moonrise/moonset times
    moonrise_tonight = site.moon_rise_time(utc_time,which='nearest',horizon=sun_horiz-site_horiz)
    moonset_tonight = site.moon_set_time(utc_time,which='nearest',horizon=sun_horiz-site_horiz)
    # print('Moon rise time: ',moonrise_tonight.iso)
    # print('Moon set time: ',moonset_tonight.iso)



#   ====================== Define 1/10hr time increments between twilights and get corresponding LST times. ===========================   

    print('\nCreating array of timesteps between twilights...')
    #create array of astropy.time objects at 1/10hr timesteps
    dt = 0.1 #set 1/10 hr time steps
    tot_nighttime = (morning_twilight - evening_twilight)*24./u.d #get number of hours between twilights
    n_timesteps = int(round(float(tot_nighttime)+0.03,1)*10.) #get number of timesteps between twilights
    step_num = np.arange(1,n_timesteps+1) #assign integer to each timestep
    timesteps = (step_num*dt*u.h+evening_twilight) #get time objects for night.

    lst = np.zeros(n_timesteps) #initlize list of lst times at each timestep
    print('\nNumber of time steps',n_timesteps)

    print('\nComputing local sidereal time throughout night...')
    for i in range(0,n_timesteps): #get lst times throughout night in hours
        temp_lst = timesteps[i].sidereal_time('apparent',coordinates.Angle(longitude*u.deg))
        lst[i]=(float(temp_lst.hour))



#   =================================== Get time dependent quanitites for Sun & Moon ==================================================   

    print('\nComputing moon positions...')
    moon_ra = np.zeros(n_timesteps) #initialize array of moon ras 
    moon_dec = np.zeros(n_timesteps) #initialize array of moon decs
    moon = coordinates.get_moon(timesteps, location=site.location) #get moon position at all times between twilights
    moon_ra = moon.ra/u.deg
    moon_dec = moon.dec/u.deg
    
    mZD,mHA,mAZ = calc_ZDHA.calc_ZDHA(lst,longitude,latitude,moon_ra,moon_dec) #compute zenith distance, hour angle, and azimuth  

    ii = np.where(mHA>12.0)[0][:] #adjust range of moon hour angles
    mHA[ii]=mHA[ii]-24.

    print('\nComputing mAM...')
    mAM = np.full(n_timesteps,20.) #set large initial AM value
    ii = np.where(mZD < 87.)[0][:]
    sec_z = 1. / np.cos(mZD[ii]*u.deg)
    mAM[ii] = sec_z - 0.0018167 * ( sec_z - 1 ) - 0.002875 * ( sec_z - 1 )**2 - 0.0008083 * ( sec_z - 1 )**3 #compute moon AMs
    
    #print('mZD',mZD)
    #print('sec_z',sec_z)
    #print('mAM',mAM)

    print('\nComputing sun positions...')
    sun_ra = np.zeros(n_timesteps) #initialize array of sun ras 
    sun_dec = np.zeros(n_timesteps) #initialize array of sun decs
    sun = coordinates.get_sun(timesteps) #get sun position at all times between twilights
    sun_ra = sun.ra/u.deg
    sun_dec = sun.dec/u.deg
    sZD,sHA,sAZ = calc_ZDHA.calc_ZDHA(lst,longitude,latitude,sun_ra,sun_dec) #compute zenith distance, hour angle, and azimuth
    
    # print('sZD',sZD)
    # print('mZD',mZD)


#   ============================================ Initialize structures ======================================================================

    # actcond = np.repeat({'iq':actual_cond[0],\
    #                         'cc':actual_cond[1],\
    #                         'sb':actual_cond[2],\
    #                         'wv':actual_cond[3]},\
    #                         n_obs)

    # plan = {'jd':jul_date,\
    #         'epoch':eqcur,\
    #         'stmid':stmid,\
    #         'UT':UT,\
    #         'lst':lst,\
    #         'year':year,\
    #         'month':month,\
    #         'day':day,\
    #         'actcond':actcond,\
    #         'mphase':moon_fraction,\
    #         'mrise':moonrise_tonight,\
    #         'mset':moonset_tonight,\
    #         'mmidra':mmidra,\
    #         'mmiddec':mmiddec,\
    #         'mZD':mZD,\
    #         'mAZ':mAZ,\
    #         'mHA':mHA,\
    #         'mAM':mAM,\
    #         'sset':sunset_tonight,\
    #         'srise':sunrise_tonight,\
    #         'twieven':evening_twilight,\
    #         'twimorn':morning_twilight,\
    #         'sZD':sZD,\
    #         'isel':np.ones(n_timesteps)}

    # obs = {'ra':0.0,\
    #         'dec':0.0,\
    #         'ZD':np.zeros(n_timesteps),\
    #         'HA':np.zeros(n_timesteps),\
    #         'AZ':np.zeros(n_timesteps),\
    #         'AM':np.zeros(n_timesteps),\
    #         'mdist':np.zeros(n_timesteps),\
    #         'sbcond':np.empty(n_timesteps,dtype='U4'),\
    #         'weight':np.zeros(n_timesteps),\
    #         'iobswin':None,\
    #         'wmax':0.0},n_obs)

#   ============================================ Get histogram ra distribution ======================================================================

    print('\nGetting distribution of total observation hours as function of ra...')
    temp_ra = np.zeros(n_obs)
    temp_dec = np.zeros(n_obs)
    #RA distribution of targets
    for i in range(0,n_obs):#correct for current epoch and store coordinates
        coord_j2000 = coordinates.SkyCoord(otcat.ra[i_obs[i]],otcat.dec[i_obs[i]], frame='icrs', unit=(u.deg, u.deg))
        current_epoch = coord_j2000.transform_to(coordinates.FK5(equinox='J'+str(utc_time.jyear)))
        temp_ra[i] = float(current_epoch.ra/u.deg) #store ra degrees
        temp_dec[i] =float(current_epoch.dec/u.deg) #store dec degrees
    obs = Gobservation(ra=temp_ra,dec=temp_dec)
    bin_edges = [0.,30.,60.,90.,120.,150.,180.,210.,240.,270.,300.,330.,360.]
    bin_nums = np.digitize(obs.ra,bins=bin_edges) #get ra bin number of each target

    #Sum number of required hours in each 30deg histogram bin, divide by mean
    hhra = np.zeros(12)
    for i in range(0,12):
        ii = np.where(bin_nums==i)[0][:]
        hhra = hhra + sum(prog_status['tot_time'][ii])
    hhra = hhra/np.mean(hhra)



#   ============================================ Begin Scheduling ===================================================================

    print('\nCycling through observations...')
    #Cycle through observations.

    acond = []
    all_ZD = []
    all_HA = []
    all_AZ = []
    all_AM = []
    all_mdist = []
    all_weight = []
    all_i_obs_win = []
    all_wmax = []
    all_i_wmax = []

    for i in range(0,n_obs):
        
        print('\nObservation: '+otcat.obs_id[i_obs[i]])

        ra = float(otcat.ra[i_obs[i]]) #target ra
        dec = float(otcat.dec[i_obs[i]]) #target dec
        

        #   ===================== Compute target ZD, HA, AZ, AM, mdist throughout night =======================

        oZD,oHA,oAZ = calc_ZDHA.calc_ZDHA(lst,longitude,latitude,ra,dec) #compute zentih distance, hour angle, and azimuth of target throughout night.
        ii = np.where(oHA>12.0)[0][:] #adjust hour angle range
        if len(ii)!=0:
            oHA[ii]=oHA[ii]-24.
        all_ZD.append(oZD) #store computed values
        all_HA.append(oHA)
        all_AZ.append(oAZ)

        #compute target AMs for < 87deg
        oAM = np.full(n_timesteps,20.) #set all AM to large initial value
        ii = np.where(oZD < 87.)[0][:]
        sec_z = 1. / np.cos((oZD[ii])*u.deg)
        oAM[ii] = sec_z - 0.0018167 * ( sec_z - 1 ) - 0.002875 * ( sec_z - 1 )**2 - 0.0008083 * ( sec_z - 1 )**3
        all_AM.append(oAM)
        
        # print('oZD',oZD)
        #print('sec_z',sec_z)
        #print('oAM',oAM)

        mdist=gcirc.degrees(moon_ra,moon_dec,ra,dec) #get distance of target from moon
        all_mdist.append(mdist)

        #   ===================== Get sky brightness throughout night and convert actual conditions =======================

        vsb=sb.sb(moon_phase,mdist,mZD,oZD,sZD,cond[i]['cc']) #get sky background from lunar phase and distance
        sbconds = np.zeros(n_timesteps,dtype='U4')
        #print('vsb',vsb)
        ii = np.where(vsb <= 19.61)[0][:]
        sbconds[ii] = 'Any'
        ii = np.where(np.logical_and(vsb>=19.61,vsb<=20.78))[0][:]
        sbconds[ii] = '80%' 
        ii = np.where(np.logical_and(vsb>=20.78,vsb<=21.37))[0][:]
        sbconds[ii] = '50%' 
        ii = np.where(vsb>=21.37)[0][:]
        sbconds[ii] = '20%' 
        #print('sbconds',sbconds)
        
        temp_cond = np.empty(n_timesteps,dtype={'names':('iq','cc','bg','wv'),'formats':('f8','f8','f8','f8')})
        aiq = np.repeat(actual_cond[0],n_timesteps)
        acc = np.repeat(actual_cond[1],n_timesteps)
        asb = sbconds
        awv = np.repeat(actual_cond[3],n_timesteps)
        temp_cond['iq'],temp_cond['cc'],temp_cond['bg'],temp_cond['wv']=convcond.convert_array(aiq,acc,asb,awv)
        acond.append(temp_cond)
        #print('aiq',aiq,'acc',acc,'asb',asb,'awv',awv)
        #print(actcond)

        #   ===================== Compute observation weights =======================

        istatus = np.where( prog_status['prog_id'] == otcat.prog_ref[i_obs[i]] )[0][:] #get obs. indices for current program

        i_obs_win = [0,0]
        time_win = [0,0]
        
        ttime = round( ( prog_status['tot_time'][i] - prog_status['obs_time'][i] ) *10 ) /10 #get observation time
        # print('ttime',ttime)

        ii = 0 #reset value
        for j in np.arange(12): #Get ra histogram bin
            if ((obs.ra[i] >= bin_edges[j]) and (obs.ra[i] <= bin_edges[j+1])): #get ra historgram bin
                ii=j
        if (ttime > 0.0):
            temp_weight = obsweight(cond=cond[i],dec=dec,AM=oAM,HA=oHA,AZ=oAZ,band=prog_status['band'][i],user_prior=otcat.user_prio[i_obs[i]],\
                                    status=0.,latitude=latitude,acond=temp_cond,wind=None,otime=0.,wra=None,elev=elev_const[i],starttime=None,verbose=True,)
            all_weight.append(temp_weight)
        else: 
            temp_weight = np.zeros(n_timesteps)
            all_weight.append(temp_weight)

        #   ===================== Observation windows =======================

        nttime = np.int(round(ttime/dt)) #get number of time steps needed for observation
        # print('nttime',nttime)

        nobswin=0
        ii = np.where(temp_weight>0.)[0][:]
        if len(ii)!=0:
            nobswin = len(ii)
            i_wmax = np.argmax(temp_weight)
            print('i_wmax',i_wmax)
            all_i_obs_win.append([ii[0],ii[-1]])
            all_i_wmax.append(np.argmax(temp_weight))
            all_wmax.append(temp_weight[i_wmax])
        else:
            all_i_obs_win.append([None,None])
            all_i_wmax.append(None)
            all_wmax.append(None)
        # print('weight',temp_weight)
        # print('nobswin ',nobswin)
        # print('wmax',wmax,i_wmax)


    print('Runtime = ',t.time()-starttime) 
