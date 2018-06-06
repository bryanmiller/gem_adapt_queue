import time as t
import numpy as np
import copy 
import calc_ZDHA
import gcirc
import sb
import astropy.units as u
from convconst import convcond
from obsweight import obsweight
from matplotlib import pyplot as plt
from astropy import (coordinates,time)

# starttime = t.time() #time program runtime




def airmasscalc(ZD):
    AM = np.full(len(ZD),20.)
    ii = np.where(ZD < 87.*u.deg)[0][:]
    sec_z = 1. / np.cos(ZD[ii])
    AM[ii] = sec_z - 0.0018167 * ( sec_z - 1 ) - 0.002875 * ( sec_z - 1 )**2 - 0.0008083 * ( sec_z - 1 )**3 #compute moon AMs
    return AM




def calc_weights(i_day,i_obs,n_obs,otcat,site,prog_status,cond,actual_cond,\
                elev_const,utc_time,local_time):

    verbose = False
    fprint = '\t{0:<34s}' #print format   



#   =================================== Get required times/quantities for current night ==================================================
    
    sun_horiz = -.83*u.degree
    equat_radius = 6378137.*u.m

    #horizon angle for the observation site
    site_horiz = -np.sqrt(2.*site.location.height/equat_radius)*(180./np.pi)*u.degree

    #sunset/sunrise times
    sunset_tonight = site.sun_set_time(utc_time,which='nearest',horizon=sun_horiz-site_horiz)
    sunrise_tonight = site.sun_rise_time(utc_time,which='next',horizon=sun_horiz-site_horiz)
    

    #nautical twilight times (sun at -12 degrees)
    evening_twilight = site.twilight_evening_nautical(utc_time,which='nearest')
    morning_twilight = site.twilight_morning_nautical(utc_time,which='next')
    

    #utc time at local midnight 
    #utc_midnight = time.Time(str(utc_time)[0:10]+' 00:00:00.000')+1.*u.d #get utc midnight
    solar_midnight = site.midnight(utc_time,which='nearest') #get local midnight in utc time
    

    #get location of moon at local midnight
    moon_at_midnight = coordinates.get_moon(solar_midnight, location=site.location, ephemeris=None)
    moon_fraction = site.moon_illumination(solar_midnight)
    moon_phase = site.moon_phase(solar_midnight)
    

    #moonrise/moonset times
    moonrise_tonight = site.moon_rise_time(utc_time,which='nearest',horizon=sun_horiz-site_horiz)
    moonset_tonight = site.moon_set_time(utc_time,which='next',horizon=sun_horiz-site_horiz)
    

    if verbose:
        print(fprint.format('Site: long,lat'),site.location.lon,site.location.lat)
        print(fprint.format('Horizon angle: '),site_horiz)

    print('')    
    print(fprint.format('Solar midnight(UTC): '),solar_midnight.iso)  
    print(fprint.format('Sun set/rise(UTC): '),sunset_tonight.iso,',',sunrise_tonight.iso)
    print(fprint.format('Evening/morning twilight(UTC): '),evening_twilight.iso,',',morning_twilight.iso)
    print(fprint.format('Moon rise/set time: '),moonrise_tonight.iso,',',moonset_tonight.iso)
    print(fprint.format('Moon location (ra,dec): '),moon_at_midnight.ra,',',moon_at_midnight.dec)
    print(fprint.format('Fraction of moon illuminated: '),'{:.2f}'.format(moon_fraction))
    print(fprint.format('Moon phase angle: '),'{:.2f}'.format(moon_phase))




#   ====================== Define 1/10hr time increments between twilights and get corresponding LST times. ===========================   
    
    #create array of astropy.time objects at 1/10hr timesteps
    dt = 0.1*u.h #set 1/10 hr time steps
    tot_nighttime = (morning_twilight - evening_twilight)*24./u.d #get number of hours between twilights
    n_timesteps = int(round(float(tot_nighttime)*10,1))+1 #get number of timesteps between twilights
    step_num = np.arange(0,n_timesteps) #assign integer to each timestep
    timesteps = (step_num*dt+evening_twilight) #get utc time objects for night.


    if verbose: print('\nComputing local sidereal times...')
    lst = site.local_sidereal_time(timesteps)#,coordinates.Angle(longitude=site.location.lon))
        

    if verbose:
        print('Total hours between twilights:',tot_nighttime*u.h) 
        print('Number of time steps:',n_timesteps)
        print('step_num',step_num)
        print('UTC times:',timesteps.iso)
        print('lst throughout night:',lst)




#   =================================== Get time dependent quanitites for Sun & Moon ==================================================   

    if verbose: print('\nComputing moon data...')
    moon = coordinates.get_moon(timesteps, location=site.location) #get moon position at all times between twilights
    if verbose: 
        print('Moon ra:',moon.ra)
        print('Moon dec:',moon.dec)
        

    mZD,mHA,mAZ = calc_ZDHA.calc_ZDHA(lst=lst,longitude=site.location.lon,latitude=site.location.lat,ra=moon.ra,dec=moon.dec) #compute zenith distance, hour angle, and azimuth  


    ii = np.where(mHA>12.0*u.hourangle)[0][:] #adjust range of moon hour angles
    mHA[ii]=mHA[ii]-24.*u.hourangle


    if verbose: print('\nComputing mAM...')
    mAM = airmasscalc(mZD)


    if verbose:
        print('mZD',mZD)
        print('mHA',mHA)
        print('mAZ',mAZ)
        print('mAM',mAM)
        # print('sec_z',sec_z)


    if verbose: print('\nComputing sun positions...')
    sun = coordinates.get_sun(timesteps) #get sun position at all times between twilights
    sZD,sHA,sAZ = calc_ZDHA.calc_ZDHA(lst=lst,longitude=site.location.lon,latitude=site.location.lat,ra=sun.ra,dec=sun.dec) #compute zenith distance, hour angle, and azimuth


    if verbose:
        print('sZD',sZD)
        print('sHA',sHA)
        print('sAZ',sAZ)




#   ============================================ Initialize structures ======================================================================


    # create plan dictionary.
    plan = {'type':'Priority schedule',\
            'date':str(local_time)[0:10],\
            'dt':dt,\
            'n_timesteps':n_timesteps,\
            #'jd':jul_date,\
            #'epoch':eqcur,\
            #'stmid':stmid,\
            'UT':timesteps,\
            'lst':lst,\
            #'year':year,\
            #'month':month,\
            #'day':day,\
            #'actcond':actcond,\
            'mphase':moon_fraction,\
            'mrise':moonrise_tonight,\
            'mset':moonset_tonight,\
            #'mmidra':mmidra,\
            #'mmiddec':mmiddec,\
            'mZD':mZD,\
            'mAZ':mAZ,\
            'mHA':mHA,\
            'mAM':mAM,\
            'sset':sunset_tonight,\
            'srise':sunrise_tonight,\
            'twieven':evening_twilight,\
            'twimorn':morning_twilight,\
            'sZD':sZD,\
            'isel':np.ones(n_timesteps)}


    # create list of observation dictionaries for current plan.
    obs_dict = {'id':'None',\
            'ra':0.0,\
            'dec':0.0,\
            'ZD':np.zeros(n_timesteps),\
            'HA':np.zeros(n_timesteps),\
            'AZ':np.zeros(n_timesteps),\
            'AM':np.zeros(n_timesteps),\
            'mdist':np.zeros(n_timesteps),\
            'sbcond':np.empty(n_timesteps,dtype='U4'),\
            'weight':np.zeros(n_timesteps),\
            'iobswin':[0,0],\
            'wmax':0.0}
    obslist = []
    [obslist.append(copy.deepcopy(obs_dict)) for ii in range(n_obs)]





#   ============================================ Get histogram ra distribution ======================================================================

    if verbose: print('\nGetting distribution of total observation hours as function of ra...')

    
    #RA distribution of targets
    for i in range(0,n_obs):#correct for current epoch and store coordinates
        coord_j2000 = coordinates.SkyCoord(otcat.ra[i_obs[i]],otcat.dec[i_obs[i]], frame='icrs', unit=(u.deg, u.deg))
        current_epoch = coord_j2000.transform_to(coordinates.FK5(equinox='J'+str(utc_time.jyear)))
        obslist[i]['ra'] = current_epoch.ra #store ra degrees
        obslist[i]['dec'] = current_epoch.dec #store dec degrees


    ras = [obslist[i]['ra'] for i in range(n_obs)]*u.deg    
    bin_edges = [0.,30.,60.,90.,120.,150.,180.,210.,240.,270.,300.,330.,360.]*u.deg


    if verbose: 
        print('Binning target ras by hour angle...')
        print('ras',ras)
        print('bins edges',bin_edges)


    bin_nums = np.digitize(ras,bins=bin_edges) #get ra bin number of each target


    if verbose: 
        print('histogram bin indices',bin_nums)


    # Sum total number of obs. hours in 30deg bins. Divide by mean
    hhra = np.zeros(12)*u.h
    for i in range(0,12):
        ii = np.where(bin_nums==i)[0][:]
        hhra[i] = hhra[i] + sum(prog_status.tot_time[ii])
    hhra = hhra/np.mean(hhra)


    if verbose: print('hhra (total observation time distribution)',hhra)




#   ============================================ Compute time dependent values for all observations ===================================================================

    # print('\nCalculating weights...')
    #Cycle through observations.

    for i in range(0,n_obs):
        
        obslist[i]['id'] = prog_status.obs_id[i]        

        if verbose:
            print('\n\n============= Observation: '+obslist[i]['id']+' =================') 
            print('ra',obslist[i]['ra'])
            print('dec',obslist[i]['dec'])




        #   ===================== Compute target ZD, HA, AZ, AM, mdist throughout night =======================

        obslist[i]['ZD'],obslist[i]['HA'],obslist[i]['AZ'] = calc_ZDHA.calc_ZDHA(lst=lst,longitude=site.location.lon,\
                                                                                latitude=site.location.lat,ra=obslist[i]['ra'],\
                                                                                dec=obslist[i]['dec'])


        ii = np.where(obslist[i]['HA']>12.*u.hourangle)[0][:] #adjust hour angle range
        if len(ii)!=0:
            obslist[i]['HA'][ii]=obslist[i]['HA'][ii]-24.*u.hourangle


        #compute airmass
        obslist[i]['AM'] = airmasscalc(obslist[i]['ZD'])
        

        mdist=gcirc.gcirc(moon.ra,moon.dec,obslist[i]['ra'],obslist[i]['dec']) #get distance of target from moon
        obslist[i]['mdist'] = mdist


        if verbose:
            print('oZD',obslist[i]['ZD'])
            print('oAM',obslist[i]['AM'])
            print('oHA',obslist[i]['HA'])
            print('oAZ',obslist[i]['AZ'])
            print('mdist',obslist[i]['mdist'])
            # print('sec_z',sec_z)




        #   ===================== Get sky brightness throughout night and convert actual conditions =======================

        if verbose: print('Calculating sky brightness...')
        vsb=sb.sb(moon_phase,mdist,mZD,obslist[i]['ZD'],sZD,cond.cc[i]) #get sky background from lunar phase and distance
        sbconds = np.zeros(n_timesteps,dtype='U4')
        

        ii = np.where(vsb <= 19.61)[0][:]
        sbconds[ii] = 'Any'
        ii = np.where(np.logical_and(vsb>=19.61,vsb<=20.78))[0][:]
        sbconds[ii] = '80%' 
        ii = np.where(np.logical_and(vsb>=20.78,vsb<=21.37))[0][:]
        sbconds[ii] = '50%' 
        ii = np.where(vsb>=21.37)[0][:]
        sbconds[ii] = '20%' 
        

        acond = np.empty(n_timesteps,dtype={'names':('iq','cc','bg','wv'),'formats':('f8','f8','f8','f8')})
        aiq = np.repeat(actual_cond[0],n_timesteps)
        acc = np.repeat(actual_cond[1],n_timesteps)
        awv = np.repeat(actual_cond[3],n_timesteps)
        acond['iq'],acond['cc'],acond['bg'],acond['wv'] = convcond(aiq,acc,sbconds,awv)
        

        obslist[i]['sbcond'] = acond['bg']
        

        if verbose:
            print('acond',acond)
            print('vsb',vsb)
            print('sbcond',obslist[i]['sbcond'])




        #   ============================= Compute observation weights ===============================

        # istatus = np.where( prog_status.prog_id == otcat.prog_ref[i_obs[i]] )[0][:] #get obs. indices for current program
        
        ttime = np.round( ( prog_status.tot_time[i] - prog_status.obs_time[i] ) *10 ) /10 #get observation time
        if verbose: 
            print('Prog total time:',prog_status.tot_time[i])
            print('Obs. total time:',prog_status.obs_time[i])
            print('ttime',ttime)


        ii = 0 #reset value
        for j in np.arange(12): #Get corresponding hour angle histogram bin 
            if ((obslist[i]['ra'] >= bin_edges[j]) and (obslist[i]['ra'] <= bin_edges[j+1])): #get ra historgram bin
                ii=j
        

        if (ttime > 0.0):

            temp_cond = {'iq':cond.iq[i],'cc':cond.cc[i],'bg':cond.bg[i],'wv':cond.wv[i]}
            temp_elev = {'type':elev_const.type[i],'min':elev_const.min[i],'max':elev_const.max[i]}

            if verbose: 
                print('Obs. weight inputs...')
                print('dec',obslist[i]['dec'])
                print('latitude',site.location.lat)
                print('band',prog_status.band[i])
                print('user_prior',otcat.user_prior[i_obs[i]])
                print('status',0.)
                print('elevation constraint ',temp_elev)
                print('condition constraints ',temp_cond)
            
            obslist[i]['weight'] = obsweight(cond=temp_cond,dec=obslist[i]['dec'],AM=obslist[i]['AM'],\
                                            HA=obslist[i]['HA'],AZ=obslist[i]['AZ'],band=prog_status.band[i],\
                                            user_prior=otcat.user_prior[i_obs[i]], status=0.,latitude=site.location.lat,\
                                            acond=acond,wind=None,otime=0.,wra=None,elev=temp_elev,starttime=None)
        else:
            obslist[i]['weight'] = np.zeros(n_timesteps)




        #   ===================== Check observation windows =======================

        nttime = np.int(np.round(ttime/dt)) #get number of time steps needed for observation
        if verbose: 
            print('nttime',nttime)

        nobswin=0
        ii = np.where(obslist[i]['weight']>0.)[0][:]
        if len(ii)!=0:
            nobswin = len(ii)
            i_wmax = np.argmax(obslist[i]['weight'])
            
            if verbose: print('i_wmax',i_wmax)
            
            obslist[i]['iobswin'] = [ii[0],ii[-1]]
            obslist[i]['wmax'] = obslist[i]['weight'][i_wmax]
        else:
            obslist[i]['iobswin'] = [-1,-1]
            obslist[i]['wmax'] = -1
            i_wmax = -1

        if verbose:
            print('weight',obslist[i]['weight'])
            print('nobswin ',nobswin)
            print('wmax',obslist[i]['wmax'],i_wmax)


    return obslist, plan, prog_status
#   ============================================ Begin scheduling ===================================================================
    

































