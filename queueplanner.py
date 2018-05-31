import time as t
import numpy as np
import scipy
import convcond
import calc_ZDHA
import gcirc
import sb
from intervals import intervals
from obsweight import obsweight
from gemini_programs import Gobservation
import astropy.units as u
from matplotlib import pyplot as plt
from astropy import (coordinates,time)
import copy 

def plan_day(i_day,i_obs,n_obs,otcat,site,prog_status,cond,actual_cond,\
                elev_const,utc_time,local_time,verbose):
    
    starttime = t.time() #time program runtime


#   =================================== Get required times/quantities for current night ==================================================
    
    degrad = 57.2957795130823
    sun_horiz = -.83*u.degree
    equat_radius = 6378137.*u.m

    #site location
    longitude = site.location.lon/u.deg
    latitude = site.location.lat/u.deg

    #horizon angle for the observation site
    site_horiz = np.sqrt(2.*site.location.height/equat_radius)*(180./np.pi)*u.degree

    #sunset/sunrise times
    sunset_tonight = site.sun_set_time(utc_time,which='nearest',horizon=sun_horiz-site_horiz)
    sunrise_tonight = site.sun_rise_time(utc_time,which='next',horizon=sun_horiz-site_horiz)
    

    #nautical twilight times (sun at -12 degrees)
    evening_twilight = site.twilight_evening_nautical(utc_time,which='nearest')
    morning_twilight = site.twilight_morning_nautical(utc_time,which='next')
    

    #utc time at local midnight 
    utc_midnight = time.Time(str(utc_time)[0:10]+' 00:00:00.000')+1.*u.d #get utc midnight
    local_midnight = utc_midnight+(utc_time-local_time) #get local midnight in utc time
    

    #get location of moon at local midnight
    moon = coordinates.get_moon(local_midnight, location=site.location, ephemeris=None)
    moon_ra = moon.ra/u.deg
    moon_dec = moon.dec/u.deg
    moon_fraction = site.moon_illumination(local_midnight)
    moon_phase = site.moon_phase(local_midnight)*degrad/u.rad
    

    #moonrise/moonset times
    moonrise_tonight = site.moon_rise_time(utc_time,which='nearest',horizon=sun_horiz-site_horiz)
    moonset_tonight = site.moon_set_time(utc_time,which='next',horizon=sun_horiz-site_horiz)
    

    if verbose:
        print('Site: long,lat',longitude,latitude)
        print('Horizon angle: ',site_horiz)
        print('Tonight\'s sun set/rise times (UTC): ',sunset_tonight.iso,',',sunrise_tonight.iso)
        print('Tonight\'s evening/morning twilight times (UTC): ',evening_twilight.iso,',',morning_twilight.iso)
        print('UTC at midnight tonight: ',utc_midnight)
    print('Local midnight tonight(in UTC): ',local_midnight)    
    print('Moon location (ra,dec): ',moon.ra,moon.dec)
    print('Fraction of moon illuminated: ',moon_fraction)
    print('Moon phase angle (deg): ',moon_phase)
    print('Moon rise time: ',moonrise_tonight.iso)
    print('Moon set time: ',moonset_tonight.iso)


#   ====================== Define 1/10hr time increments between twilights and get corresponding LST times. ===========================   

    if verbose: print('\nCreating array of timesteps between twilights...')
    #create array of astropy.time objects at 1/10hr timesteps
    dt = 0.1 #set 1/10 hr time steps
    tot_nighttime = (morning_twilight - evening_twilight)*24./u.d #get number of hours between twilights
    n_timesteps = int(round(float(tot_nighttime)+0.03,1)*10.) #get number of timesteps between twilights
    step_num = np.arange(1,n_timesteps+1) #assign integer to each timestep
    timesteps = (step_num*dt*u.h+evening_twilight) #get utc time objects for night.

    lst = np.zeros(n_timesteps) #initlize list of lst times at each timestep
    if verbose: print('\nNumber of time steps',n_timesteps)

    if verbose: print('\nComputing local sidereal time throughout night...')
    for i in range(0,n_timesteps): #get lst times throughout night in hours
        temp_lst = timesteps[i].sidereal_time('apparent',coordinates.Angle(longitude*u.deg))
        lst[i]=(float(temp_lst.hour))

    if verbose: print('lst throughout night:',lst)



#   =================================== Get time dependent quanitites for Sun & Moon ==================================================   

    if verbose: print('\nComputing moon positions...')
    moon_ra = np.zeros(n_timesteps) #initialize array of moon ras 
    moon_dec = np.zeros(n_timesteps) #initialize array of moon decs
    moon = coordinates.get_moon(timesteps, location=site.location) #get moon position at all times between twilights
    moon_ra = moon.ra/u.deg
    moon_dec = moon.dec/u.deg
    
    mZD,mHA,mAZ = calc_ZDHA.calc_ZDHA(lst,longitude,latitude,moon_ra,moon_dec) #compute zenith distance, hour angle, and azimuth  

    ii = np.where(mHA>12.0)[0][:] #adjust range of moon hour angles
    mHA[ii]=mHA[ii]-24.

    if verbose: print('\nComputing mAM...')
    mAM = np.full(n_timesteps,20.) #set large initial AM value
    ii = np.where(mZD < 87.)[0][:]
    sec_z = 1. / np.cos(mZD[ii]*u.deg)
    mAM[ii] = sec_z - 0.0018167 * ( sec_z - 1 ) - 0.002875 * ( sec_z - 1 )**2 - 0.0008083 * ( sec_z - 1 )**3 #compute moon AMs
    
    if verbose:
        print('mZD',mZD)
        print('mHA',mHA)
        print('mAZ',mAZ)
        # print('sec_z',sec_z)
        print('mAM',mAM)

    if verbose: print('\nComputing sun positions...')
    sun_ra = np.zeros(n_timesteps) #initialize array of sun ras 
    sun_dec = np.zeros(n_timesteps) #initialize array of sun decs
    sun = coordinates.get_sun(timesteps) #get sun position at all times between twilights
    sun_ra = sun.ra/u.deg
    sun_dec = sun.dec/u.deg
    sZD,sHA,sAZ = calc_ZDHA.calc_ZDHA(lst,longitude,latitude,sun_ra,sun_dec) #compute zenith distance, hour angle, and azimuth
    
    if verbose:
        print('sZD',sZD)
        print('sHA',sHA)
        print('sAZ',sAZ)





#   ============================================ Initialize structures ======================================================================

    # actcond = np.repeat({'iq':actual_cond[0],\
    #                         'cc':actual_cond[1],\
    #                         'sb':actual_cond[2],\
    #                         'wv':actual_cond[3]},\
    #                         n_obs)

    plan = {#'jd':jul_date,\
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
        obslist[i]['ra'] = float(current_epoch.ra/u.deg) #store ra degrees
        obslist[i]['dec'] = float(current_epoch.dec/u.deg) #store dec degrees

    ras = [obslist[i]['ra'] for i in range(n_obs)]    
    bin_edges = [0.,30.,60.,90.,120.,150.,180.,210.,240.,270.,300.,330.,360.]
    bin_nums = np.digitize(ras,bins=bin_edges) #get ra bin number of each target

    #Sum number of required hours in each 30deg histogram bin, divide by mean
    hhra = np.zeros(12)
    for i in range(0,12):
        ii = np.where(bin_nums==i)[0][:]
        hhra = hhra + sum(prog_status['tot_time'][ii])
    hhra = hhra/np.mean(hhra)




#   ============================================ Compute time dependent values for all observations ===================================================================

    print('\nCalculating weights...')
    #Cycle through observations.

    acond = []

    for i in range(0,n_obs):
        
        if verbose: print('\n\n============= Observation: '+otcat.obs_id[i_obs[i]]+' =================')

        ra = obslist[i]['ra'] #target ra
        dec = obslist[i]['dec'] #target dec
        


        #   ===================== Compute target ZD, HA, AZ, AM, mdist throughout night =======================

        oZD,oHA,oAZ = calc_ZDHA.calc_ZDHA(lst,longitude,latitude,ra,dec) #compute zentih distance, hour angle, and azimuth of target throughout night.
        ii = np.where(oHA>12.0)[0][:] #adjust hour angle range
        if len(ii)!=0:
            oHA[ii]=oHA[ii]-24.
        obslist[i]['ZD'] = oZD #store computed values
        obslist[i]['HA'] = oHA
        obslist[i]['AZ'] = oAZ

        #compute target AMs for < 87deg
        oAM = np.full(n_timesteps,20.) #set all AM to large initial value
        ii = np.where(oZD < 87.)[0][:]
        sec_z = 1. / np.cos((oZD[ii])*u.deg)
        oAM[ii] = sec_z - 0.0018167 * ( sec_z - 1 ) - 0.002875 * ( sec_z - 1 )**2 - 0.0008083 * ( sec_z - 1 )**3
        obslist[i]['AM'] = oAM
        
        mdist=gcirc.degrees(moon_ra,moon_dec,ra,dec) #get distance of target from moon
        obslist[i]['mdist'] = mdist

        if verbose:
            print('ra,dec',ra,dec)
            print('oZD',oZD)
            print('oAM',oAM)
            print('oHA',oHA)
            print('oAZ',oAZ)
            # print('sec_z',sec_z)
            print('mdist',mdist)


        #   ===================== Get sky brightness throughout night and convert actual conditions =======================

        vsb=sb.sb(moon_phase,mdist,mZD,oZD,sZD,cond[i]['cc']) #get sky background from lunar phase and distance
        sbconds = np.zeros(n_timesteps,dtype='U4')
        
        ii = np.where(vsb <= 19.61)[0][:]
        sbconds[ii] = 'Any'
        ii = np.where(np.logical_and(vsb>=19.61,vsb<=20.78))[0][:]
        sbconds[ii] = '80%' 
        ii = np.where(np.logical_and(vsb>=20.78,vsb<=21.37))[0][:]
        sbconds[ii] = '50%' 
        ii = np.where(vsb>=21.37)[0][:]
        sbconds[ii] = '20%' 
        
        
        temp_cond = np.empty(n_timesteps,dtype={'names':('iq','cc','bg','wv'),'formats':('f8','f8','f8','f8')})
        aiq = np.repeat(actual_cond[0],n_timesteps)
        acc = np.repeat(actual_cond[1],n_timesteps)
        asb = sbconds
        awv = np.repeat(actual_cond[3],n_timesteps)
        temp_cond['iq'],temp_cond['cc'],temp_cond['bg'],temp_cond['wv']=convcond.convert_array(aiq,acc,asb,awv)
        
        acond.append(temp_cond)
        obslist[i]['sbcond'] = temp_cond['bg']
        
        if verbose:
            print('vsb',vsb)
            print('sbconds',sbconds)



        #   ===================== Compute observation weights =======================

        istatus = np.where( prog_status['prog_id'] == otcat.prog_ref[i_obs[i]] )[0][:] #get obs. indices for current program
        
        ttime = round( ( prog_status['tot_time'][i] - prog_status['obs_time'][i] ) *10 ) /10 #get observation time
        if verbose: print('ttime',ttime)

        ii = 0 #reset value
        for j in np.arange(12): #Get ra histogram bin
            if ((obslist[i]['ra'] >= bin_edges[j]) and (obslist[i]['ra'] <= bin_edges[j+1])): #get ra historgram bin
                ii=j
        if (ttime > 0.0):
            temp_weight = obsweight(cond=cond[i],dec=obslist[i]['dec'],AM=obslist[i]['AM'],HA=obslist[i]['HA'],AZ=obslist[i]['AZ'],band=prog_status['band'][i],user_prior=otcat.user_prio[i_obs[i]],\
                                    status=0.,latitude=latitude,acond=temp_cond,wind=None,otime=0.,wra=None,elev=elev_const[i],starttime=None)
            obslist[i]['weight'] = temp_weight
        else:
            obslist[i]['weight'] = np.zeros(n_timesteps)


        #   ===================== Observation windows =======================

        nttime = np.int(round(ttime/dt)) #get number of time steps needed for observation
        if verbose: print('nttime',nttime)

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
        # print('weight',temp_weight)
        # print('nobswin ',nobswin)
        # print('wmax',wmax,i_wmax)


#   ============================================ Begin scheduling ===================================================================
    
    print('\nScheduling...\n')
    ntcal = 0
    plan['isel'][:] = -1
    nsel = 0
    ii = np.where(plan['isel'] == -1)[0][:]
    
    while len(ii)!=0:
        
        if verbose: print('\nIteration: ',nsel+1)

        #define min block
        nminuse = 10

        #bin indices of adjacent timeslots
        indx = intervals(ii)

        #get indices of first unscheduled time block 
        iint = ii[np.where(indx==1)[0][:]]
        
        if verbose:
            print('ii:',ii)
            print('iint: ',iint)
            # print('indx',indx)
            # print('iint',iint)

        gow = True
        while gow: #schedule observation with maximum weight in time interval
            maxweight = 0.
            iimax = -1

            for i in np.arange(n_obs):
                if (len(ii) >= 2):
                    if verbose: print(obslist[i]['weight'][iint])
                    i_wmax = np.argmax(obslist[i]['weight'][iint])+iint[0]
                    wmax = obslist[i]['weight'][i_wmax]
                else:
                    wmax = obslist[i]['weight'][iint]

                if (wmax > maxweight):
                    maxweight = wmax
                    iimax = i

                if verbose:
                    print('maxweight',maxweight)
                    print('iimax',iimax)

            if (iimax == -1):
                gow = False
            else:
                if verbose:
                    print('iobswin',obslist[iimax]['iobswin'])
                    print('iint',iint)
                # Time interval limited by observing window or plan window?
                istart = np.maximum(obslist[iimax]['iobswin'][0],iint[0])
                iend = np.minimum(obslist[iimax]['iobswin'][1],iint[-1])
                nobswin = iend - istart + 1

                if verbose:
                    print('nobswin',nobswin)
                    print('Inst: ',otcat.inst[i_obs[iimax]])
                    print('Disperser: ',otcat.disperser[i_obs[iimax]])

                if (otcat.inst[i_obs[iimax]]!='GMOS'):
                    if np.logical_and(otcat.disperser[i_obs[iimax]]!='Mirror',otcat.disperser[i_obs[iimax]]=='null'):
                        ntcal = 3
                        if verbose: print('add ntcall=3')

                # total time remaining in observation, include calibration time
                ttime = round((prog_status['tot_time'][iimax] - prog_status['obs_time'][iimax]) *10. + 0.5 ) /10
                nttime = int(round(ttime / dt) + ntcal) #number of steps

                # Try not to leave little pieces of programs
                # if (nttime - nminuse) <= nminuse:
                #     nminuse = nttime

                #Set weights to zero if observation can be done within window
                if np.logical_or(nttime<nobswin,nobswin>nminuse):
                    gow = False
                else:
                    obslist[iimax]['weight'][iint] = 0.
                    if verbose: print('Block too short to schedule...')

            if verbose:
                print('weight of chosen obs',obslist[iimax]['weight'])
                print('istart',istart)
                print('iend',iend)
                print('ttime',ttime)
                print('nttime',nttime)

        if iimax == -1:
            plan['isel'][iint] = -2
        else:
            # pick optimal observing window by integrating weight function
            if np.logical_and(nttime < nobswin , nttime != 0):
                
                #indices where obs. is already scheduled
                jj = np.where(plan['isel']==iimax)[0][:]

                x = np.arange(nttime,dtype=float)
                maxf = 0.0
                jstart = 0
            
                if nttime > 1:
                    # NOTE: integrates over one extra time slot...
                    # ie. if nttime = 14, then the program will choose 15
                    # x values to do trapz integration (therefore integrating
                    # 14 time slots). 
                    if verbose:
                        print('\nIntegrating max obs. over window...')
                        print('istart',istart)
                        print('iend',iend)
                        print('nttime',nttime)
                        # print('obs weights all: ',obslist[iimax]['weight'])
                        print('j values',np.arange(istart,iend-nttime+2))
                    for j in range(istart,iend-nttime+2):
                        f = scipy.integrate.trapz(obslist[iimax]['weight'][j:j+nttime],x)
                        if verbose:
                            print('j',j)
                            print('obs wieght',obslist[iimax]['weight'][j:j+nttime])
                            print('x',x)
                            print('integral',f)
                        if f>maxf:
                            maxf = f
                            jstart = j

                    jend = jstart + nttime - 1

                else:
                    jstart = np.argmax(obslist[iimax]['weight'][iint])
                    maxf = np.amax(obslist[iimax]['weight'][jstart])
                    jend = jstart + nttime - 1

                if verbose:
                    print('maxf',maxf)    
                    print('jstart',jstart)
                    print('jend',jend)

                if jstart < nminuse:
                    if np.logical_and( plan['isel'][0]==-1 , obslist[iimax]['weight'][0]>0. ):
                        jstart = 0
                        jend = jstart + nttime - 1
                elif (n_timesteps-jend) < nminuse:
                    if np.logical_and( plan['isel'][n_timesteps-1]==-1 , obslist[iimax]['weight'][n_timesteps-1]>0. ):
                        jend = n_timesteps - 1
                        jstart = jend - nttime + 1        

                # dstart = jstart - istart - 1
                # wstart = obslist[iimax]['weight'][istart]
                # dend = iend - jend + 1
                # wend = obslist[iimax]['weight'][iend] 
                # if np.logical_and(dstart < nminuse , dend < nminuse):
                #     if np.logical_and(wstart > wend , wstart > 0.):
                #         jstart = istart
                #         jend = istart + nttime - 1
                #     elif wend > 0.:
                #         jstart = iend - nttime + 1
                #         jend = iend
                # elif np.logical_and(dstart < nminuse , wstart > 0.):
                #     jstart = istart
                #     jend = istart + nttime - 1
                # elif np.logical_and(dend < nminuse , wstart > 0.):
                #     jstart = iend - nttime + 1
                #     jend = iend
                # else:
                #     if (jj[0] < istart):
                #         jstart = istart
                #         jend = istart + nttime - 1
                #     else:
                #         jstart = iend - nttime + 1
                #         jend = iend
                    
                #set weights of scheduled times to negatives
                obslist[iimax]['weight'] = obslist[iimax]['weight'] * -1

            else:
                jstart = istart
                jend = iend
                #set weights of scheduled times to negatives
                obslist[iimax]['weight'] = obslist[iimax]['weight'] * -1

            if verbose:
                print('Final jstart',jstart)
                print('Final jend',jend)
                print('New obs. weights: ',obslist[iimax]['weight'])
            
            #increment number of selected observations
            nsel = nsel + 1

            #set timeslots in plan to index of scheduled observation 
            plan['isel'][jstart:jend] = iimax
            
            if verbose:
                print('Current obs time: ',prog_status['obs_time'][iimax])
                print('Current tot time: ',prog_status['tot_time'][iimax])

            #update time observed
            ntmin = np.minimum(nttime - ntcal , nobswin)
            if verbose:
                print('nttime - ntcal , nobswin: ',nttime - ntcal , nobswin)
                print('ntmin: ',ntmin)

            prog_status['obs_time'][iimax] = prog_status['obs_time'][iimax] + dt*ntmin
            if verbose:
                print('New obs time: ',prog_status['obs_time'][iimax])
                print('Obs tot time: ',prog_status['tot_time'][iimax])

            #update time status for observations in program
            ii_obs = np.where(prog_status['prog_id'] == otcat.prog_ref[i_obs[iimax]])[0][:]
            prog_status['comp_time'][iimax] = prog_status['comp_time'][iimax] + dt*ntmin/prog_status['tot_time'][iimax]
            if verbose: print('New comp time: ',prog_status['comp_time'][iimax])

            #add time to total if observation was not fully completed
            if prog_status['obs_time'][iimax] < prog_status['tot_time'][iimax]:
                prog_status['tot_time'][iimax] = prog_status['tot_time'][iimax] + 0.3

            # #if ntmin == nttime, observation is complete. Set all weights to negative
            # if prog_status['obs_time'][iimax] >= prog_status['tot_time'][iimax]:
            #     if (jstart!=0):
            #         obslist[iimax]['weight'][0:(jstart-1)] = obslist[iimax]['weight'][0:(jstart-1)] * -1
            #     if (jend!=0):
            #         obslist[iimax]['weight'][(jend+1):n_timesteps-1] = obslist[iimax]['weight'][(jend+1):n_timesteps-1] * -1    
        
            # if verbose: 
            # print('Obs. added to program:',prog_status['obs_id'][iimax],obslist[iimax])
        
        if verbose: print('Current program: ',plan['isel'])

        ii = np.where(plan['isel'] == -1)[0][:]

    print('Runtime = ',t.time()-starttime) 

    return obslist,plan

































