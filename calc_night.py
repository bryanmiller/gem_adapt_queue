import astropy.units as u
from astropy import (coordinates,time)
from gemini_classes import TimeInfo,SunInfo,MoonInfo,TargetInfo

# import time as t
# starttime = t.time() # runtime clock
# print('Runtime: ',t.time()-starttime) # runtime clock


def calc_night(obs,site,starttime,endtime=None,dt=0.1,utc_to_local=0.*u.h):
    
    """
    Calculate time dependent information for a given night or similar
    observing window(if shorter than time between nautical twilights).

    Input
    ----------
    obs : gemini_classes.Gobservations
        Gemini observation information

    site : astroplan.Observer
        Observer site object

    starttime : string, astropy.time.core.Time
        Start of observation window.  Observation window
        will begin either at starttime or time of evening
        nautical twilight (whichever occurs second).

    endtime (optional) : string, astropy.time.core.Time
        End of observation window.  Observation window
        will end either at endtime or time of morning
        nautical twilight (whichever occurs first).
        Defaults to morning nautical twilight.

    utc_to_local (optional) : 'astropy.units.quantity.Quantity'
        Number of hours to convert from UTC to local time
        (eg. utc_to_local = -10.*u.h for gemini_north)

    Returns
    ---------
    timeinfo : gemini_classes.TimeInfo
        Observation window time interval info.

    suninfo : gemini_classes.SunInfo
        Sun info at observation window time intervals

    mooninfo : gemini_classes.MoonInfo
        Moon info at observation window time intervals.

    targetinfo : list of gemini_classes.TargetInfo class objects
        Target info at observation window time intervals
    """

    verbose = False

    n_obs = len(obs.obs_id) # number of observations in obs.

    # Compute time info for the night
    timeinfo = TimeInfo(site=site,starttime=starttime,endtime=endtime,dt=dt,utc_to_local=utc_to_local)

    # utc time at solar midnight
    solar_midnight = site.midnight(starttime, which='nearest')  # get local midnight in utc time

    # Compute sun and moon info at times throughout night
    suninfo = SunInfo(site=site, utc_times=timeinfo.utc)
    mooninfo = MoonInfo(site=site, utc_times=timeinfo.utc)

    if verbose:
        print(timeinfo)
        print(suninfo)
        print(mooninfo)

    # cycle through observations. Get target info at times throughout night
    targetinfo = []
    for i in range(0,n_obs):
        coord_j2000 = coordinates.SkyCoord(obs.ra[i],obs.dec[i], frame='icrs', unit=(u.deg, u.deg))
        current_epoch = coord_j2000.transform_to(coordinates.FK5(equinox='J'+str(starttime.jyear))) #coordinate for current epoch
        target = TargetInfo(site=site,utc_times=timeinfo.utc,name=obs.obs_id[i],ra=current_epoch.ra,
                            dec=current_epoch.dec) #initialize targetinfo class
        targetinfo.append(target)
        if verbose: print(target)

    # print info
    timeinfo.table(showall=False)
    suninfo.table(showall=False)
    mooninfo.table(showall=False)

    return timeinfo, suninfo, mooninfo, targetinfo



