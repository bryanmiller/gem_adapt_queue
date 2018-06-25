import astropy.units as u
from joblib import Parallel, delayed
from astropy import (coordinates)
from gemini_classes import TimeInfo,SunInfo,MoonInfo,TargetInfo

# import time as t
# starttime = t.time() # runtime clock
# print('Runtime: ',t.time()-starttime) # runtime clock

def _init_target(ra, dec, id, starttime, site, utc_times):
    #coord_j2000 = coordinates.SkyCoord(ra, dec, frame='icrs', unit=(u.deg, u.deg))
    #current_epoch = coord_j2000.transform_to(
    #    coordinates.FK5(equinox='J' + str(starttime.jyear)))  # coordinate for current epoch
    target = TargetInfo(site=site, utc_times=utc_times, name=id, ra=current_epoch.ra,
                        dec=current_epoch.dec)  # initialize targetinfo class
    return target

@u.quantity_input(utc_to_local=u.h)
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


    # Compute time info for observing window, then compute sun and moon values at times.
    timeinfo = TimeInfo(site=site,starttime=starttime,endtime=endtime,dt=dt,utc_to_local=utc_to_local)
    suninfo = SunInfo(site=site, utc_times=timeinfo.utc)
    mooninfo = MoonInfo(site=site, utc_times=timeinfo.utc)
    if verbose:
        print(timeinfo)
        print(suninfo)
        print(mooninfo)

    timer = False
    if timer:
        import time as t
        timerstart = t.time()  # runtime clock
    targetinfo = Parallel(n_jobs=25)(delayed(TargetInfo)(ra=obs.ra[i], dec=obs.dec[i], name=obs.obs_id[i],
                                     site=site, utc_times=timeinfo.utc) for i in range(len(obs.obs_id)))
    if timer: print('\n\tInitialize Targets: ', t.time() - timerstart)  # runtime clock

    # print night details to terminal
    [print(line) for line in timeinfo.table()]
    [print(line) for line in suninfo.table()]
    [print(line) for line in mooninfo.table()]

    return timeinfo, suninfo, mooninfo, targetinfo



