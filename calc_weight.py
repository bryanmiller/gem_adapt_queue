import numpy as np
import re
import astropy.units as u

__all__=[]

def weight_cond_match(cond, acond, negHA):
    """
    Match condition constraints to actual conditions:
        - Set cmatch to zero for times where the required conditions are worse than the actual conditions.
        - Multiply cmatch by 0.75 at times when the actual image quality conditions are better than required.
        - Multiply cmatch by 0.75 at times when the actual cloud conditions are better than required.

    Input
    -------
    cmatch : array of floats
        initial cmatch weights are set to 1.

    cond : dictionary
        required image quality 'iq', cloud conditions 'cc', 'sky background 'bg'
         and water vapor 'wv' conditions for observation (one value for each)

    acond : dictionary
        actual image quality 'iq', cloud conditions 'cc', 'sky background 'bg'
         and water vapor 'wv' at the observing site.  Should have one value for
         iq, cc, and wv.  acond['bg'] is an array of the sky background at all times
         throughout the observing window.

    Return
    -------
    cmatch : array of floats
        New cmatch weights
    """
    verbose = False

    cmatch = np.ones(len(acond['bg']))
    # Where actual conditions worse than requirements
    bad_iq = acond['iq'] > cond['iq']
    bad_cc = acond['cc'] > cond['cc']
    bad_bg = acond['bg'] > cond['bg']
    bad_wv = acond['wv'] > cond['wv']
    i_bad_cond = np.where(np.logical_or(np.logical_or(bad_iq, bad_cc), np.logical_or(bad_bg, bad_wv)))[0][:]

    # Multiply weights by 0 where actual conditions worse than required .
    cmatch[i_bad_cond] = 0.

    # Where actual iq, cc conditions better than required
    better_iq = acond['iq'] < cond['iq']
    better_cc = acond['cc'] < cond['cc']

    # Multiply weights by 0.75 where iq better than required
    i_better_iq = np.where(acond['iq'] < cond['iq'])[0][:]
    if len(i_better_iq) != 0 and negHA:
        cmatch = cmatch * 0.75

    # Multiply weights by 0.75 where cc better than required
    i_better_cc = np.where(acond['cc'] < cond['cc'])[0][:]
    if len(i_better_cc) != 0 and negHA:
        cmatch = cmatch * 0.75

    if verbose:
        print('iq worse than required', bad_iq)
        print('cc worse than required', bad_cc)
        print('bg worse than required', bad_bg)
        print('wv worse than required', bad_wv)
        print('i_bad_cond', i_bad_cond)
        print('iq better than required', better_iq)
        print('cc better than required', better_cc)

    return cmatch

def weight_tot_cond(cond):
    """
    Returns a value representative of the required observing conditions (sum of the reciprocals cubed).

    twcond = (1./cond['iq'])**3 + (1./cond['cc'])**3 + (1./cond['bg'])**3 + (1./cond['wv'])**3

    Input
    -------
    cond : dictionary
        required image quality 'iq', cloud conditions 'cc', 'sky background 'bg'
         and water vapor 'wv' conditions for observation (one value for each)

    Return
    -------
    twcond : float
        total conditions weight
    """
    return (1./cond['iq'])**3 + (1./cond['cc'])**3 + (1./cond['bg'])**3 + (1./cond['wv'])**3

def weight_am(AM, HA, elev):
    """
    Airmass weights:
        - Set wam to 0. at times where the the airmass is greater than 2.
        - Set wam to 0. at times where the elevation constraint is not met

    Input
    -------
    wam : array of floats
        initial wam weights are set to 1.

    AM : array of floats
        Target airmass at times throughout observing window.

    HA : array of 'astropy.units.quantity.Quantity's
         Target hourangle at times throughout observing window.

    Return
    -------
    wam : array of floats
        New wam weights
    """
    wam = np.ones(len(AM))
    i_bad_AM = np.where(AM > 2.)[0][:]
    wam[i_bad_AM] = 0.

    if elev['type']=='Airmass':
        i_bad_elev = np.where(np.logical_or(AM<elev['min'],AM>elev['max']))[0][:]
        wam[i_bad_elev] = 0.
    elif elev['type']=='Hour Angle':
        i_bad_elev = np.where(np.logical_or(HA<elev['min'],HA>elev['max']))[0][:]
        wam[i_bad_elev] = 0.

    return wam

def weight_wind(wind, AZ):
    """
    Wind conditions weights:
        - Set wwind to 0. at times where the wind speed is greater than 10km/h
            AND the telescope is pointed within 20deg of the wind direction.

    Input
    -------
    wwind : array of floats
        initial wwind weights are set to 1.

    wind : array of 'astropy.units.quantity.Quantity'
        wind[0] = wind speed in meters per hour
        wind[1] = wind direction in degrees

    Return
    -------
    wwind : array of floats
        New wwind weights
    """
    wwind = np.ones(len(AZ))
    ii = np.where(np.logical_and(wind[0] > 10.e3 * u.m / u.h, abs(AZ - wind[1]) < 20. * u.deg))[0][:]
    wwind[ii] = 0.
    return wwind

def weight_ra(targetinfo, tot_time, obs_time):
    """
    Compute a weight representing the observation target ra-distribution.

    Input
    -------
    targetinfo : list of 'gemini_classes.TargetInfo'
        list of objects with target information

    Return
    -------
    wra : array of floats
        ra-distribution weights at 30deg intervals
    """
    verbose = False

    ras = np.array([target.ra / u.deg for target in targetinfo]) * u.deg
    bin_edges = [0., 30., 60., 90., 120., 150., 180., 210., 240., 270., 300., 330., 360.] * u.deg

    if verbose:
        print('target ra distribution...')
        print('ras', ras)
        print('bins edges', bin_edges)

    bin_nums = np.digitize(ras, bins=bin_edges) - 1  # get ra bin index for each target

    if verbose:
        print('histogram bin indices', bin_nums)

    # Sum total observing hours in bins and divide mean (wra weight)
    wra = np.zeros(12) * u.h
    for i in np.arange(0, 12):
        ii = np.where(bin_nums == i)[0][:]
        wra[i] = wra[i] + sum(tot_time[ii] - obs_time[ii])
    if verbose: print('Total time (ra distribution)', wra)
    wra = wra / np.mean(wra)

    if verbose: print('wra (ra distribution weight)', wra)
    return wra, bin_edges

def weight_ha(latitude, dec, HA):
    """
    Compute a weight representing the target location and visibility window.

    Input
    -------
    latitude : 'astropy.coordinates.angles.Latitude'
        observer latitude

    dec : 'astropy.coordinates.angles.Latitude'
        target declination

    Return
    -------
    wra : float
        right ascension weighting
    """
    verbose = False

    if latitude < 0:
        decdiff = latitude - dec
    else:
        decdiff = dec - latitude

    declim = [90., -30., -45., -50, -90.] * u.deg
    wval = [1.0, 1.3, 1.6, 2.0]
    wdec = 0.
    for i in np.arange(4):
        if np.logical_and(decdiff < declim[i], decdiff >= declim[i+1]):
            wdec = wval[i]

    # HA - if within -1hr of transit at  twilight it gets higher weight
    if abs(decdiff) < 40. * u.deg:
        c = wdec * np.array([3., 0.1, -0.06])  # weighted to slightly positive HA
    else:
        c = wdec * np.array([3., 0., -0.08])  # weighted to 0 HA if Xmin > 1.3
    wha = c[0] + c[1] / u.hourangle * HA + c[2] / (u.hourangle ** 2) * HA ** 2
    ii = np.where(wha <= 0)[0][:]
    wha[ii] = 0.

    if np.amin(HA) >= -1. * u.hourangle:
        wha = wha * 1.5
        if verbose: print('multiplied wha by 1.5')

    if verbose:
        print('wdec', wdec)
        print('lat', latitude)
        print('decdiff', decdiff)
        print('HA/unit^2', HA / (u.hourangle ** 2))
        print('min HA', np.amin(HA).hour)

    return wha

def weight_band(band):
    """
    Compute ranking band weight

    Input
    -------
    band : int
        ranking band of 1, 2, 3 or 4
    """
    return (4. - np.int(band)) * 1000

def weight_userpriority(user_prior):
    """
    Compute user priority weight

    Input
    -------
    user prior : string
        Low, Medium, High, or Target of Opportunity
    """
    if user_prior == 'Target of Opportunity':
        wprior = 500.
    elif user_prior == 'High':
        wprior = 2.
    elif user_prior == 'Medium':
        wprior = 1.
    elif user_prior == 'Low':
        wprior = 0.
    else:
        wprior = 0.
    return wprior

def weight_status(prstatus, obstatus):
    """
    Compute a weight representing the observation and program completion status.

    Input
    -------
    prstatus : boolean
        True if any observations in program are started or completed

    obstatus : float
        fraction of total observation time completed
    """
    if prstatus:
        wstatus = 1.5
        if obstatus > 0.:
            wstatus = 2.0
    else:
        wstatus = 1.
    return wstatus

def vsb_to_sbcond(vsb):
    """
    Convert visible sky background magnitudes to decimal conditions.

        Conversion scheme:
            1.0 |          vsb < 19.61
            0.8 | 19.61 <= vsb < 20.78
            0.5 | 20.78 <= vsb < 21.37
            0.2 | 21.37 <= vsb


    Input
    -------
    target :  'gemini_classes.TargetInfo'
        TargetInfo object with time dependent vsb quantities

    Return
    -------
    sbcond : array of floats
        sky background condition values
    """

    sbcond = np.zeros(len(vsb), dtype=float)
    ii = np.where(vsb < 19.61)[0][:]
    sbcond[ii] = 1.
    ii = np.where(np.logical_and(vsb >= 19.61, vsb < 20.78))[0][:]
    sbcond[ii] = 0.8
    ii = np.where(np.logical_and(vsb >= 20.78, vsb < 21.37))[0][:]
    sbcond[ii] = 0.5
    ii = np.where(vsb >= 21.37)[0][:]
    sbcond[ii] = 0.2
    return sbcond

def convert_elev(elev_const):
    """
    Convert elevation constraint for observation and return as a dictionary
        of the form {'type':string,'min':float,'max':float}

    Input
    -------
    elev_const : string
        elevation constraint type and limits
        (eg. elev_const = '{Hour Angle -2.00 2.00}')

    Returns
    --------
    dictionary
        (eg. elev = {'type':'Airmass', 'min':0.2, 'max':0.8})
    """
    if (elev_const.find('None') != -1) or (elev_const.find('null') != -1) or (elev_const.find('*NaN') != -1):
        return {'type': 'None', 'min': 0., 'max': 0.}
    elif elev_const.find('Hour') != -1:
        nums = re.findall(r'[+-]?\d+(?:\.\d+)?', elev_const)
        return {'type': 'Hour Angle', 'min': float(nums[0])* u.hourangle, 'max': float(nums[1])* u.hourangle}
    elif elev_const.find('Airmass') != -1:
        nums = re.findall(r'[+-]?\d+(?:\.\d+)?', elev_const)
        return {'type': 'Airmass', 'min': float(nums[0]), 'max': float(nums[1])}
    else:
        raise TypeError('Could not determine elevation constraint from string: ', elev_const)

def obsweight(dec, AM, HA, AZ, band, user_prior, prstatus, latitude,
              cond, acond, obstatus, elev, wind=[0.*u.m/u.s, 0.*u.deg], wra=1.):
    """
    Calculate weights for single observation.

    weighting schemes:
    1. (twcond + wstatus * wha + wband + wprior + wbal + wra) * cmatch * wam * wcplt * wwind

    weights
    --------
    twcond - value representative of required observation conditions
    wstatus - observation and program status (increased if observation or program are partially completed)


    Inputs
    --------
    cond - required sky conditions
    acond - actual sky conditions
    dec  - target declination
    AM   - target airmasses
    HA   - target hour angles
    AZ   - target azimuths
    latitude - observer latitude
    band - ranking band (1, 2, 3, 4)
    user_prior - user priority (Low, Medium, High)
    pstatus - program completion as fraction
    ostatus - observation completion as fraction
    wind - wind conditions (speed (m/s) and direction)
    wra - RA distribution weights
    elev - elevation constraint

    Return
    --------
    weight - array of floats with length nt
    """
    verbose = False

    # Weighting schemes
    weighting1 = True

    # ======================== Matching actual + required conditions ========================
    cmatch = weight_cond_match(cond=cond, acond=acond, negHA=min(HA) < 0.*u.hourangle)
    if verbose:
        print('cond', cond)
        print('acond', acond)
        print('cmatch', cmatch)
        print('minHA<0', min(HA) < 0.*u.hourangle)

    # ===================== Total required conditions =====================
    if weighting1:
        twcond = weight_tot_cond(cond=cond)
        if verbose:
            print('twcond', twcond)
    else:
        twcond = 0.

    # ======================== Airmass/elevation constraints ========================
    if weighting1:
        wam = weight_am(AM=AM, HA=HA, elev=elev)
        if verbose:
            print('AM', AM)
            print('HA.hour', HA.hour)
            print('elev', elev)
            print('wam', wam)
    else:
        wam = np.ones(len(AM))

    # ======================== Wind ========================
    # Wind, do not point within 20deg of wind if over limit
    if weighting1:
        wwind = weight_wind(wind=wind, AZ=AZ)
        if verbose: print('wwind',wwind)
    else:
        wwind = np.ones(len(AZ))

    # ==================== Hour Angle / Location  ====================
    wha = weight_ha(latitude=latitude, dec=dec, HA=HA)
    if verbose: print('wha', wha)

    # ======================== Band ========================
    wband = weight_band(band=band)
    if verbose: print('wband',wband)

    # ======================== User Priority ========================
    wprior = weight_userpriority(user_prior=user_prior)
    if verbose: print('wprior',wprior)

    # ======================== Completion Status ========================
    wstatus = weight_status(prstatus=prstatus, obstatus=obstatus)
    if verbose: print('wstatus',wstatus)

    # ======================== Partner Balance ========================
    wbal = 0.
    if verbose:
       print('wbal',wbal)
       print('wra', wra)

    # ======================== Final weighting formula ========================
    weight = (twcond + wstatus * wha + wband + wprior + wbal + wra) * cmatch * wam * wwind
    if verbose: print('Total weight',weight)

    return weight



def calc_weight(site,obs,timeinfo,targetinfo,acond):

    """
    Calculate weights for multiple observations.

    Parameters
    ----------
    site : 'astroplan.Observer'
        Observing site info

    obs : 'gemini_classes.Gobservations'
        OT catalog info

    timeinfo : 'gemini_classes.TimeInfo'
        Times info for observing window

    targetinfo: 'gemini_classes.TargetInfo'
        Time dependent values for observation targets

    acond : dictionary
        Actual conditions - percentage sky visibility conditions converted to
        decimal values

    Return
    --------
    targetinfo : List of 'gemini_classes.TargetInfo'
        Return list of TargetInfo objects with calculated weights
    """

    verbose = False

    #   ================== ra distribution weights ====================
    wra, bin_edges = weight_ra(targetinfo=targetinfo, tot_time=obs.tot_time, obs_time=obs.obs_time)
    if verbose: print('wra (ra distribution weight)',wra)

    nt = timeinfo.nt
    n_obs = len(obs.obs_id)
    for i in range(n_obs):

        ttime = np.round((obs.tot_time[i] - obs.obs_time[i]) * 10) / 10  # remaining time in observation
        if verbose:
            ii = np.where(obs.prog_ref == obs.prog_ref[i])[0][:]
            ptime = np.round(sum(obs.tot_time[ii] - obs.obs_time[ii]) * 10) / 10
            print('Prog. remaining time (ptime):',ptime)
            print('Obs. remaining time (ttime):',ttime)

        if ttime > 0.:

            ii = 0  # reset index value
            for j in np.arange(12):  # Get hour angle histogram bin of current target
                if bin_edges[j] <= obs.ra[i] < bin_edges[j + 1]:
                    ii = j

            # ========= Program partially completed ==========
            jj = np.where(obs.prog_ref == obs.prog_ref[i])[0][:]
            prstatus = np.any(obs.obs_time[jj] > 0.)

            # ========= convert sky background mag ==========
            sbcond = vsb_to_sbcond(vsb=targetinfo[i].vsb)

            cond = {'iq': obs.iq[i], 'cc': obs.cc[i], 'bg': obs.bg[i], 'wv': obs.wv[i]}
            actualcond = {'iq': acond[0], 'cc': acond[1], 'bg': sbcond, 'wv': acond[3]}
            if verbose:
                print('cond', cond)
                print('acond', actualcond)

            # ========= convert elevation constraint ==========
            elev = convert_elev(elev_const=obs.elev_const[i])

            if verbose:
                print('Obs. id: ',obs.obs_id[i])
                print('dec',targetinfo[i].dec.deg)
                print('latitude',site.location.lat)
                print('band',obs.band[i])
                print('user_prior',obs.user_prior[i])
                print('prog. partially completed (prstatus): ',prstatus)
                print('elevation constraint ',elev)
                print('condition constraints',cond)
                print('actual conditions',acond)

            targetinfo[i].weight = obsweight(dec=targetinfo[i].dec, AM=targetinfo[i].AM,
                                         HA=targetinfo[i].HA, AZ=targetinfo[i].AZ, band=obs.band[i],
                                         user_prior=obs.user_prior[i], prstatus=prstatus,
                                         latitude=site.location.lat, cond=cond, acond=actualcond,
                                         obstatus=obs.obs_time[i], wra=wra[ii], elev=elev)
        else:
            targetinfo[i].weight = np.zeros(nt)


    return targetinfo