# Matt Bonnyman 17 July 2018
# This module contains the individual weight function components.
# obsweight is the main function.

import numpy as np
import astropy.units as u


def radist(ra, tot_time, obs_time, verbose = False):
    """
    Compute weighting factors for RA distribution of total remaining observation time.
    Observations are binned using 30 degree regions around the celestial sphere.

    Parameters
    ----------
    ra : array of 'astropy.units' degrees
        Right ascensions of all remaining observations in queue

    tot_time : array of 'astropy.units' hours
        Total times of observations

    obs_time : array of 'astropy.units' hours
        Observed times of observations

    Returns
    -------
    array of floats
    """

    bin_edges = [0., 30., 60., 90., 120., 150., 180., 210., 240., 270., 300., 330., 360.] * u.deg

    if verbose:
        print('target ra distribution...')
        print('ra', ra)
        print('tot_time', tot_time)
        print('obs_time', obs_time)
        print('bins edges', bin_edges)

    bin_nums = np.digitize(ra, bins=bin_edges) - 1  # get ra bin index for each target

    if verbose:
        print('histogram bin indices', bin_nums)

    # Sum total observing hours in 30 degree bins
    bin_factors = np.zeros(12) * u.h
    for i in np.arange(0, 12):
        ii = np.where(bin_nums == i)[0][:]
        bin_factors[i] = bin_factors[i] + sum(tot_time[ii] - obs_time[ii])

    if verbose:
        print('Total time (ra distribution)', bin_factors)

    bin_factors = bin_factors / np.mean(bin_factors)

    if verbose:
        print('bin_factors (ra distribution weight)', bin_factors)

    # Generate list of ra weights corresponding to order of observations in obstable
    wra = np.empty(len(ra))  # reset index value
    for j in np.arange(12):  # Get hour angle histogram bin of current target
        wra[np.where(np.logical_and(ra >= bin_edges[j], ra < bin_edges[j + 1]))[0][:]] = bin_factors[j]

    return wra


def cond_match(iq, cc, bg, wv, skyiq, skycc, skywv, skybg, negha, user_prior, verbose = False):
    """
    Match condition constraints to actual conditions:
        - Set cmatch to zero for times where the required conditions
            are worse than the actual conditions.
        - Multiply cmatch by 0.75 at times when the actual image quality
            conditions are better than required.
        - Multiply cmatch by 0.75 at times when the actual cloud conditions
            are better than required.

    Parameters
    ----------
    iq : float
        observation image quality constraint percentile

    cc : float
        observation cloud condition constraint percentile

    bg : float
        observation sky background constraint percentile

    wv : float
        observation water vapour constraint percentile

    skyiq : np.array of float
        sky image quality percentile along time grid

    skycc : np.array of float
        sky cloud condition percentile along time grid

    skywv : np.array of float
        sky water vapour percentile along time grid

    skybg : array of floats
        target sky background percentiles along time grid

    skybg : np.ndarray of floats
        actual sky background conditions converted from sky brightness magnitudes

    negha : boolean
        True if target is visible at negative hour angles.

    Returns
    -------
    cmatch : array of floats
        cmatch weights
    """

    cmatch = np.ones(len(skybg))

    # Where actual conditions worse than requirements
    bad_iq = skyiq > iq
    bad_cc = skycc > cc
    bad_bg = skybg > bg
    bad_wv = skywv > wv

    # Multiply weights by 0 where actual conditions worse than required .
    i_bad_cond = np.where(np.logical_or(np.logical_or(bad_iq, bad_cc), np.logical_or(bad_bg, bad_wv)))[0][:]
    cmatch[i_bad_cond] = 0.

    # Multiply weights by 0.75 where iq better than required and target
    # does not set soon and not a ToO. Effectively drop one band.
    i_better_iq = np.where(skyiq < iq)[0][:]
    if len(i_better_iq) != 0 and negha and 'Target of Opportunity' not in user_prior:
        cmatch = cmatch * 0.75

    # Multiply weights by 0.75 where cc better than required and target
    # does not set soon and is not a ToO. Effectively drop one band.
    i_better_cc = np.where(skycc < cc)[0][:]
    if len(i_better_cc) != 0 and negha and 'Target of Opportunity' not in user_prior:
        cmatch = cmatch * 0.75

    if verbose:
        print(iq, cc, bg, wv)
        print(skyiq, skycc, skybg, skywv)
      #  print('iq worse than required', bad_iq)
      #  print('cc worse than required', bad_cc)
      #  print('bg worse than required', bad_bg)
      #  print('wv worse than required', bad_wv)
      #  print('i_bad_cond', i_bad_cond)
      #  print('iq better than required', i_better_iq)
      #  print('cc better than required', i_better_cc)

    return cmatch


def total_cond(iq, cc, bg, wv):
    """
    Returns a weighting factor representative of the quality of conditions required to execute the observation.

    twcond = (1./cond['iq'])**3 + (1./cond['cc'])**3 + (1./cond['bg'])**3 + (1./cond['wv'])**3

    Parameters
    ----------
    iq : float
        observation image quality constraint percentile

    cc : float
        observation cloud condition constraint percentile

    bg : float
        observation sky background constraint percentile

    wv : float
        observation water vapour constraint percentile

    Returns
    -------
    twcond : float
        total conditions weight
    """
    return (1./iq)**3 + (1./cc)**3 + (1./bg)**3 + (1./wv)**3


def airmass(am, ha, elev):
    """
    Compute airmass weights:
        - 0. if airmass is greater than 2.
        - 0. if elevation constraint not satisfied.

    Parameters
    ----------
    am : array of floats
        target airmass at times throughout observing window.

    ha : array of 'astropy.units' hourangles
        target hour angles along time grid

    elev : dictionary
        observation elevation constraint.  Keys 'type', 'min', and 'max'.

    Returns
    -------
    wam : array of floats
        airmass weights
    """

    wam = np.ones(len(am))
    i_bad_AM = np.where(am > 2.1)[0][:]
    wam[i_bad_AM] = 0.

    if elev['type'] == 'Airmass':
        i_bad_elev = np.where(np.logical_or(am < elev['min'], am > elev['max']))[0][:]
        wam[i_bad_elev] = 0.
    elif elev['type'] == 'Hour Angle':
        i_bad_elev = np.where(np.logical_or(ha < elev['min'], ha > elev['max']))[0][:]
        wam[i_bad_elev] = 0.

    return wam


def windconditions(dir, vel, az, verbose = False):
    """
    Wind condition weights:
        - 0. if wind speed is greater than 10km/h
            AND the telescope is pointed within 20deg of the wind direction.

    Parameters
    ----------
    az : np.array of 'astropy.units' degrees
        target azimuth angles along time grid

    dir : np.array of 'astropy.units' degrees
        wind direction along time grid

    vel : np.array of 'astropy.units' m/s
        wind velocity along time grid

    Return
    -------
    wwind : array of floats
        wind condition weights
    """

    if verbose:
        print('Wind vel:', vel)
        print('Wind dir:', dir)
        print('AZ', az)

    wwind = np.ones(len(az))
    ii = np.where(np.logical_and(vel > 10.*u.m/u.s,
                                 np.logical_or(abs(az - dir) <= 20.*u.deg, 360.*u.deg - abs(az - dir) <= 20.*u.deg)))[0][:]
    if len(ii) != 0:
        wwind[ii] = 0.

    if verbose:
        print('ii ((vel > 10.*u.m/u.s) and (abs(dir - az) < 20.*u.deg))', ii)
        print('wwind', wwind)

    return wwind


def hourangle(latitude, dec, ha, verbose = False):
    """
    Compute a weight representing the target location and visibility window.

    Parameters
    ----------
    latitude : '~astropy.coordinates.angles.Latitude' or '~astropy.units'
        observatory latitude

    dec : '~astropy.units' degree
        target declination

    ha : np.ndarray of '~astropy.units' hourangle
        target hour angles along time grid

    Return
    -------
    wha : float array
        hourangle weights
    """

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
    wha = c[0] + c[1] / u.hourangle * ha + c[2] / (u.hourangle ** 2) * ha ** 2
    ii = np.where(wha <= 0)[0][:]
    wha[ii] = 0.

    if np.amin(ha) >= -1. * u.hourangle:
        wha = wha * 1.5
        if verbose:
            print('multiplied wha by 1.5')

    if verbose:
        print('wdec', wdec)
        print('lat', latitude)
        print('decdiff', decdiff)
        print('HA/unit^2', ha / (u.hourangle ** 2))
      #  print('min HA', np.amin(ha).hour)
        print('min HA', np.amin(ha))

    return wha


def rankingband(band):
    """
    Compute ranking band weight.

    Parameters
    ----------
    band : int
        observation ranking band  (1, 2, 3 or 4)
    """
    return (4. - np.int(band)) * 1000


def userpriority(user_prior):
    """
    Compute user priority weight.

    Parameters
    ----------
    user_prior : string
        observation user priority (Low, Medium, High, or Target of Opportunity)
    """
    if 'Target of Opportunity' in user_prior:
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


def status(prog_comp, obs_comp):
    """
    Compute weighting factor representative of observation and program status.
        - 1.0 if observation and program have not been observed
        - 1.5 if program has been partially observed
        - 2.0 if observation has been partially observed

    Parameters
    ----------
    prog_comp : float
        fraction of program completed.

    obs_comp : float
        fraction of observation completed.

    Returns
    -------
    wstatus : float
        program status weighting factor
    """
    if prog_comp > 0.0:
        wstatus = 1.5
        if obs_comp > 0.0:
            wstatus = 2.0
    else:
        wstatus = 1.
    return wstatus


def complete(prog_comp, obs_comp):
    """
    Observation completion weighting factor.
        - 1.0 if observation not completed
        - 0.0 if observation or program are completed

    Parameters
    ----------
    prog_comp : float
        fraction of program completed.

    obs_comp : float
        fraction of observation completed.

    Returns
    -------
    float
        completion weighting factor
    """
    if obs_comp >= 1. or prog_comp >= 1.:
        return 0
    else:
        return 1


def time_wins(grid_size, i_wins, verbose = False):
    """
    Set weights to 0 if they are not within the observation time windows.

    grid_size : int
        number of spaces in time grid

    i_wins : list of integer pair(s)
        indices of available time windows along time grid.

        Example
        -------
        An observation with 4 time windows within the current night...
        time_wins[i] = [
                        [0, 10],
                        [30, 50],
                        [80,100],
                        [110, 120]
                       ]

    Returns
    -------
    weights : np.array of floats
        new observation weights along time grid.

    """

    if verbose:
        print('i_wins:')
        [print(win) for win in i_wins]

    weights = np.zeros(grid_size)
    indarrays = []
    for win in i_wins:  # get indices spanned by windows
        indarrays.append(np.arange(win[0], win[1]+1))
    indices = np.concatenate(indarrays)
    weights[indices] = 1.

    if verbose:
        print(indices)

    return weights


def obsweight(obs_id, ra, dec, iq, cc, bg, wv, elev_const, i_wins, band, user_prior, AM, HA, AZ, latitude, prog_comp,
              obs_comp, skyiq, skycc, skybg, skywv, winddir, windvel, wra,  verbose = False, debug = False):
    """
    Calculate observation weights.

    Parameters
    ----------
    obs_id : string
        observation identifier (only needed if printing output)

    ra : 'astropy.units' degrees
        observation right ascension

    dec : 'astropy.units' degrees
        observation declination

    iq : float
        observation image quality constraint percentile

    cc : float
        observation cloud condition constraint percentile

    bg : float
        observation sky background constraint percentile

    wv : float
        observation water vapour constraint percentile

    elev_const : dictionary
        observation elevation constraint (type, min, max).

        Example
        -------
        elev_const = {type='Hour Angle', min='-2.00', max='2.00'}

    i_wins : list of integer pair(s)
        indices of observation time window(s) along time grid.

        Example
        -------
        an observation with two time windows would look something like...
        i_wins = [
                  [0,80],
                  [110, 130],
                 ]


    band : int
        observation ranking band (1, 2, 3, 4)

    user_prior : string
        observation user priority ('Low', 'Medium', 'High', 'Target of Opportunity')

    obs_comp : np.array of float
        fraction of observation completed

    AM : np.array of floats
        target airmasses along time grid

    HA : np.array of 'astropy.units' hourangles
        target hour angles along time grid

    AZ : np.array of 'astropy.units' radians
        target azimuth angles along time grid

    skyiq : np.array of float
        sky image quality percentile along time grid

    skycc : np.array of float
        sky cloud condition percentile along time grid

    skywv : np.array of float
        sky water vapour percentile along time grid

    skybg : array of floats
        target sky background percentiles along time grid

    latitude : '~astropy.coordinates.angles.Latitude' or '~astropy.unit.Quantity'
        observatory latitude

    prog_comp : float
        Completion fraction of program

    winddir : np.array of 'astropy.units' degrees
        wind direction along time grid

    windvel : np.array of 'astropy.units' m/s
        wind velocity along time grid

    wra : np.ndarray of floats
        RA time distribution weighting factor

    Returns
    -------
    weights : np.ndarray of floats
    """
    verbose2 = debug  # only show obs. info and final weight

    if verbose or verbose2:
        print(obs_id, ra, dec, iq, cc, bg, wv, elev_const, band, user_prior, obs_comp)

    # -- Match time windows --
    wwins = time_wins(grid_size=len(skyiq), i_wins=i_wins)
    if verbose:
        print('wwins', wwins)

    # -- Matching required conditions to actual --
    cmatch = cond_match(iq=iq, cc=cc, bg=bg, wv=wv, skyiq=skyiq, skycc=skycc, skywv=skywv, skybg=skybg,
                        negha=min(HA) < 0. * u.hourangle, user_prior=user_prior, verbose = verbose)
    if verbose:
        print('iq, cc, bg, wv', iq, cc, bg, wv)
        print('skyiq, skycc, skybg, skywv', skyiq, skycc, skybg, skywv)
        print('cmatch', cmatch)
        print('minHA<0', min(HA) < 0. * u.hourangle)

    # -- Total required conditions --
    twcond = total_cond(iq=iq, cc=cc, bg=bg, wv=wv)
    if verbose:
        print('twcond', twcond)

    # -- Airmass/elevation constraints --
    wam = airmass(am=AM, ha=HA, elev=elev_const)
    if verbose:
        print('AM', AM)
        print('HA.hour', HA)
        print('elev', elev_const)
        print('wam', wam)

    # -- Wind --
    # Wind, do not point within 20deg of wind if over limit
    wwind = windconditions(dir=winddir, vel=windvel, az=AZ, verbose=verbose)
    if verbose:
        print('wwind', wwind)

    # -- Hour Angle / Location  --
    wha = hourangle(latitude=latitude, dec=dec, ha=HA, verbose=verbose)
    if verbose:
        print('wha', wha)

    # -- Band --
    wband = rankingband(band=band)
    if verbose:
        print('wband', wband)

    # -- User Priority --
    wprior = userpriority(user_prior=user_prior)
    if verbose:
        print('wprior', wprior)

    # -- Program/Observation Status --
    wstatus = status(prog_comp=prog_comp, obs_comp=obs_comp)
    if verbose:
        print('wstatus', wstatus)

    # -- Observation completion --
    wcplt = complete(prog_comp=prog_comp, obs_comp=obs_comp)
    if verbose:
        print('wcplt', wcplt)

    # -- Partner Balance --
    wbal = 0.
    if verbose:
        print('wbal', wbal)
        print('wra', wra)

    # if 'Target of Opportunity' in user_prior:  # stop ToOs from dropping a band when sky conditions are good.
    #     cmatch = 1.

    # ****** Final weighting function ******
    weight = (twcond + wstatus * wha + wprior + wband  + wbal + wra) * cmatch * wam * wwind * wcplt * wwins
    if verbose or verbose2:
        print('Total weight', weight)

    return weight
