# Matt Bonnyman 2018-07-12

import numpy as np
import astropy.units as u
from astropy.time import Time
from joblib import Parallel, delayed
from astropy.table import Table, Column
from astropy.coordinates import get_moon

from airmass import airmass
from sun_horizon import sun_horizon
from calc_zd_ha_az import calc_zd_ha_az


def get_moon_rise_time(site, midnight, horizon):
    return site.moon_rise_time(midnight, horizon=horizon, which='nearest')


def get_moon_set_time(site, midnight, horizon):
    return site.moon_set_time(midnight, horizon=horizon, which='nearest')


def get_moon_fraction(site, midnight):
    return site.moon_illumination(midnight)


def get_moon_phase(site, midnight):
    return site.moon_phase(midnight).value


def moon_table(site, solar_midnight, utc, lst):
    """
    Compute Moon data for scheduling period and return '~astropy.table' object.

    Example Table
    -------------
    >>> import astropy.units as u
    >>> from astroplan import Observer
    >>> site = Observer.at_site('gemini_south')
    >>> start = '2018-07-01'
    >>> utc_to_local = -4 * u.h
    >>> timedata = timetable(site, start, utc_to_local)
    >>> print(moon_table(site, timetable['solar_midnight'].data, timetable['utc'].data, timetable['lst'].data))

        fraction        phase         ra_mid       dec_mid              ra [118]                     dec [118]
                         rad           deg           deg                  deg                           deg
    --------------- ------------- ------------- ------------- ---------------------------- --------------------------
    0.0180504202392 2.87207394213 129.042250359 18.8745316663 124.46942338 .. 133.95157917 19.7627882458 .. 18.410637...

                      ZD [118]                        HA [118]                       AZ [118]
                       deg                          hourangle                         rad
       ------------------------------ ------------------------------- ------------------------------
    ...91.7509852667 .. 118.682184103 5.35237289874 .. -7.54773653324 5.09386841498 .. 1.47422033296

    Parameters
    ----------
    site : '~astroplan.Observer'
        observatory site object

    solar_midnight : '~astropy.time.core.Time' array
        Solar midnight times for nights in scheduling period.

    utc : 'astropy.time.ore.Time' arrays
        UTC time grids for nights in scheduling period in format accepted by '~astropy.time'.

    lst : arrays of floats
        local sidereal time hour angles along time grids of nights in scheduling period.

    Returns
    -------
    '~astropy.table.Table'
        Table of Moon data with rows corresponding to nights in scheduling period.

        Columns
        -------
        fraction : float
            fraction of moon illuminated at solar midnight on nights of scheduling period.

        phase : float (with radian table quantity)
            moon phase angle at solar midnight on nights of scheduling period.

        ra_mid : float (with degree table quantity)
            right ascension at solar midnight on nights of scheduling period.

        dec_mid : float (with degree table quantity)
            declination at solar midnight on nights of scheduling period.

        ra : arrays of float (with degree table quantity)
            right ascensions along time grids.

        dec : arrays of float (with degree table quantity)
            declinations along time grids.

        ZD : arrays of float (with degree table quantity)
            zenith distances along time grids.

        HA : arrays of float (with hourangle table quantity)
            hour angles along time grids.

        AZ : arrays of float (with radian table quantity)
            azimuth angles along time grids.

        AM : arrays of float
            air masses along time grids.
    """

    verbose = False

    i_day = np.arange(len(solar_midnight))

    # sun_horiz = sun_horizon(site)  # angle from zenith at rise/set
    # set = Column(Parallel(n_jobs=10)(delayed(get_moon_set_time)
    #                                  (site, solar_midnight[i], horizon=sun_horiz) for i in i_day), name='set')
    # rise = Column(Parallel(n_jobs=10)(delayed(get_moon_rise_time)
    #                                   (site, solar_midnight[i], horizon=sun_horiz) for i in i_day), name='rise')

    fraction = Column(Parallel(n_jobs=10)(delayed(get_moon_fraction)(site, solar_midnight[i]) for i in i_day),
                      name='fraction')

    phase = Column(Parallel(n_jobs=10)(delayed(get_moon_phase)(site, solar_midnight[i]) for i in i_day), name='phase',
                   unit='rad')

    moon_midnight = Parallel(n_jobs=10)(delayed(get_moon)(solar_midnight[i], location=site.location) for i in i_day)
    ra_mid = Column([moon_midnight[i].ra.value for i in i_day], name='ra_mid', unit='deg')
    dec_mid = Column([moon_midnight[i].dec.value for i in i_day], name='dec_mid', unit='deg')

    moon = Parallel(n_jobs=10)(delayed(get_moon)(Time(utc[i]), location=site.location) for i in i_day)
    ra = Column([moon[i].ra.value for i in i_day], name='ra', unit='deg')
    dec = Column([moon[i].dec.value for i in i_day], name='dec', unit='deg')

    ZDHAAZ = Parallel(n_jobs=10)(delayed(calc_zd_ha_az)(lst=lst[i]*u.hourangle, latitude=site.location.lat,
                                                        ra=moon[i].ra, dec=moon[i].dec) for i in i_day)

    ZD = Column([ZDHAAZ[i][0].value for i in i_day], name='ZD', unit='deg')
    HA = Column([ZDHAAZ[i][1].value for i in i_day], name='HA', unit='hourangle')
    AZ = Column([ZDHAAZ[i][2].value for i in i_day], name='AZ', unit='rad')
    AM = Column([airmass(ZDHAAZ[i][0]) for i in i_day], name='AM')

    if verbose:
        print('i_day', i_day)
        # print(set)
        # print(rise)
        print(fraction)
        print(phase)
        print(ra_mid)
        print(dec_mid)
        print(ra)
        print(dec)
        print(ZD)
        print(HA)
        print(AZ)

    return Table((fraction, phase, ra_mid, dec_mid, ra, dec, ZD, HA, AZ, AM))
