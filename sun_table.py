# Matt Bonnyman 2018-07-12

import numpy as np
import astropy.units as u
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from astropy.coordinates import get_sun
from astropy.table import Table, Column

from sun_horizon import sun_horizon
from calc_zd_ha_az import calc_zd_ha_az


def sun_table(latitude, solar_midnight, lst):
    """
    Compute sun data for scheduling period and return '~astropy.table' object.

    Example Table
    -------------
    >>> import astropy.units as u
    >>> from astroplan import Observer
    >>> site = Observer.at_site('gemini_south')
    >>> start = '2018-07-01'
    >>> utc_to_local = -4 * u.h
    >>> timedata = timetable(site, start, utc_to_local)
    >>> print(sun_table(site.location.lat, solar_midnight, timetable['lst']))

          ra           dec                 ZD [118]                       HA [118]                       AZ [118]
         deg           deg                   deg                        hourangle                         rad
    ------------- ------------- ------------------------------ ------------------------------ -----------------------
    113.199439044 21.7248406161 101.871234114 .. 102.617140841 6.10370518784 .. -6.1642605248 5.03174794829 .. 1.2584...

    Parameters
    ----------
    latitude : 'astropy.coordinates.angles.Latitude' or 'astropy.units.quantity.Quantity'
        observatory latitude.

    solar_midnight : 'astropy.time.core.Time'
        array or list of solar_midnights for nights in scheduling period.

    lst : lists or arrays of float
        local sidereal time hour angles along time grids for nights in scheduling period.

    Returns
    -------
    suntable : 'astropy.table.Table'
        Table of Sun data with rows corresponding to nights in scheduling period.

        Columns
        -------
        ra : float (with degree table quantity)
            right ascension at solar midnight on night of scheduling period

        dec : float (with degree table quantity)
            declination at solar midnight on night of scheduling period

        ZD : arrays of float (with degree table quantity)
            zenith distances along time grids.

        HA : arrays of float (with hourangle table quantity)
            hour angles along time grids.

        AZ : arrays of float (with radian table quantity)
            azimuth angles along time grids.
    """

    verbose = False

    i_day = np.arange(len(solar_midnight))
    ncpu = cpu_count()
    sun = Parallel(n_jobs=ncpu)(delayed(get_sun)(solar_midnight[i]) for i in i_day)

    ra = Column([sun[i].ra.value for i in i_day], name='ra', unit='deg')
    dec = Column([sun[i].dec.value for i in i_day], name='dec', unit='deg')

    if verbose:
        print('i_day', i_day)
        print('ra[0], unit', sun[0].ra, sun[0].ra.unit)
        print('ra[0], unit', sun[0].dec, sun[0].dec.unit)
        print('lst', lst[0] * u.hourangle)
        print('site latitude', latitude)
        [print(sun[i].ra.value) for i in i_day]
        [print(sun[i].dec.value) for i in i_day]

    ZDHAAZ = Parallel(n_jobs=ncpu)(delayed(calc_zd_ha_az)(lst=lst[i] * u.hourangle, latitude=latitude,
                                                        ra=sun[i].ra, dec=sun[i].dec) for i in i_day)

    ZD = Column([ZDHAAZ[i][0].value for i in i_day], name='ZD', unit='deg')
    HA = Column([ZDHAAZ[i][1].value for i in i_day], name='HA', unit='hourangle')
    AZ = Column([ZDHAAZ[i][2].value for i in i_day], name='AZ', unit='deg')

    return Table((ra, dec, ZD, HA, AZ))
