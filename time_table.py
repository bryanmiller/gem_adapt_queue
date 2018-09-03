# Matt Bonnyman 2018-07-12

import numpy as np
import astropy.units as u
from joblib import Parallel, delayed
from astropy.table import Table, Column


def solar_midnight(site, time, which='nearest'):
    return site.midnight(time, which=which)


def evening_twilight(site, time, which='nearest'):
    return site.twilight_evening_nautical(time, which=which)


def morning_twilight(site, time, which='next'):
    return site.twilight_morning_nautical(time, which=which)


def utc_times(start, end, dt):
    intarray = np.arange(int(((end - start) / dt).value) + 1)
    # print(type(dt))
    # print(intarray)
    # print(start)
    # print((start + dt * intarray).scale)
    return start + dt * intarray

def local_times(utc, utc_to_local):
    return (utc + utc_to_local)


def lst_times(utc, site):
    return site.local_sidereal_time(utc).value


def time_table(site, start, utc_to_local, dt=0.1*u.h, end=None):
    """
    Compute time grid data for nights in scheduling period and store in '~astropy.table.Table' object.

    Example Table
    -------------
    >>> import astropy.units as u
    >>> from astroplan import Observer
    >>> site = Observer.at_site('gemini_south')
    >>> start = '2018-07-01'
    >>> utc_to_local = -4 * u.h
    >>> print(time_table(site, start, utc_to_local)

       date                             utc [117]                                         local [117]

    ---------- ------------------------------------------------------ -----------------------------------------------...
    2018-07-21 '2018-04-01 23:31:02.109' .. '2018-04-02 10:01:02.109' '2018-04-01 19:31:02.109' .. '2018-04-02 06:01:...


                         lst [117]             evening_twilight   morning_twilight   solar_midnight
                         hourangle
       ------- ------------------------------ ----------------- ------------------ -----------------
    ...02.109' 14.2395130239 .. 1.87127303291 2458321.457428859 2458321.9443663936 2458321.700971051

    Parameters
    ----------
    site : 'astroplan.Observer'
        Observer site information

    start : 'astropy.time.core.Time'
        UTC corresponding to 16:00 local time on start date

    end : 'astropy.time.core.Time' or None
        UTC corresponding to 16:00 local time on end date. Default is None.

    dt : 'astropy.units' hours
        Size of time grid spacing [DEFAULT=0.1hr]

    utc_to_local : 'astropy.unit.Quantity'
        Hour difference between utc and local time.

    Returns
    -------
    'astropy.table'
        Table of time data with rows corresponding to nights in scheduling period.

        Columns
        -------
        date : str
            date on evening of observing period

        utc : arrays of str
            UTC times along time grid (string format required for later portion of program)

        local : arrays of str
            local times along time grid (string format required for later portion of program)

        lst : arrays of float (with hourangle quantity)
            local sidereal times along time grid

        twilight_evening : 'astropy.time.core.Time'
            evening UTC 12 degree nautical twilight

        twilight_morning : 'astropy.time.core.Time'
            morning UTC 12 degree nautical twilight

        solar_midnight : 'astropy.time.core.Time'
            UTC solar midnight
    """

    verbose = False

    dt_temp = dt  # Set size of time interval
    dt = (start + dt_temp) - start  # time delta object equivalent to dt

    if end is None:
        n_days = 0
    else:
        n_days = int((end-start).to(u.d).value)

    evenings = [(start + i*u.d) for i in np.arange(n_days+1)]
    daynums = np.arange(0, n_days+1)

    if verbose:
        print('start,end', start, end)
        print('site', site)
        print('utc_to_local', utc_to_local)
        print('n_days', n_days)
        print('daynums', daynums)
        print('evenings',evenings)

    dates = Column([(evening+utc_to_local).iso[0:10] for evening in evenings], name='date')

    solar_midnights = Column(Parallel(n_jobs=10)(delayed(solar_midnight)(site, evening)
                                                 for evening in evenings), name='solar_midnight')

    evening_twilights = Column(Parallel(n_jobs=10)(delayed(evening_twilight)(site, evening)
                                                   for evening in evenings), name='twilight_evening')

    morning_twilights = Column(Parallel(n_jobs=10)(delayed(morning_twilight)(site, evening)
                                                   for evening in evenings), name='twilight_morning')

    utc_list = Parallel(n_jobs=10)(delayed(utc_times)(evening_twilights[i], morning_twilights[i], dt)
                                   for i in daynums)

    local_arrays = Column(Parallel(n_jobs=10)(delayed(local_times)(utc_list[i], utc_to_local)
                                              for i in daynums), name='local')

    lst_arrays = Column(np.array(Parallel(n_jobs=10)(delayed(lst_times)(utc_list[i], site)
                                            for i in daynums)), name='lst', unit='hourangle')

    utc_arrays = Column([utc for utc in utc_list], name='utc')

    if verbose:
        try:
            print(Table((dates, utc_arrays, local_arrays, lst_arrays,
                     evening_twilights, morning_twilights, solar_midnights)))
        except ValueError as e:
            print(e)
            print(' Error: Table could not print. '
                  'Python sometimes can not print sexagesimal notation for hour angles. '
                  'The source of this problem is unknown.')

    return Table((dates, utc_arrays, local_arrays, lst_arrays, evening_twilights, morning_twilights, solar_midnights))


def test_timetable():
    from astropy.time import Time
    from astroplan import Observer

    start = Time('2018-07-01 20:00:00')
    end = Time('2018-07-02 20:00:00')
    utc_to_local = -4.*u.h
    site = Observer.at_site('gemini_south')

    assert isinstance(time_table(site, start, utc_to_local, end=end), Table)

    print('Test successful!')
    return

if __name__=='__main__':
    test_timetable()
