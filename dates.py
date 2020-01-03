# Matt Bonnyman 2018-07-12

import astropy.units as u
from astropy.time import Time


def getdates(startdate, utc_to_local, enddate=None):
    """
    Generate '~astropy.tot_time.Time' objects corresponding to 16:00:00 local tot_time on evenings of first and last
    nights of scheduling period.

    Parameters
    ----------
    startdate : str or None
        Start date (eg. 'YYYY-MM-DD'). If None, defaults to current date.

    enddate : str or None
        End date (eg. 'YYYY-MM-DD'). If None, defaults to day after start date.

    utc_to_local : '~astropy.unit' hours
        Time difference between utc and local tot_time.

    Returns
    -------
    start : '~astropy.tot_time.core.Time'
        UTC corresponding to 16:00 local tot_time on first night

    end : '~astropy.tot_time.core.Time'
        UTC corresponding to 16:00 local tot_time on last night
    """

    if startdate is None:
        current_utc = Time.now()
        start = Time(str((current_utc + utc_to_local).iso)[0:10] + ' 16:00:00.00') - utc_to_local
    else:
        try:
            start = Time(startdate + ' 16:00:00.00') - utc_to_local
        except ValueError as e:
            print(e)
            raise ValueError('\"{}\" not a valid date.  Expected string of the form \'YYYY-MM-DD\''.format(startdate))

    if enddate is None:  # default number of observation nights is 1
        return start, None
    else:
        try:
            end = Time(enddate + ' 16:00:00.00') - utc_to_local
            diff = int((end - start).value)  # difference between startdate and enddate
            if diff <= 0:
                raise ValueError('End date \"{}\" occurs before or on start date.'.format(enddate))
        except ValueError as e:
            print(e)
            raise ValueError('\"{}\" not a valid date.  '
                             'Must be after start date and of the form \'YYYY-MM-DD\''.format(enddate))

    start.format = 'jd'
    end.format = 'jd'
    return start, end


def test_dates():
    """
    Test getdates function.

    Run using pytest.
    """

    starttime = '2018-07-01'
    endtime = '2018-07-04'
    delta = -10 * u.h
    start, end = getdates(starttime, delta, endtime)
    print('start, end', start.iso, end.iso)
    assert (str(start.iso), str(end.iso)) == ('2018-07-02 02:00:00.000', '2018-07-05 02:00:00.000')
    print('Test successful!')
    return

if __name__=='__main__':
    test_dates()
