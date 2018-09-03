from astropy.time import Time


def deltat(time_strings):
    """
    Get dt

    Parameters
    ----------
    time_strings : array of strings
        iso format strings of utc or local times in timetable

    Returns
    -------
    dt : '~astropy.units.quantity.Quantity'
        differential time length

    """
    return (Time(time_strings[1]) - Time(time_strings[0])).to('hour').round(2)

def test_deltat():
    import astropy.units as u
    times = Time(['2018-01-01 12:00:00', '2018-01-01 12:30:00'])
    dt = deltat(times)
    print('dt =', dt)
    assert dt == 0.5 * u.h
    print('Test successful!')
    return

if __name__=='__main__':
    test_deltat()
