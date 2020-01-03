# Matt Bonnyman 2018-07-12

import numpy as np
import astropy.units as u
from astroplan import Observer


def getsite(site_name, daylightsavings):
    """
    Initialize '~astroplan.Observer' for Cerro Pachon or Mauna Kea.

    Parameters
    ----------
    site_name : string
        Observatory site name.
        Allowed locations...
        - 'gemini_north' or 'MK' (Mauna Kea)
        - 'gemini_south' or 'CP' (Cerro Pachon).

    daylightsavings : boolean
        Toggle daylight savings tot_time (does not apply to Gemini North)

    Returns
    -------
    site : '~astroplan.Observer'
        Observatory site location object.

    timezone_name : string
        Pytz timezone name for observatory site

    utc_to_local : '~astropy.units' hours
        Time difference between utc and local tot_time in hours.
    """

    if np.logical_or(site_name == 'gemini_south', site_name == 'CP'):
        site_name = 'gemini_south'
        timezone_name = 'Chile/Continental'
        if daylightsavings:
            utc_to_local = -3.*u.h
        else:
            utc_to_local = -4.*u.h
    elif np.logical_or(site_name == 'gemini_north', site_name == 'MK'):
        site_name = 'gemini_north'
        timezone_name = 'US/Hawaii'
        utc_to_local = -10.*u.h
    else:
        print('Input error: Could not determine observer location and timezone. '
              'Allowed inputs are \'gemini_south\', \'CP\'(Cerro Pachon), \'gemini_north\', and \'MK\'(Mauna Kea).')
        raise ValueError

    # Create Observer object for observatory site
    # Note: can add timezone=timezone_name later
    # if desired (useful if pytz objects used)
    site = Observer.at_site(site_name)

    return site, timezone_name, utc_to_local

def test_getsite():
    gs, tz, utc_to_local = getsite('CP', True)
    print(gs, tz, utc_to_local)
    assert gs.name == 'gemini_south'
    assert tz == 'Chile/Continental'
    assert utc_to_local == -3.*u.h

    gn, tz, utc_to_local = getsite('MK', True)
    print(gn, tz, utc_to_local)
    assert gn.name == 'gemini_north'
    assert tz == 'US/Hawaii'
    assert utc_to_local == -10. * u.h
    print('Test successful!')
    return

if __name__=='__main__':
    test_getsite()
