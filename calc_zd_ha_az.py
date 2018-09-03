# The calculations are from "Astronomical Photometry" by Henden & Kaitchuck
# IDL version Bryan Miller 2004
# converted to python Matt Bonnyman May 22, 2018

import copy
import numpy as np
import astropy.units as u


def calc_zd_ha_az(lst, latitude, ra, dec):
    """
    Calculate zenith distance, hour angle, and azimuth for a point on the sky.

    Parameters
    ----------
    lst : np.array of '~astropy.units.quantity.Quantity'
        Numpy array of local sidereal time(s).

    latitude : '~astropy.coordinates.angles.Latitude'
        Observer latitude.

    ra : '~astropy.coordinates.angles.Longitude'
        Right ascension.

    dec : '~astropy.coordinates.angles.Latitude'
        Declination.

    Returns
    -------
    ZD : np.array of '~astropy.units.quantity.Quantity'
        Zenith distance angle(s).

    HA : np.array of '~astropy.units.quantity.Quantity'
        Hour angle(s).

    AZ : np.array of '~astropy.units.quantity.Quantity'
        Azimuth angle(s).
    """

    verbose = False
    verbosesteps = False

    hra = (lst - ra)
    if verbosesteps:
        print('lst', lst.value)
        print('hra', hra.value)

    h1 = np.arcsin(np.sin(latitude) * np.sin(dec) + np.cos(latitude) * np.cos(dec) * np.cos(hra))
    if verbosesteps: print('h1', h1)

    AZ = np.arccos((np.sin(dec) - np.sin(latitude) * np.sin(h1)) / (np.cos(latitude) * np.cos(h1)))
    if verbosesteps: print('AZtemp', AZ.value)

    HA = copy.deepcopy(hra)
    ii = np.where(HA < -12. * u.hourangle)[0][:]
    if (len(ii) != 0):
        HA[ii] = HA[ii] + 24.0 * u.hourangle
        if verbosesteps: print('HA < -12 fixed', HA.value)

    ii = np.where(HA > 12.0 * u.hourangle)[0][:]
    if (len(ii) != 0):
        HA[ii] = HA[ii] - 24. * u.hourangle
        if verbosesteps: print('HA > 12 fixed', HA.value)

    ii = np.where(HA > 0. * u.hourangle)[0][:]
    if (len(ii) != 0):
        AZ[ii] = 360. * u.deg - AZ[ii]
        if verbosesteps: print('AZ fixed for HA > 0', AZ.value)

    ZD = 90. * u.deg - h1

    ZDcopy = copy.deepcopy(ZD)
    if verbosesteps: print('ZDcopy', ZD.value)

    ii = np.where(ZD > 85. * u.deg)[0][:]
    if (len(ii) != 0):
        ZDcopy[ii] = 85.0 * u.deg
        if verbosesteps: print('ZDcopy fixed', ZDcopy.value)

    ZD = ZD - 0.00452 * 800.0 * np.tan(ZDcopy) / (273.0 + 10.0) * u.deg

    if verbose:
        print('ra',ra.value)
        print('dec', dec.value)
        print('latitude', latitude.value)
        print('lst', lst.value)
        print('ZD ', ZD.value)
        print('HA', HA.value)
        print('AZ', AZ.value, AZ.unit)

    return ZD, HA, AZ
