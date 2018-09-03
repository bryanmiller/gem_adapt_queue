# Bryan Miller
# Matt Bonnyman 18 July 2018

import numpy as np
import astropy.units as u


def airmass(zd):
    """
    Calculate airmasses

    Parameter
    ---------
    zd : array of '~astropy.units.Quantity'
        zenith distance angles

    Return
    ---------
    am : array of floats
        airmasses
    """
    am = np.full(len(zd), 20.)
    ii = np.where(zd < 87. * u.deg)[0][:]
    sec_z = 1. / np.cos(zd[ii])
    am[ii] = sec_z - 0.0018167 * ( sec_z - 1 ) - 0.002875 * ( sec_z - 1 )**2 - 0.0008083 * ( sec_z - 1 )**3
    return am
