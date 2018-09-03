# Bryan Miller 2004
# Converted to python - Matt Bonnyman 2018-07-12

import numpy as np
import astropy.units as u

def sun_horizon(site):
    """
    Calculate angle between sun and horizon at sunset for observer's location.

    Parameter
    ---------
    site : '~astroplan.Observer'

    Return
    --------
    '~astropy.units.Quantity'
    """
    sun_horiz = -.83 * u.deg
    equat_radius = 6378137. * u.m
    return sun_horiz - np.sqrt(2. * site.location.height / equat_radius) * (180. / np.pi) * u.deg
