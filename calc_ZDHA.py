
import numpy as np
import astropy.units as u

def calc_ZDHA(lst,latitude,longitude,ra,dec):
    # The calculations are from "Astronomical Photometry" by Henden & Kaitchuck
    # IDL version Bryan Miller 2004
    # converted to python Matt Bonnyman May 22, 2018 
    
    """Requires numpy arrays of lst time steps throughout night (in hours),
    observer lat, lon, and target ra, dec.
    Computes zenith distance, hour angle, and azimuth of target throughout night.
    Return values in 3 numpy arrays."""

    n = len(lst)
    ZD = np.zeros(n)
    HA = np.zeros(n)
    AZ = np.zeros(n)

    degrad  =   57.2957795130823/u.rad

    H = 15.0 * (lst - ra/15.0)

    sin_h1 = np.sin(latitude/degrad) * np.sin(dec/degrad) + \
         np.cos(latitude/degrad) * np.cos(dec/degrad) * np.cos(H/degrad)
    h1 = np.arcsin(sin_h1)*degrad

    cos_A = ( np.sin(dec/degrad) - np.sin(latitude/degrad) * np.sin(h1/degrad))\
        /(np.cos(latitude/degrad) * np.cos(h1/degrad))
    AZ = np.arccos(cos_A)*degrad

    HA = H/15.0
    HA = HA + 24.0 * (HA < -12)

    indeces = np.where(HA>0)[0][:]
    if (len(indeces) != 0):
        AZ[indeces] = 360.0 - AZ[indeces]

    ZD = 90 - h1

    ZDc = ZD
    indeces = np.where(ZD > 85)[0][:]
    if (len(indeces) != 0): 
        ZDc[indeces]=85.0

    ZD=ZD - 0.00452 * 800.0 * np.tan(ZDc/degrad)/(273.0 + 10.0)

    return ZD,HA,AZ

