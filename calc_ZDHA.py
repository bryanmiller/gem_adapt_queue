import copy
import numpy as np
import astropy.units as u

def calc_ZDHA(lst,latitude,ra,dec):

    verbose = False

    # The calculations are from "Astronomical Photometry" by Henden & Kaitchuck
    # IDL version Bryan Miller 2004
    # converted to python Matt Bonnyman May 22, 2018 

    n = len(lst)
    ZD = np.zeros(n)
    AZ = np.zeros(n)

    HA = (lst - ra)
    if verbose: print('HA',HA)

    sin_h1 = np.sin(latitude) * np.sin(dec) + \
         np.cos(latitude) * np.cos(dec) * np.cos(HA[:])
    h1 = np.arcsin(sin_h1)
    if verbose: print('h1',h1)

    cos_A = ( np.sin(dec) - np.sin(latitude) * np.sin(h1))\
        /(np.cos(latitude) * np.cos(h1))
    AZ = np.arccos(cos_A)
    if verbose: print('AZ',AZ)

    ii = np.where(HA < -12.*u.hourangle)[0][:]
    if (len(ii) != 0):
        HA[ii] = HA[ii] + 24.0*u.hourangle 
        if verbose: print('HA < -12 fixed',HA)

    ii = np.where(HA > 12.0 * u.hourangle)[0][:]
    if (len(ii) != 0):
        HA[ii] = HA[ii] - 24. * u.hourangle
        if verbose: print('HA > 12 fixed', HA)

    ii = np.where(HA>0.*u.hourangle)[0][:]
    if (len(ii) != 0):
        AZ[ii] = 360.*u.deg - AZ[ii]
        if verbose: print('AZ fixed',AZ)

    ZD = 90.*u.deg - h1
    if verbose: print('ZD',ZD)

    ZDcopy = copy.deepcopy(ZD)
    ii = np.where(ZD > 85.*u.deg)[0][:]
    if (len(ii) != 0): 
        ZDcopy[ii]=85.0*u.deg
        if verbose: print('ZDcopy fixed',ZDcopy)

    ZD = ZD - 0.00452 * 800.0 * np.tan(ZDcopy)/(273.0 + 10.0)*u.deg
    
    if verbose: print('ZD final',ZD)

    return ZD,HA,AZ




