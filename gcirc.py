
import numpy as np
import astropy.units as u

#Matt Bonnyman
#May 22, 2018

def degrees(ra1,dec1,ra2,dec2):
    """Computes angular distance between two points.  
    Provide input coordinates in degrees as unitless floats.
    Function will return degrees"""
    degrad  =   57.2957795130823/u.rad

    #more rigorous great circle angular distance calculation.
    del_dec_div2 = ( dec2 - dec1 ) / (2.0 * degrad)
    del_ra_div2 =  ( ra2 - ra1 ) / (2.0 * degrad)
    sin_theta = np.sqrt( np.sin(del_dec_div2)**2 + np.cos(dec1/degrad) * np.cos(dec2/degrad) * np.sin(del_ra_div2)**2 )
    distance_rad = 2.0*np.arcsin(sin_theta) 

    return distance_rad*degrad #return distance in degrees

    #original version used angular distance
    #cos_theta = (np.sin(dec2/decrad)*np.sin(dec1/decrad)) +\
    #    (np.cos(dec2/decrad)*np.cos(dec1/decrad)*np.cos((ra2/decrad)-(ra1/decrad)))
    #radian_distance = np.arccos(cos_theta)