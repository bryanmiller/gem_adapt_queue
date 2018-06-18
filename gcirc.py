import numpy as np


#Matt Bonnyman
#May 22, 2018

def gcirc(ra1,dec1,ra2,dec2):
    """Computes angular distance between two points.  
    
    Parameter
    ---------
    ra1 :

    dec1 :

    ra2 :

    dec2 :
    """

    #more rigorous great circle angular distance calculation.
    del_dec_div2 = ( dec2 - dec1 ) / (2.0)
    del_ra_div2 =  ( ra2 - ra1 ) / (2.0)
    sin_theta = np.sqrt( np.sin(del_dec_div2)**2 + np.cos(dec1) * np.cos(dec2) * np.sin(del_ra_div2)**2 )
    distance_rad = 2.0*np.arcsin(sin_theta)

    return distance_rad #return distance in degrees

    #original version used angular distance
    #cos_theta = (np.sin(dec2/decrad)*np.sin(dec1/decrad)) +\
    #    (np.cos(dec2/decrad)*np.cos(dec1/decrad)*np.cos((ra2/decrad)-(ra1/decrad)))
    #radian_distance = np.arccos(cos_theta)

if __name__=='__main__':
    import astropy.units as u
    ra1 = 0. *u.deg
    dec1 = 0. *u.deg
    ra2 = 180. *u.deg
    dec2 = 30. * u.deg

    print('\nTest gcirc.py...')
    print('input: ')
    print('\tpoint 1(ra,dec): ('+str(ra1)+', '+str(dec1)+')')
    print('\tpoint 2(ra,dec): (' + str(ra2) + ', ' + str(dec2) + ')')
    print('output: ',gcirc(ra1,dec1,ra2,dec2).round(8))
    print('expected output: ', '5/6 * pi -->', (5/6*np.pi*u.rad).round(8))
    assert(gcirc(ra1,dec1,ra2,dec2).round(8)==(5*np.pi/6*u.rad).round(8))
    print('Test successful!')
