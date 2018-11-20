# Matt Bonnyman 22 May 2018
import numpy as np
import astropy.units as u

def gcirc(ra1, dec1, ra2, dec2, degree=False):
    """
    Compute angular distance between two points on great circle.

    Parameter
    ---------
    ra1 : 'astropy.units.quantity.Quantity'
        right ascension of first point

    dec1 : 'astropy.units.quantity.Quantity'
        declination of first point

    ra2 : 'astropy.units.quantity.Quantity'
        right ascension of second point

    dec2 : 'astropy.units.quantity.Quantity'
        declination of second point

    degree : Output in degrees?
    """

    # more rigorous great circle angular distance calculation.
    del_dec_div2 = (dec2 - dec1) / 2.0
    del_ra_div2 = (ra2 - ra1) / 2.0
    sin_theta = np.sqrt(np.sin(del_dec_div2)**2
                        + np.cos(dec1)
                        * np.cos(dec2)
                        * np.sin(del_ra_div2)**2
                        )
    distance_rad = 2.0 * np.arcsin(sin_theta)
    if degree:
        distance = distance_rad.to(u.deg)
    else:
        distance = distance_rad
    return distance


def test_gcirc():

    ra1 = 0. * u.deg
    dec1 = 0. * u.deg
    ra2 = 180. * u.deg
    dec2 = 30. * u.deg

    print('\nTest gcirc.py...')

    print('input: ')
    print('\tpoint 1(ra,dec): (' + str(ra1) + ', ' + str(dec1) + ')')
    print('\tpoint 2(ra,dec): (' + str(ra2) + ', ' + str(dec2) + ')')

    print('output:')
    print('\t' + str(gcirc(ra1, dec1, ra2, dec2).round(8)))

    print('was expecting 5*pi/6...')

    assert (gcirc(ra1, dec1, ra2, dec2).round(8) == (5. * np.pi / 6. * u.rad).round(8))

    print('\t' + str(gcirc(ra1, dec1, ra2, dec2, degree=True).round(4)))

    print('was expecting 150 deg...')
    assert (gcirc(ra1, dec1, ra2, dec2, degree=True).round(4) == (150. * u.deg).round(4))

    print('Test successful!')


if __name__ == '__main__':
    test_gcirc()
