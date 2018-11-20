import numpy as np
import astropy.units as u


def ztwilight(alt):
    """
    From IDL GQPT Bryan Miller 2004
    """
    y = (-1. * alt - 9.0 * u.deg) / (9.0 * u.deg)      #  /* my polynomial's argument...*/
    val = ((2.0635175 * y + 1.246602) * y - 9.4084495) * y + 6.132725
    return val


def xair(z):
    """
    From IDL GQPT Bryan Miller 2004
    """
    sin2 = np.sin(z)**2
    val = 1.0/(1.0-0.96*sin2)**0.5
    return val


def sb(mpa, mdist, mZD, ZD, sZD, cc, verbose = False):
    """
    Calculate sky brightness based on formulas from Krisciunas & Schaefer 1991
    Bryan Miller
    November 5, 2004
    June 1, 2015 - added cc parameter while testing cloud scattering corrections

    Matt Bonnyman
    converted from IDL to Python May 23, 2018

    Parameters
    ----------
    mpa : '~astropy.units.Quantity'
        Moon phase angle at solar midnight in degrees

    mdist : array of '~astropy.units.Quantity'
        Numpy array of angular distances between target and moon

    mZD : array of '~astropy.units.Quantity'
        Numpy array of Moon zenith distance angles

    ZD : array of '~astropy.units.Quantity'
        Numpy array of target zenith distance angles

    sZD : array of '~astropy.units.Quantity'
        Numpy array of Sun zenith distance angles

    cc : array of floats
        Current cloud condition.

    Returns
    ---------
    skybright : float
        Numpy array of sky background magnitudes at target location
    """

    k = 0.172  # mag/airmass relation for Hale Pohaku
    a = 2.51189
    Q = 27.78151

    sun_alt = 90.0 * u.deg - sZD  # sun altitude
    if verbose:
        print('sun_alt', sun_alt)

    Vzen = np.ones(len(ZD))
    Vzen = Vzen * 21.587  # Dark sky zenith V surface brightness
    ii = np.where(sun_alt > -18.5 * u.deg)[0][:]
    Vzen[ii] = Vzen[ii] - ztwilight(sun_alt[ii])  # correction for sky brightness

    Bzen = 0.263 * a**(Q - Vzen)  # zenith sky brightness
    Bsky = Bzen * xair(ZD) * 10.0**(-0.4*k*(xair(ZD)-1.0))  # sky contribution

    n = len(Bsky)
    Bmoon = np.zeros(n)

    istar = 10. ** (-0.4 * (3.84 + 0.026 / u.deg * abs(mpa) + 4.e-9 * u.deg ** -4 * mpa ** 4.))

    ii = np.where(mZD < 90.8 * u.deg)[0][:]

    jj = ii[np.where(mdist[ii] > 10. * u.deg)[0][:]]
    if len(jj) != 0:
        fpjj = (1.06 + np.cos(mdist[jj]) ** 2) * 10.0 ** 5.36 + 10.0 ** (6.15 - (mdist[jj]) / (40.0 * u.deg))
        Bmoon[jj] = fpjj * istar * 10 ** (-0.4 * k * xair(mZD[jj])) * (1.0 - 10 ** (-0.4 * k * xair(ZD[jj])))

    kk = np.where(ii != jj)[0][:]
    if len(kk) != 0:
        fpkk = 6.2e7 * u.deg**2 / (mdist[kk]**2)  # original said fp=6.2d7/mdist[j]^2
        Bmoon[kk] = fpkk * istar * 10 ** (-0.4 * k * xair(mZD[kk])) * (1.0 - 10 ** (-0.4 * k * xair(ZD[kk])))

    # hh = np.where(np.logical_and(cc > 0.5, cc < 0.8))[0][:]
    # if len(hh) != 0:  # very simple increase in SB if there are thin clouds
    #     Bmoon[hh] = 2.0 * Bmoon[hh]

    skybright = Q - np.log10((Bmoon + Bsky)/0.263) / np.log10(a)  # sky brightness in Vmag/arcsec^2

    if verbose:
        print('Vzen', Vzen)
        print('Bzen', Bzen)
        print('istar', istar)
        print('Bmoon', Bmoon)
        print('Bsky', Bsky)
        print('skybright', skybright)

    return skybright
