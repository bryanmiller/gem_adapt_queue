import numpy as np
import astropy.units as u


def ztwilight(alt):
    y = (-1.*alt - 9.0*u.deg) / (9.0*u.deg)      #  /* my polynomial's argument...*/
    val = ((2.0635175 * y + 1.246602) * y - 9.4084495)*y + 6.132725 
    return val  

def xair(z):
    sin2 = np.sin(z)**2
    val = 1.0/(1.0-0.96*sin2)**0.5
    return val

def sb(mpa,mdist,mZD,ZD,sZD,cc=0.5):
    """
    Calculate sky brightness based on formulas from Krisciunas & Schaefer 1991
    Bryan Miller
    November 5, 2004
    June 1, 2015 - added cc parameter while testing cloud scattering corrections

    Matt Bonnyman
    converted from IDL to Python May 23, 2018

    Parameters
    ----------
    mpa : astropy.units.Quantity
        Moon phase angle at solar midnight in radians

    mdist : astropy.units.Quantity
        Numpy array of angular distances between target and moon

    mZD : astropy.units.Quantity
        Numpy array of Moon zenith distance angles

    ZD : astropy.units.Quantity
        Numpy array of target zenith distance angles

    sZD : astropy.units.Quantity
        Numpy array of Sun zenith distance angles

    cc (optional) : float
        Current cloud condition (DEFAULT = 0.5).

    Returns
    ---------
    sb : float
        Numpy array of sky background magnitudes at target location
    """
    verbose = False

    k = 0.172 #mag/airmass relation for Hale Pohaku
    a = 2.51189
    Q = 27.78151

    sun_alt = 90.0*u.deg - sZD #sun altitude
    if verbose: print('sun_alt',sun_alt)

    Vzen = np.ones(len(ZD)) 
    Vzen = Vzen*21.587  # Dark sky zenith V surface brightness
    ii = np.where(sun_alt > -18.5*u.deg)[0][:]
    Vzen[ii] = Vzen[ii] - ztwilight(sun_alt[ii])  # correction for sky brightness

    Bzen = 0.263 * a**(Q-Vzen)  # zentih sky brightness
    Bsky = Bzen * xair(ZD) * 10.0**(-0.4*k*(xair(ZD)-1.0))  # sky constribution

    n = len(Bsky)
    Bmoon = np.zeros(n)

    istar = 10. ** (-0.4 * (3.84 + 0.026 / u.deg * abs(mpa) + 4.e-9 * u.deg ** -4 * mpa ** 4.))

    ii = np.where(mZD<90.8*u.deg)[0][:]

    jj = ii[np.where(mdist[ii]>10.*u.deg)[0][:]]
    if len(jj)!=0:
        fpjj = (1.06 + np.cos(mdist[jj]) ** 2) * 10.0 ** 5.36 + 10.0 ** (6.15 - (mdist[jj]) / (40.0 * u.deg))
        Bmoon[jj] = fpjj * istar * 10 ** (-0.4 * k * xair(mZD[jj])) * (1.0 - 10 ** (-0.4 * k * xair(ZD[jj])))

    kk = np.where(ii!=jj)[0][:]
    if len(kk) != 0:
        fpkk = fp = 6.2e7*u.deg**2 / (mdist[kk]**2) #original said fp=6.2d7/mdist[j]^2
        Bmoon[kk] = fpkk * istar * 10 ** (-0.4 * k * xair(mZD[kk])) * (1.0 - 10 ** (-0.4 * k * xair(ZD[kk])))

    if cc>0.5: #Very simple increase in SB if there are clouds
        Bmoon = 2.0 * Bmoon

    sb = Q - np.log10( (Bmoon + Bsky)/0.263) / np.log10(a) #sky brightness in Vmag/arcsec^2

    if verbose:
        print('Vzen',Vzen)
        print('Bzen',Bzen)
        print('istar',istar)
        print('fp',fp)
        print('Bmoon',Bmoon)
        print('Bsky',Bsky)
        print('sb',sb)

    return sb

    # for j in range(0,n):
    #     if mZD[j]<90.8*u.deg:
    #         istar = 10.**(-0.4*(3.84 + 0.026/u.deg * abs(mpa) + 4.e-9*u.deg**-4 * mpa**4.))
    #         if mdist[j]>10.*u.deg:
    #             fp = (1.06 + np.cos(mdist[j])**2) * 10.0**5.36 + 10.0**(6.15 - (mdist[j]) / (40.0*u.deg))
    #         else:
    #             fp = 6.2e7*u.deg**2 / (mdist[j]**2) #original said fp=6.2d7/mdist[j]^2
    #
    #         Bmoon[j] = fp * istar * 10**(-0.4 * k * xair(mZD[j])) * (1.0 - 10**(-0.4 * k * xair(ZD[j])))