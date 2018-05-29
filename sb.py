import numpy as np
import math as m
import astropy.units as u

def ztwilight(alt):
    y = (-1.*alt - 9.0) / 9.0      #  /* my polynomial's argument...*/
    val = ((2.0635175 * y + 1.246602) * y - 9.4084495)*y + 6.132725 
    return val  

def xair(z):
    degrad  =   57.2957795130823/u.rad
    sin2 = np.sin(z/degrad)**2
    val = 1.0/(1.0-0.96*sin2)**0.5
    return val

def sb(mpa,mdist,mZD,ZD,sZD,cc):
    # calculate sky brightness based on formulas from Krisciunas & Schaefer 1991
    # Bryan Miller
    # November 5, 2004
    # June 1, 2015 - added cc parameter while testing cloud scattering corrections

    # Matt Bonnyman
    # copied to python May 23, 2018

    # Parameters:
    # mpa = moon phase angle in degrees
    # mdist = moon/object distance in degreee
    # mZD = moon zenith distance [deg]
    # ZD = object zenith distance [deg]
    # sZD = Sun zenith distance [deg]
    # cc = Cloud Cover constraint
    
    degrad = 57.2957795130823/u.rad
    k = 0.172 #mag/airmass relation for Hale Pohaku
    a = 2.51189
    Q = 27.78151

    sun_alt = 90.0 - sZD #sun altitude

    #print('sun_alt',sun_alt)

    Vzen = np.ones(len(ZD)) 
    Vzen = Vzen*21.587 #Dark sky zenith V surface brightness
    ii = np.where(sun_alt > -18.5)[0][:]
    Vzen[ii] = Vzen[ii] - ztwilight(sun_alt[ii]) #correction for sky brightness
    
    
    Bzen = 0.263 * a**(Q-Vzen) #zentih sky brightness
    Bsky = Bzen * xair(ZD) * 10.0**(-0.4*k*(xair(ZD)-1.0)) #sky constribution

    n = len(Bsky)
    istar = 0.0
    fp = 0.0
    Bmoon = np.zeros(n)
    for j in range(0,n):
        #print('mZD',mZD[j])
        #print('ZD',ZD[j])
        #print('mpa',mpa)
        if mZD[j]<90.8:
            istar = 10.0**(-0.4*(3.84 + 0.026 * abs(mpa) + (4.0e-9)*(mpa)**4.0))
            #print('istar',istar)
            #print('mdist',mdist[j])
            if mdist[j]>10.0:
                fp = (1.06 + np.cos(mdist[j]/degrad)**2) * 10.0**5.36 + 10.0**(6.15 - mdist[j]/40.0)
            else:
                fp = 6.2e7 /(mdist[j]**2) #original said fp=6.2d7/mdist[j]^2
            #print('fp',fp)
            Bmoon[j] = fp * istar * 10**(-0.4 * k * xair(mZD[j])) * (1.0 - 10**(-0.4 * k * xair(ZD[j])))
            #print('Bmoon',Bmoon[j])

    if cc!=0.5: #Very simple increase in SB if there are clouds
        Bmoon = 2.0 * Bmoon 
    
    sb = Q - np.log10( (Bmoon + Bsky)/0.263) / np.log10(a) #sky brightness in Vmag/arcsec^2
    # print('istar',istar)
    # print('fp',fp)
    # print('Bmoon',Bmoon)
    # print('Bsky',Bsky)
    # print('sb',sb)

    return sb
