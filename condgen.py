# Generates conditions for simulations
import re
import random
import numpy as np
import scipy.stats as st
import astropy.units as u
from astroplan import Observer
import matplotlib.units as units
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def percent_string(percentile):
    """
    Convert decimal percentile to string.
    Round to nearest percent.
    i.e. 0.78695 --> '79%'
    """
    return str(int(round(percentile,2)*100))+'%'

class condgen(object):

    def __init__(self, iq=1., cc=1., wv=1.):
        """
        Generate image quality, cloud condition, and water vapor
        condition percentiles.

        parameters
        ----------
        iq : int
            image quality percentile condition rounded to nearest integer

        cc : int
            cloud condition percentile condition rounded to nearest integer

        wv : int
            water vapor percentile condition rounded to nearest integer

        class methods
        -------------
        gauss : generate random conditions from a gaussian distribution
        """

        self.iq = iq
        self.cc = cc
        self.wv = wv

    @classmethod
    def gauss(cls):
        """
        Generate random iq, cc, wv condition percentiles
        and return as strings.
        i.e. iq,cc,wv = '73%','92%','3%'.
        """
        verbose = False

        def gauss_percentile():
            """
            Generate randon number from gaussian distribtion.
            Return as string percentile rounded to nearest percent.
            """
            rand = abs(random.gauss(0,1))
            percentile = st.norm.cdf(rand)-st.norm.cdf(-rand)
            return percent_string(percentile)

        iq = gauss_percentile()
        cc = gauss_percentile()
        wv = gauss_percentile()

        if verbose:
            print('iq = ',iq)
            print('cc = ',cc)
            print('wv = ',wv)

        return cls(iq=iq, cc=cc, wv=wv)

class wind(object):

    def __init__(self, site):

        if site.name == 'gemini_south':
            dir = 330. * u.deg
            dsig = 20. * u.deg
            vel = 5. * u.km / u.h
            vsig = 3. * u.km / u.h
        elif site.name == 'gemini_north':
            dir = 330. * u.deg
            dsig = 20. * u.deg
            vel = 5. * u.km / u.h
            vsig = 3. * u.km / u.h
        else:
            dir = 0. * u.deg
            dsig = 0. * u.deg
            vel = 0. * u.km / u.h
            vsig = 0. * u.km / u.h

        wdir = random.gauss(dir, dsig)
        if wdir>360*u.deg:
            wdir = wdir - 360*u.deg
        self.dir = wdir

        wvel = random.gauss(vel, vsig)
        if wvel<0. * u.km/u.h:
            wvel = 0. * u.km/u.h
        self.vel = wvel

def test_condgen(pp, function):

    """
    Create plots of conditions generated
    from a given distribution. 
    """

    n_nums = 300
    x = np.arange(n_nums)

    iqs = []
    ccs = []
    wvs = []
    rands = []
    bins = np.linspace(0,1,50)
    randbins = np.linspace(-3,3,50)
    for i in range(n_nums):
        cond = function()
        iqs.append(int(re.findall('\d+',cond.iq)[0])/100)
        ccs.append(int(re.findall('\d+',cond.cc)[0])/100)
        wvs.append(int(re.findall('\d+',cond.wv)[0])/100)
        rands.append(random.gauss(0,1))

    fig = plt.figure()
    fig.suptitle('Test: '+str(function), fontsize=12)

    ax1 = plt.subplot(221)
    ax1.set_title('iq')
    ax1.hist(iqs,bins=bins)
    
    ax2 = plt.subplot(222)
    ax2.set_title('cc')
    ax2.hist(ccs,bins=bins)
    
    ax3 = plt.subplot(223)
    ax3.set_title('wv')
    ax3.hist(wvs,bins=bins)
    
    ax4 = plt.subplot(224)
    ax4.set_title('example distribution')
    ax4.hist(rands,bins=randbins)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    pp.savefig()
    plt.clf()

    fig = plt.figure()

    ax1 = plt.subplot(221)
    ax1.set_title('iq')
    ax1.plot(x,iqs)
    
    ax2 = plt.subplot(222)
    ax2.set_title('cc')
    ax2.plot(x,ccs)
    
    ax3 = plt.subplot(223)
    ax3.set_title('wv')
    ax3.plot(x,wvs)
    
    ax4 = plt.subplot(224)
    ax4.set_title('example distribution')
    ax4.plot(x,rands)

    fig.tight_layout()
    pp.savefig()
    plt.clf()

    return

def test_wind(pp):

    """
    Create plots of conditions generated
    from a given distribution.
    """
    site = Observer.at_site('gemini_south')

    n_nums = 500
    x = np.arange(n_nums)

    dir = []
    vel = []

    for i in x:
        w = wind(site=site)
        dir.append(w.dir)
        vel.append(w.vel)

    # for i in x:
    #     print(x[i])
    # for i in x:
    #     print(dir[i])
    # for i in x:
    #     print(vel[i])

    fig = plt.figure()
    fig.suptitle('Test: ' + str(wind), fontsize=12)

    ax1 = plt.subplot(211)
    ax1.set_title('Direction')
    ax1.set_ylim(220,380)
    for i in x:
        ax1.scatter(x[i], dir[i], marker='.')

    ax2 = plt.subplot(212)
    ax2.set_title('Velocity')
    for i in x:
        ax2.scatter(x[i], vel[i], marker='.')

    fig.tight_layout()
    pp.savefig()
    plt.clf()

    return

if __name__=='__main__':

    filename = 'condgen_test_plots.pdf'
    # save plots to file
    pp = PdfPages(filename)

    test_condgen(pp=pp, function=condgen.gauss)
    # test_wind(pp=pp)

    pp.close()

    print(' Test successful. Output: ' + str(filename))
