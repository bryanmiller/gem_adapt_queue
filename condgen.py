# Generates conditions for simulations
import re
import random
import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def percent_string(percentile):
    """
    Convert decimal percentile to string.
    Round to nearest percent.
    i.e. 0.78695 --> '79%'
    """
    return str(int(round(percentile,2)*100))+'%'

def gauss_cond():
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

    return iq, cc, wv



def test(pp,function):

    """
    Create plots of conditions generated
    from a given distribution. 
    """

    n_nums = 500
    x = np.arange(n_nums)

    iqs = []
    ccs = []
    wvs = []
    rands = []
    bins = np.linspace(0,1,50)
    randbins = np.linspace(-3,3,50)
    for i in range(n_nums):
        iq,cc,wv = function()
        iqs.append(int(re.findall('\d+',iq)[0])/100)
        ccs.append(int(re.findall('\d+',cc)[0])/100)
        wvs.append(int(re.findall('\d+',wv)[0])/100)
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

if __name__=='__main__':
    
    # save plots to file
    pp = PdfPages('condgen_test_plots.pdf')

    test(pp=pp,function=gauss_cond)

    pp.close()


