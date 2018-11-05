# Matt Bonnyman 25 July 2018

import re
import numpy as np


def convertcond(iq, cc, bg, wv):
    """
    Convert actual weather conditions
    to decimal values in range [0,1].
    Conditions 'Any' or 'null' are assigned 1.
    Percentages converted to decimals.

    Accepts str or numpy.array of str.

    Parameters
    ----------
    iq : string or np.ndarray
        Image quality

    cc : string or np.ndarray
        Cloud condition

    bg : string or np.ndarray
        Sky background

    wv : string or np.ndarray
        Water vapor

    Returns
    ---------
    iq : float or np.ndarray
        Image quality

    cc : float or np.ndarray
        Cloud condition

    bg : float or np.ndarray
        Sky background

    wv : float or np.ndarray
        Water vapor
    """
    verbose = False

    if verbose:
        print(' Inputs conds...')
        print(' iq', iq)
        print(' cc', cc)
        print(' bg', bg)
        print(' wv', wv)

    errormessage = 'Must be type str, float, or np.ndarray'

    if isinstance(iq, np.ndarray):
        iq = np.array(list(map(conviq, iq)))
    elif isinstance(iq, str) or isinstance(iq, float):
        iq = conviq(iq)
    else:
        raise ValueError(errormessage)

    if isinstance(cc, np.ndarray):
        cc = np.array(list(map(convcc, cc)))
    elif isinstance(cc, str) or isinstance(cc, float):
        cc = convcc(cc)
    else:
        raise ValueError(errormessage)

    if isinstance(bg, np.ndarray):
        bg = np.array(list(map(convbg, bg)))
    elif isinstance(bg, str) or isinstance(bg, float):
        bg = convbg(bg)
    else:
        raise ValueError(errormessage)

    if isinstance(wv, np.ndarray):
        wv = np.array(list(map(convwv, wv)))
    elif isinstance(wv, str) or isinstance(wv, float):
        wv = convwv(wv)
    else:
        raise ValueError(errormessage)

    if verbose:
        print(' Converted conds...')
        print(' iq', iq)
        print(' cc', cc)
        print(' bg', bg)
        print(' wv', wv)

    return iq, cc, bg, wv


def conviq(string):
    """
    Convert image quality percentile string to decimal value.
    """
    if np.logical_or('Any' in string, 'null' in string):
        iq = 1.
    else:
        iq = float(re.findall(r'[\d\.\d]+', string)[0])/100
        if iq <= 0.2:
            iq = 0.2
#        elif 0.2 < iq <= 0.5:
#            iq = 0.5
        elif 0.2 < iq <= 0.7:
            iq = 0.7
        elif 0.7 < iq <= 0.85:
            iq = 0.85
        else:
            iq = 1.
    return iq


def convcc(string):
    """
    Convert cloud condition percentile string to decimal value.
    """
    if np.logical_or('Any' in string, 'null' in string):
        cc = 1.
    else:
        cc = float(re.findall(r'[\d\.\d]+',string)[0])/100
        if cc <= 0.5:
#            cc = 0.2
#        elif 0.2 < cc <= 0.5:
            cc = 0.5
        elif 0.5 < cc <= 0.7:
            cc = 0.7
        elif 0.5 < cc <= 0.80:
            cc = 0.80
        else:
            cc = 1.
    return cc


def convbg(string):
    """
    Convert sky background percentile string to decimal value.
    """
    if np.logical_or('Any' in string, 'null' in string):
        bg = 1.
    else:
        bg = float(re.findall(r'[\d\.\d]+',string)[0])/100
        if bg <= 0.2:
            bg = 0.2
        elif 0.2 < bg <= 0.5:
            bg = 0.5
#        elif 0.5 < bg <= 0.7:
#            bg = 0.7
        elif 0.5 < bg <= 0.80:
            bg = 0.80
        else:
            bg = 1.
    return bg


def convwv(string):
    """
    Convert water vapour percentile string to decimal value.
    """
    if np.logical_or('Any' in string, 'null' in string):
        wv = 1.
    else:
        wv = float(re.findall(r'[\d\.\d]+', string)[0]) / 100
        if wv <= 0.2:
            wv = 0.2
        elif 0.2 < wv <= 0.5:
            wv = 0.5
#        elif 0.5 < wv <= 0.7:
#            wv = 0.7
        elif 0.5 < wv <= 0.80:
            wv = 0.80
        else:
            wv = 1.
    return wv


def inputcond(iq, cc, wv):
    """
    Handle user input for conditions constraints.
    If not 'any' or 'Any', first 2 digits are converted to a percentage.
    Otherwise, a ValueError is raised.

    Examples
    --------
    'any' --> 'Any'
    '10'  --> '10%'
    '85%' --> '85%'
    '08'  --> '08%'
    '808' --> '80%'
    '8'   --> '8%'
    '8%'  --> ValueError

    Parameters
    ----------
    iq : str
        image quality percentile

    cc : str
        cloud condition percentile

    wv : str
        water vapor percentile

    Returns
    -------
    newiq, newcc, newwv : str, str, str
        Formatted condition constraints strings
    """

    if iq.lower() == 'any':
        iq = 'Any'
    else:
        try:
            float(iq[0:2])
            iq = iq[0:2] + '%'
        except ValueError:
            raise ValueError
    if cc.lower() == 'any':
        cc = 'Any'
    else:
        try:
            float(cc[0:2])
            cc = cc[0:2] + '%'
        except ValueError:
            raise ValueError
    if wv.lower() == 'any':
        wv = 'Any'
    else:
        try:
            float(wv[0:2])
            wv = wv[0:2] + '%'
        except ValueError:
            raise ValueError
    return iq, cc, wv


def sb_to_cond(sb):
    """
    Convert visible sky background magnitudes to decimal conditions.

        Conversion scheme:
            1.0 |          vsb < 19.61
            0.8 | 19.61 <= vsb < 20.78
            0.5 | 20.78 <= vsb < 21.37
            0.2 | 21.37 <= vsb


    Input
    -------
    sb :  np.ndarray of floats
        TargetInfo object with time dependent vsb quantities

    Return
    -------
    cond : np.ndarray of floats
        sky background condition values
    """

    cond = np.empty(len(sb), dtype=float)
    ii = np.where(sb < 19.61)[0][:]
    cond[ii] = 1.
    ii = np.where(np.logical_and(sb >= 19.61, sb < 20.78))[0][:]
    cond[ii] = 0.8
    ii = np.where(np.logical_and(sb >= 20.78, sb < 21.37))[0][:]
    cond[ii] = 0.5
    ii = np.where(sb >= 21.37)[0][:]
    cond[ii] = 0.2
    return cond


def test_conditions():

    assert (convertcond('5%', '5%', '5%', '5%') == (0.2, 0.2, 0.2, 0.2))
    assert (convertcond('59%', '59%', '59%', '59%') == (0.7, 0.7, 0.7, 0.7))

    testiq = np.array(['15%', '25%', '55%', '75%', '80%', '85%', 'Any', 'null'])
    testcc = np.array(['15%', '25%', '55%', '75%', '80%', '85%', 'Any', 'null'])
    testbg = np.array(['15%', '25%', '55%', '75%', '80%', '85%', 'Any', 'null'])
    testwv = np.array(['15%', '25%', '55%', '75%', '80%', '85%', 'Any', 'null'])
    iq, cc, bg, wv = convertcond(testiq, testcc, testbg, testwv)
    assert ((iq == np.array([0.2, 0.5, 0.7, 0.85, 0.85, 0.85, 1., 1.])).all())
    assert ((cc == np.array([0.2, 0.5, 0.7, 0.85, 0.85, 0.85, 1., 1.])).all())
    assert ((bg == np.array([0.2, 0.5, 0.7, 0.85, 0.85, 0.85, 1., 1.])).all())
    assert ((wv == np.array([0.2, 0.5, 0.7, 0.85, 0.85, 0.85, 1., 1.])).all())

    assert (inputcond('any', '12', '9') == ('Any', '12%', '9%'))
    assert (inputcond('908', '1', '44') == ('90%', '1%', '44%'))
    print('Test successful!')

if __name__=='__main__':
    test_conditions()
