import numpy as np
import re

def conditions( iq, cc, bg, wv):
    
    """
    Convert weather condition constraints found in OT catalog
    to decimal values in range [0,1].
    Conditions 'Any' or 'null' are assigned 1.
    Percentages converted to decimals.
    Image qualities of 70% are set to 50%.
    Water vapour values of 50% are set to 20%.

    Parameters
    ----------
    iq : string
        Numpy array of image quality constraints

    cc : string
        Numpy array of cloud condition constraints

    bg : string
        Numpy array of sky background constraints

    wv : string
        Numpy array of water vapor constraints

    Returns
    ---------
    iq : float
        Numpy array of converted image quality constraints

    cc : float
        Numpy array of converted cloud condition constraints

    bg : float
        Numpy array of converted sky background constraints

    wv : float
        Numpy array of converted water vapor constraints
    """

    #============ image quality =============
    ii = np.where([np.logical_or('Any' in str,'null' in str) for str in iq])[0][:]
    if len(ii)!=0:
        iq[ii] = 1. #set 'Any' or 'null' to 1.

    iq = iq.astype('<U2') #trim array elements to 2 characters
    iq = np.asarray(iq,dtype='f8') #convert array to numpy type w/ double precision floats

    ii = np.where(iq!=1.)[0][:] #get indeces of percentage IQs
    if len(ii) != 0:
        iq[ii] = iq[ii]/100. #divide all IQ percentages by 100

    ii = np.where(iq==0.7)[0][:]
    if len(ii) != 0:
        iq[ii] = 0.5 #change 70% image qualities to 50%

    #========= cloud condition ==============
    ii = np.where([np.logical_or('Any' in str, 'null' in str) for str in cc])[0][:]
    if len(ii) != 0:
        cc[ii] = 1. #set 'Any' or 'null' to 1.

    cc = cc.astype('<U2') #trim array elements to 2 characters
    cc = np.asarray(cc,dtype='f8') #convert array to numpy type w/ double precision floats

    ii = np.where(cc!=1.)[0][:] #get indeces of percentage CCs
    if len(ii) != 0:
        cc[ii] = cc[ii]/100. #divide all CC percentages by 100

    #=========== sky background =============
    ii = np.where([np.logical_or('Any' in str, 'null' in str) for str in bg])[0][:]
    if len(ii) != 0:
        bg[ii] = 1. #set 'Any' or 'null' to 1.

    bg = bg.astype('<U2') #trim array elements to 2 characters
    bg = np.asarray(bg,dtype='f8') #convert array to numpy type w/ double precision floats

    ii = np.where(bg!=1.)[0][:] #get indeces of percentage BGs
    if len(ii) != 0:
        bg[ii] = bg[ii]/100. #divide all BG percentages by 100

    #============= water vapour =============
    ii = np.where([np.logical_or('Any' in str, 'null' in str) for str in wv])[0][:]
    if len(ii) != 0:
        wv[ii] = 1. #set 'Any' or 'null' to 1.

    wv = wv.astype('<U2') #trim array elements to 2 characters
    wv = np.asarray(wv,dtype='f8') #convert array to numpy type w/ double precision floats

    ii = np.where(wv!=1.)[0][:] #get indeces of percentage WVs
    if len(ii) != 0:
        wv[ii] = wv[ii]/100. #divide all WV percentages by 100

    ii = np.where(wv==0.5)[0][:]
    if len(ii) != 0:
        wv[ii] = 0.2 #change 50% water vapour values to 20%

    return iq,cc,bg,wv




def actual_conditions( iq, cc, bg, wv):
    """
    Convert actual weather conditions
    to decimal values in range [0,1].
    Conditions 'Any' or 'null' are assigned 1.
    Percentages converted to decimals.

    Parameters
    ----------
    iq : string
        Image quality

    cc : string
        Cloud condition

    bg : string
        Sky background

    wv : string
        Water vapor

    Returns
    ---------
    iq : float
        Image quality

    cc : float
        Cloud condition

    bg : float
        Sky background

    wv : float
        Water vapor
    """

    # ============ image quality =============
    if np.logical_or('Any' in iq,'null' in iq):
        iq = 1.
    else:
        iq = float(re.findall(r'[\d\.\d]+',iq)[0])/100
        if iq <= 0.2:
            iq = 0.2
        elif 0.2< iq <= 0.5:
            iq = 0.5
        elif 0.5< iq <= 0.7:
            iq = 0.7
        elif 0.7< iq <= 0.85:
            iq = 0.85
        else:
            iq = 1.
    # ========= cloud condition ==============
    if np.logical_or('Any' in cc,'null' in cc):
        cc = 1.
    else:
        cc = float(re.findall(r'[\d\.\d]+',cc)[0])/100
        if cc <= 0.2:
            cc = 0.2
        elif 0.2 < cc <= 0.5:
            cc = 0.5
        elif 0.5 < cc <= 0.7:
            cc = 0.7
        elif 0.5 < cc <= 0.85:
            cc = 0.85
        else:
            cc = 1.
    # =========== sky background =============
    if np.logical_or('Any' in bg,'null' in bg):
        bg = 1.
    else:
        bg = float(re.findall(r'[\d\.\d]+',bg)[0])/100
        if bg <= 0.2:
            bg = 0.2
        elif 0.2 < bg <= 0.5:
            bg = 0.5
        elif 0.5 < bg <= 0.7:
            bg = 0.7
        elif 0.5 < bg <= 0.85:
            bg = 0.85
        else:
            bg = 1.
    # ============= water vapour =============
    if np.logical_or('Any' in wv,'null' in wv):
        wv = 1.
    else:
        wv = float(re.findall(r'[\d\.\d]+',wv)[0])/100
        if wv <= 0.5:
            wv = 0.2
        elif 0.5 < wv <= 0.7:
            wv = 0.7
        elif 0.5 < wv <= 0.85:
            wv = 0.85
        else:
            wv = 1.
    return iq,cc,bg,wv


def _test_actual_conditions():
    print('\nRunning _test_actual_conditions...')
    iq, cc, bg, wv = actual_conditions('5%', '5%', '5%', '5%')
    print('Test input/ouput: '+str([iq, cc, bg, wv]) + ' == [0.2, 0.2, 0.2, 0.5]')
    assert([iq, cc, bg, wv] == [0.2, 0.2, 0.2, 0.5])

    iq, cc, bg, wv = actual_conditions('59%', '59%', '59%', '59%')
    print('Test input/ouput: '+str([iq, cc, bg, wv]) + '  == [0.85, 0.7, 0.7, 0.7]')
    assert ([iq, cc, bg, wv] == [0.85, 0.7, 0.7, 0.7])

def _test_conditions():
    print('\nRunning _test_conditions...')
    testiq = np.array(['10%', '20%', '50%', '70%', '80%', '85%', 'Any', 'null'])
    testcc = np.array(['10%', '20%', '50%', '70%', '80%', '85%', 'Any', 'null'])
    testbg = np.array(['10%', '20%', '50%', '70%', '80%', '85%', 'Any', 'null'])
    testwv = np.array(['10%', '20%', '50%', '70%', '80%', '85%', 'Any', 'null'])

    print('Test input: ')
    print('\tiq: ',testiq)
    print('\tcc: ', testcc)
    print('\tbg: ', testbg)
    print('\twv: ', testwv)
    iq, cc, bg, wv = conditions( testiq, testcc, testbg, testwv)
    print('Test ouput: ')
    print('\tiq: ',iq)
    print('\tcc: ', cc)
    print('\tbg: ', bg)
    print('\twv: ', wv)
    assert ((iq == np.array([0.1, 0.2, 0.5, 0.5, 0.8, 0.85, 1., 1.])).all())
    assert ((cc == np.array([0.1, 0.2, 0.5, 0.7, 0.8, 0.85, 1., 1.])).all())
    assert ((bg == np.array([0.1, 0.2, 0.5, 0.7, 0.8, 0.85, 1., 1.])).all())
    assert ((wv == np.array([0.1, 0.2, 0.2, 0.7, 0.8, 0.85, 1., 1.])).all())

if __name__=='__main__':
    _test_actual_conditions()
    _test_conditions()
    print('Test successful!')