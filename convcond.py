
import numpy as np
import re

#read lists of condition contraints and convert to numerical values
def convert_array(iq,cc,bg,wv):
    """This function converts the weather conditions found in the catalog into 
    numpy arrays of floats ranging from 0-1.
    All conditions containing 'Any' or 'null' are set to 1.
    All percentage conditions are divided by 100.
    Image qualities of 70% are changed to 50%.
    Water vapour values of 50% are changed to 20%."""

    #============ convert image quality values =============
    for i in range(0,len(iq)):
        if np.logical_or('Any' in iq[i],'null' in iq[i]):
            iq[i]=1. #set all 'Any' or 'null' IQs to 1.
    iq = iq.astype('<U2') #trim array elements to 2 characters
    iq = np.asarray(iq,dtype='f8') #convert array to numpy type w/ double precision floats
    ii = np.where(iq!=1.)[0][:] #get indeces of percentage IQs
    iq[ii] = iq[ii]/100. #divide all IQ percentages by 100

    indeces = np.where(iq==0.7)[0][:]
    if len(indeces)!=0: #change 70% image qualities to 50%
        iq[indeces] = 0.5


    #========= convert cloud condition values ==============
    for i in range(0,len(cc)):
        if np.logical_or('Any' in cc[i],'null' in cc[i]):
            cc[i]=1. #set all 'Any' or 'null' CCs to 1.
    cc = cc.astype('<U2') #trim array elements to 2 characters
    cc = np.asarray(cc,dtype='f8') #convert array to numpy type w/ double precision floats
    ii = np.where(cc!=1.)[0][:] #get indeces of percentage CCs
    cc[ii] = cc[ii]/100. #divide all CC percentages by 100


    #=========== convert sky background values =============
    for i in range(0,len(bg)):
        if np.logical_or('Any' in bg[i],'null' in bg[i]):
            bg[i]=1. #set all 'Any' or 'null' BGs to 1.
    bg = bg.astype('<U2') #trim array elements to 2 characters
    bg = np.asarray(bg,dtype='f8') #convert array to numpy type w/ double precision floats
    ii = np.where(bg!=1.)[0][:] #get indeces of percentage BGs
    bg[ii] = bg[ii]/100. #divide all BG percentages by 100

    #============= convert water vapour values =============
    for i in range(0,len(wv)):
        if np.logical_or('Any' in wv[i],'null' in wv[i]):
            wv[i]=1. #set all 'Any' or 'null' WVs to 1.
    wv = wv.astype('<U2') #trim array elements to 2 characters
    wv = np.asarray(wv,dtype='f8') #convert array to numpy type w/ double precision floats
    ii = np.where(wv!=1.)[0][:] #get indeces of percentage WVs
    wv[ii] = wv[ii]/100. #divide all WV percentages by 100

    indeces = np.where(wv==0.5)[0][:]
    if len(indeces)!=0: #change 50% water vapour values to 20%
        wv[indeces] = 0.2

    return iq,cc,bg,wv 


