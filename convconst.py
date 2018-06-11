import numpy as np
import re



def convelev(elev_const):
    """Convert elevation constraints"""
    if (elev_const.find('None')!=-1) or (elev_const.find('null')!=-1)  or (elev_const.find('*NaN')!=-1):
        min = 0.
        max = 0.
        type = 'None'
        #print('Read none, null, or *NaN',elev_const)
    elif elev_const.find('Hour')!=-1:
        nums = re.findall(r'\d+.\d+',elev_const)
        min = nums[0]
        max = nums[1]
        type = 'Hour Angle'
        #print('Read Hour',elev_const)
    elif elev_const.find('Airmass')!=-1:
        nums = re.findall(r'\d+.\d+',elev_const)
        #print('Read Airmass',AirmassConstraint(min=float(nums[0]),max=float(nums[1])))
        min = nums[0]
        max = nums[1]
        type = 'Airmass'
    else:
        print('Could not read elevation constraint type: see convconst.elev_const... Input value = ',elev_const)
        None
    return type,np.float(min),np.float(max)




def convcond(iq,cc,bg,wv):
    """
    Convert weather condition contraints found in the catalog 
    to decimal values in range [0,1].
    Conditions 'Any' or 'null' are assigned 1.
    Percentages converted to decimals.
    Image qualities of 70% are set to 50%.
    Water vapour values of 50% are set to 20%.
    """

    #============ image quality =============
    for i in range(0,len(iq)):
        if np.logical_or('Any' in iq[i],'null' in iq[i]):
            iq[i]=1. #set all 'Any' or 'null' IQs to 1.
    iq = iq.astype('<U2') #trim array elements to 2 characters
    iq = np.asarray(iq,dtype='f8') #convert array to numpy type w/ double precision floats
    ii = np.where(iq!=1.)[0][:] #get indeces of percentage IQs
    iq[ii] = iq[ii]/100. #divide all IQ percentages by 100

    indeces = np.where(iq==0.7)[0][:]
    if len(indeces)!=0:
        iq[indeces] = 0.5 #change 70% image qualities to 50%


    #========= cloud condition ==============
    for i in range(0,len(cc)):
        if np.logical_or('Any' in cc[i],'null' in cc[i]):
            cc[i]=1. #set all 'Any' or 'null' CCs to 1.
    cc = cc.astype('<U2') #trim array elements to 2 characters
    cc = np.asarray(cc,dtype='f8') #convert array to numpy type w/ double precision floats
    ii = np.where(cc!=1.)[0][:] #get indeces of percentage CCs
    cc[ii] = cc[ii]/100. #divide all CC percentages by 100


    #=========== sky background =============
    for i in range(0,len(bg)):
        if np.logical_or('Any' in bg[i],'null' in bg[i]):
            bg[i]=1. #set all 'Any' or 'null' BGs to 1.
    bg = bg.astype('<U2') #trim array elements to 2 characters
    bg = np.asarray(bg,dtype='f8') #convert array to numpy type w/ double precision floats
    ii = np.where(bg!=1.)[0][:] #get indeces of percentage BGs
    bg[ii] = bg[ii]/100. #divide all BG percentages by 100

    #============= water vapour =============
    for i in range(0,len(wv)):
        if np.logical_or('Any' in wv[i],'null' in wv[i]):
            wv[i]=1. #set all 'Any' or 'null' WVs to 1.
    wv = wv.astype('<U2') #trim array elements to 2 characters
    wv = np.asarray(wv,dtype='f8') #convert array to numpy type w/ double precision floats
    ii = np.where(wv!=1.)[0][:] #get indeces of percentage WVs
    wv[ii] = wv[ii]/100. #divide all WV percentages by 100

    indeces = np.where(wv==0.5)[0][:]
    if len(indeces)!=0: 
        wv[indeces] = 0.2 #change 50% water vapour values to 20%

    return iq,cc,bg,wv 


