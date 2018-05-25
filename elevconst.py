from astroplan import AirmassConstraint
import re
import numpy as np

def convert(elev_const):
    """Convert"""
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
        print('Could not read elevation constraint type: see elevconst.py',elev_const)
        None
    return type,min,max