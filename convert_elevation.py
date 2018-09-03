# Convert elevation constraints in catalog file to python dictionary
# Matt Bonnyman 2018-08-12

import re
import astropy.units as u


def convert_elevation(elev_const):
    """
    Convert elevation constraint for observation and return as a dictionary
        of the form {'type':string,'min':float,'max':float}

    Input
    -------
    elev_const : string
        elevation constraint type and limits
        (eg. elev_const = '{Hour Angle -2.00 2.00}')

    Returns
    --------
    dictionary
        (eg. elev = {'type':'Airmass', 'min':0.2, 'max':0.8})
    """

    if (elev_const.find('None') != -1) or (elev_const.find('null') != -1) or (elev_const.find('*NaN') != -1):
        return {'type': 'None', 'min': 0., 'max': 0.}
    elif elev_const.find('Hour') != -1:
        nums = re.findall(r'[+-]?\d+(?:\.\d+)?', elev_const)
        return {'type': 'Hour Angle', 'min': float(nums[0]) * u.hourangle, 'max': float(nums[1]) * u.hourangle}
    elif elev_const.find('Airmass') != -1:
        nums = re.findall(r'[+-]?\d+(?:\.\d+)?', elev_const)
        return {'type': 'Airmass', 'min': float(nums[0]), 'max': float(nums[1])}
    else:
        raise TypeError('Could not determine elevation constraint from string: ', elev_const)
        return


def test_convelev():
    hrconst = convert_elevation('{Hour Angle -2.00 2.00}')
    print(hrconst)
    assert hrconst['type'] == 'Hour Angle'
    assert hrconst['min'] == -2. * u.hourangle
    assert hrconst['max'] == 2. * u.hourangle

    amconst = convert_elevation('{Airmass 1.00 1.80}')
    print(amconst)
    assert amconst['type'] == 'Airmass'
    assert amconst['min'] == 1.
    assert amconst['max'] == 1.8

    print('Test successful!')

if __name__=='__main__':
    test_convelev()