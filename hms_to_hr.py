# Matt Bonnyman 25 July 2018

import numpy as np

def hms_to_hr(timestring):
    """
    Convert time string of the form 'hh:mm:ss.ss' to float

    Parameter
    ---------
    timestring : string
        Time string in the form 'hh:mm:ss.ss'.  Seconds are accepted with any number of significant digits.

    Return
    ---------
    float
        equivalent decimal hour value
    """
    (h, m, s) = timestring.split(':')
    return np.int(h) + np.int(m)/60 + np.float(s)/3600

def test_hms():
    string = '16:45:00'
    hours = hms_to_hr(string)
    print('16:45:00 = ', hours)
    assert hours == 16.75
    print('Test successful!')
    return

if __name__=='__main__':
    test_hms()
