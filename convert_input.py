import re
import numpy as np
import astropy.units as u
from astropy import time


def site(site_name,dst):

    """
    Check observatory name input.
    Return:
        - astroplan.Observer object site name ('gemini_south' or 'gemini_north')
        - time difference between local and utc for observatory location
        - pytz object timezone name ('Chile/Continental' or 'US/Hawaii')
    """

    if np.logical_or(site_name=='gemini_south',site_name=='CP'):
        site_name = 'gemini_south'
        timezone_name = 'Chile/Continental'
        if dst == True:
            utc_to_local = -3.*u.h
        else:
            utc_to_local = -4.*u.h 
    elif np.logical_or(site_name=='gemini_north',site_name=='MK'):
        site_name = 'gemini_north'
        timezone_name = 'US/Hawaii'
        utc_to_local = -10.*u.h
    else:
        print('Input error: Could not determine observer location and timezone. Allowed inputs are \'gemini_south\', \'CP\'(Cerro Pachon),\'gemini_north\', or \'MK\'(Mauna Kea).)')
        exit()

    return site_name,timezone_name,utc_to_local



def dates(startdate,enddate,utc_to_local):

    """
    Check startdate and enddate input.
    Return
        - astropy.time.Time object for 18:00:00 local time on the startdate
        - List of ints counting days between startdate and enddate 
            with day units (day_nums = [0*u.d,1*u.d,2*u.d,...]).
    """

    dform = re.compile('\d{4}-\d{2}-\d{2}') #yyyy-mm-dd format
    
    if startdate == None:
        current_time = time.Time.now() + utc_to_local    
        start_time = time.Time(str(current_time)[0:10] + ' 18:00:00.00') #18:00 local on first night
    else:
        if dform.match(startdate): #check startdate format
            start_time = time.Time(startdate + ' 18:00:00.00') #18:00 local on first night
        else:
            print('\nInput error: \"'+startdate+'\" not a valid start date.  Must be in the form \'yyyy-mm-dd\'')
            exit()

    if enddate == None: #default number of observation nights is 1
        day_nums = [0]
    else:
        if dform.match(enddate): #check enddate format
            end_time = time.Time(enddate+' 18:00:00.00') #time object of 18:00:00 local time  
            d = int((end_time - start_time).value + 1) #number of days between startdate and enddate
            if d >= 0: 
                day_nums = np.arange(d) #list of ints with day units e.g. [0*u.d,1*u.d,2*u.d,...]
            else: 
                print('\nInput error: Selected end date \"'+enddate+'\" is prior to the start date.')
                exit()
        else:
            print('\nInput error: \"'+enddate+'\" not a valid end date.  Must be in the form \'yyyy-mm-dd\'')
            exit()

    return start_time, day_nums*u.d


#Perform tests
if __name__=='__main__':
        
    print('\nTesting convert_input_dates():')
    
    a = '2020-01-01'
    b = '2020-12-31'
    c = -12.*u.h
    d,e = convert_input_dates(a,b,c)
    testtime = time.Time('2020-01-01 18:00:00.00')
    
    if d!=testtime:
        print('Test failure!')
        print(d,'!=',testtime)
        
    if len(e)!=366:
        print('Test failure!')
        print(len(e),'!=',366)
        
    