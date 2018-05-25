import numpy as np
import astropy.units as u
from astropy.time import Time

__all__ = ["Gemini_programs"]

def conv_time(array_times):
    """
    Convert time string in form hh:mm:ss.ss to hours
    """
    hour = []
    values = np.core.defchararray.split(array_times,':')
    hour = [(float(value[0]) + float(value[1])/60 + float(value[2])/3600)*u.h for value in values]
    return hour

class Gemini_programs(object):
    
    """ 
    A container class for information about a program's 
    status, partners, observations... 
    

    Definitions
    --------
    gemprgid        (string)                unique program id
    partner         (string/int index)      partner name
    pi              (string)                principle investigator
    allocated_time  (float)                 hour quantity of allocated program time
    program_time    (float)                 hour quantity of time charged to program
    parter_time     (float)                 hour quantity of time charged to partner
    active          (boolean)               program ready for scheduling 
    progstart       (astropy.time)          date of program activation
    progend         (astropy.time)          date of program deactivation
    completed       (boolean)  
    too_status      (integer)               1=None, 2=standard, 3=rapid
    scirank         (integer)               Science rank, band, or TAC ranking
    obs             (list of objects)?      Identifiers of related observations
    
    """
    
    #@u.quantity_input(allocated_time=u.h, charged_time=u.h, partner_time=u.h)
    def __init__(self, gemprgid=None, partner=None, pi=None, \
                allocated_time=None, charged_time=None, partner_time=None, \
                active=False, progstart=None, progend=None, completed=False, \
                too_status=1, scirank=None, obs=None):

        self.gemprgid = gemprgid
        self.partner = partner
        self.pi = pi
        self.allocated_time = conv_time(allocated_time)
        self.charged_time = conv_time(charged_time)
        self.partner_time = conv_time(partner_time)
        self.active = active
        self.progstart = progstart
        self.progend = progend
        self.completed = completed
        self.too_status = too_status
        self.scirank = scirank
        self.obs = obs 
    

