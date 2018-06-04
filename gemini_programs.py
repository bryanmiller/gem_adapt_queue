import numpy as np
import astropy.units as u
from astropy.time import Time

__all__ = ["Gemini_programs"]


def conv_time(array_times):
    """
    Convert time string from hh:mm:ss.ss to hour quantity
    """
    hour = []
    values = np.core.defchararray.split(array_times,':')
    hour = [(float(value[0]) + float(value[1])/60 + float(value[2])/3600)*u.h for value in values]
    return hour


class Gprogram(object):
    
    """ 
    A container class for information about Gemini programs. 
    

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



class Gobservation(object):
    
    """ 
    A container class for information about a single Gemini observation
    over a continuous time range. 

    Definitions
    --------
    id              (string)                unique observation id
    ra              (float)                 hour quantity of allocated program time
    dec             (float)                 hour quantity of time charged to program
    ZD              (array of floats)                 hour quantity of time charged to partner
    HA              (array of floats)               program ready for scheduling 
    AZ              (array of floats)          date of program activation
    AM              (array of floats)          date of program deactivation
    mdist           (array of floats)  
    sbcond          (array of floats)               1=None, 2=standard, 3=rapid
    weight          (array of floats)               Science rank, band, or TAC ranking
    iobswin         (int. array len=2)      Identifiers of related observations
    
    """
    
    #@u.quantity_input(allocated_time=u.h, charged_time=u.h, partner_time=u.h)
    def __init__(self, n, id=None, ra=None, dec=None, ZD=None, \
                HA=None, AZ=None, AM=None, \
                mdist=None, sbcond=None, weight=None, iobswin=None, \
                wmax=0.):

        self.name = id
        self.ra = ra
        self.dec = dec
        self.ZD = np.zeros(n)
        self.HA = np.zeros(n)
        self.AZ = np.zeros(n)
        self.AM = np.zeros(n)
        self.mdist = np.zeros(n)
        self.sbcond = np.zeros(n)
        self.weight = np.zeros(n)
        self.iobswin = np.zeros(2)
        self.wmax = wmax


# obs_dict = {'id':'None',\
#             'ra':0.0,\
#             'dec':0.0,\
#             'ZD':np.zeros(n_timesteps),\
#             'HA':np.zeros(n_timesteps),\
#             'AZ':np.zeros(n_timesteps),\
#             'AM':np.zeros(n_timesteps),\
#             'mdist':np.zeros(n_timesteps),\
#             'sbcond':np.empty(n_timesteps,dtype='U4'),\
#             'weight':np.zeros(n_timesteps),\
#             'iobswin':[0,0],\
#             'wmax':0.0}

class Ggroup(object):
    
    """ 
    A container class for information about a program's 
    status, partners, observations... 
    

    Definitions
    --------
    gemgrid         (string)                unique group id
    name            (string)                string name
    type            (string/int.val)        folder/scheduling/other
    total_time      (float)                 hour quantity of total time
    timing_constraints                      inherent from observations?
    
    """
    
    #@u.quantity_input(allocated_time=u.h, charged_time=u.h, partner_time=u.h)
    def __init__(self, gemgrid=None, name=None, nametype=None, \
                total_time=None, observations=None, timing_constraints=None):

        self.gemgrid = gemgrid
        self.name = name
        self.type = nametype
        self.total_time = conv_time(total_time)
        self.observations = observations
        self.timing_constraints = timing_constraints

    

