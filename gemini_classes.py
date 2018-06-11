import re
import numpy as np
import astropy.units as u
import convconst
from astropy.time import Time


__all__ = ["Gemini_classes"]




def conv_time(array_times):
    """
    Convert time string from hh:mm:ss.ss to hour quantity
    """
    hour = []
    values = np.core.defchararray.split(array_times,':')
    hour = [(float(value[0]) + float(value[1])/60 + float(value[2])/3600)*u.h for value in values]
    return hour




class Gelevconst(object):
    """
    A container class for elevation constraints
    """
    def __init__(self, elevconst = None):

        n = len(elevconst)
        elevtype = np.empty(n,dtype='U15')
        elevmin = np.zeros(n)
        elevmax = np.zeros(n)

        for i in range(n):
            elevtype[i], elevmin[i], elevmax[i] = convconst.convelev(elevconst[i])    

        self.type = elevtype
        self.min = elevmin
        self.max = elevmax




class Gcondition(object):
    """ 
    A container class for iq, cc, bg, and wv conditions. 
    """
    def __init__(self, iq=None, cc=None, bg=None, wv=None):
        
        iq,cc,bg,wv = convconst.convcond(iq,cc,bg,wv)

        self.iq = iq
        self.cc = cc
        self.bg = bg
        self.wv = wv



class Gprogstatus(object):
    """ 
    A container class for information about Gemini programs. 
    

    Definitions
    --------
    prog_id        (string)                unique program id
    obs_id         (string)                unique observation id
    target         (string)                target id
    band           (int)                   priority band 1,2,3 or 4.
    comp_time      (float)                 fraction of total time completed
    tot_time       (float)                 hour quantity of remaining observation time
    obs_time       (float)                 hour quantity of scheduled obs. time [DEF=0.0]
    """
    #@u.quantity_input(allocated_time=u.h, charged_time=u.h, partner_time=u.h)
    def __init__(self, prog_id=None, obs_id=None, target=None, \
                band=None, comp_time=None, tot_time=None, obs_time=False):

        self.prog_id = prog_id
        self.obs_id = obs_id
        self.target = target
        self.band = band
        self.comp_time = comp_time
        self.tot_time = tot_time
        self.obs_time = obs_time

        # Old method for storing this information...
        # prog_status = np.empty(n_obs,dtype={'names':('prog_id','obs_id','target','band','comp_time','tot_time','obs_time'),'formats':('U30','U30','U60','i8','f8','f8','f8')})
        # prog_status['prog_id']=otcat.prog_ref[i_obs]
        # prog_status['obs_id']=otcat.obs_id[i_obs]
        # prog_status['target']=otcat.target[i_obs]
        # prog_status['band']=otcat.band[i_obs]
        # prog_status['comp_time']=charged_time/planned_exec_time
        # prog_status['tot_time']=planned_exec_time*u.h
        # prog_status['obs_time']=np.zeros(n_obs)*u.h


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
        self.allocated_time = allocated_time
        self.charged_time = charged_time
        self.partner_time = partner_time
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




class Gcatalog2018(object):
    """
    A Container class for all observation data in the OT catalog.
    This is a copy of the otcat organization in 
    Bryan Miller's IDL version. 
    """
    def __init__(self, catinfo=None):

        self.prog_ref = catinfo.prog_ref
        self.obs_id = catinfo.obs_id
        self.pi = catinfo.pi
        self.inst = catinfo.inst
        self.target = catinfo.target
        self.ra = catinfo.ra
        self.dec = catinfo.dec
        self.band = catinfo.band
        self.partner = catinfo.partner
        self.obs_status = catinfo.obs_status
        # self.qa_status = catinfo.obs_qa
        # self.dataflow_step = catinfo.dataflow_step
        self.planned_exec_time = catinfo.planned_exec_time
        self.planned_pi_time = catinfo.planned_pi_time
        self.charged_time = catinfo.charged_time
        self.obs_class = catinfo.obs_class
        self.bg = catinfo.sky_bg
        self.wv = catinfo.wv
        self.cc = catinfo.cloud
        self.iq = catinfo.iq
        self.user_prior = catinfo.user_prio
        # self.ao = catinfo.ao
        self.group = catinfo.group
        # self.group_type = catinfo.gt
        self.elev_const = catinfo.elev_const
        self.time_const = catinfo.time_const
        # self.ready = catinfo.ready
        # self.color_filter = catinfo.color_filter
        # self.nd_filter = catinfo.neutral_density_filter
        # self.binning = catinfo.binning
        # self.windowing = catinfo.windowing
        # self.lens = catinfo.lens
        # self.cass_rotator = catinfo.cass_rotator
        # self.bh_ccdamps = catinfo.ccd_amplifiers
        # self.bh_ccdgain = catinfo.ccd_gain
        # self.bh_ccdspeed = catinfo.ccd_speed
        self.bh_xbin = catinfo.ccd_x_binning
        self.bh_ybin = catinfo.ccd_y_binning
        # self.bh_fibre = catinfo.entrance_fibre
        # self.bh_expmeter_filter = catinfo.exposure_meter_filter
        # self.bh_hartmann = catinfo.hartmann_flap
        # self.bh_issport = catinfo.iss_port
        # self.bh_pslitfilter = catinfo.post_slit_filter
        # self.bh_roi = catinfo.region_of_interest
        self.f2_disperser = catinfo.disperser
        self.f2_filter = catinfo.filter
        # self.f2_readmode = catinfo.read_mode
        # self.f2_lyot = catinfo.lyot_wheel
        self.f2_fpu = catinfo.focal_plane_unit
        self.grcwlen = catinfo.grating_ctrl_wvl
        self.xbin = catinfo.x_bin
        self.ybin = catinfo.y_bin
        # self.roi = catinfo.builtin_roi
        # self.nodshuffle = catinfo.nod_shuffle
        # self.dtax = catinfo.dta_x_offset
        # self.custom_mask = catinfo.custom_mask_mdf
        # self.preimage = catinfo.mos_pre_imaging
        # self.amp_count = catinfo.amp_count
        self.disperser = catinfo.disperser_2
        self.filter = catinfo.filter_2
        self.fpu = catinfo.fpu
        # self.detector = catinfo.detector_manufacturer
        # self.FIELD063 = catinfo.grating_ctrl_wvl_2
        # self.FIELD064 = catinfo.x_bin_2
        # self.FIELD065 = catinfo.y_bin_2
        # self.FIELD066 = catinfo.builtin_roi_2
        # self.FIELD067 = catinfo.nod_shuffle_2
        # self.FIELD068 = catinfo.dta_x_offset_2
        # self.FIELD069 = catinfo.custom_mask_mdf_2
        # self.FIELD070 = catinfo.mos_pre_imaging_2
        # self.FIELD071 = catinfo.amp_count_2
        # self.FIELD072 = catinfo.disperser_3
        # self.FIELD073 = catinfo.filter_3
        # self.FIELD074 = catinfo.fpu_2
        # self.FIELD075 = catinfo.detector_manufacturer_2
        # self.pixel_scale = catinfo.pixel_scale
        # self.FIELD077 = catinfo.disperser_4
        # self.FIELD078 = catinfo.focal_plane_unit_2
        # self.cross_dispersed = catinfo.cross_dispersed
        # self.FIELD080 = catinfo.read_mode_2
        self.crwlen = catinfo.central_wavelength
        # self.iss_port = catinfo.iss_port_2
        # self.FIELD083 = catinfo.well_depth
        # self.FIELD084 = catinfo.filter_4
        # self.readmode = catinfo.read_mode_3
        # self.astrometric = catinfo.astrometric_field
        # self.FIELD087 = catinfo.disperser_5
        # self.adc = catinfo.adc
        # self.observing_mode = catinfo.observing_mode
        # self.coadds = catinfo.coadds
        # self.exptime = catinfo.exposure_time
        # self.FIELD092 = catinfo.disperser_6
        self.mask = catinfo.mask
        # self.eng_mask = catinfo.engineering_mask
        # self.FIELD095 = catinfo.filter_
        # self.order = catinfo.disperser_order
        # self.nici_fpu = catinfo.focal_plane_mask
        # self.nici_pupil = catinfo.pupil_mask
        # self.nici_cassrot = catinfo.cass_rotator_2
        # self.nici_imgmode = catinfo.imaging_mode
        # self.nici_dichroic = catinfo.dichroic_wheel
        # self.nici_fw1 = catinfo.filter_red_channel
        # self.nici_fw2 = catinfo.filter_blue_channel
        # self.nici_welldepth = catinfo.well_depth_2
        # self.nici_dhs = catinfo.dhs_mode
        # self.imaging_mirror = catinfo.imaging_mirror
        # self.FIELD107 = catinfo.disperser_7
        # self.FIELD108 = catinfo.mask_2
        # self.FIELD109 = catinfo.filter_6
        # self.FIELD110 = catinfo.read_mode_4
        # self.camera = catinfo.camera
        # self.FIELD112 = catinfo.disperser_8
        # self.FIELD113 = catinfo.mask_3
        # self.FIELD114 = catinfo.filter_7
        # self.beam_splitter = catinfo.beam_splitter
        # self.FIELD116 = catinfo.read_mode_5
        # self.FIELD117 = catinfo.mask_4
        # self.FIELD118 = catinfo.filter_8
        # self.FIELD119 = catinfo.disperser_9
        # self.FIELD120 = catinfo.disperser_10
        # self.FIELD121 = catinfo.mask_5
        # self.FIELD122 = catinfo.filter_9

        ii = np.where(self.f2_fpu != 'null')[0][:]
        if len(ii)!=0:
            self.fpu[ii]=self.f2_fpu[ii]

        ii = np.where(self.crwlen != 'null')[0][:]
        if len(ii)!=0:
            self.grcwlen[ii]=self.crwlen[ii]

        ii = np.where(self.bh_xbin != 'null')[0][:]
        if len(ii)!=0:
            self.xbin[ii]=self.bh_xbin[ii]

        ii = np.where(self.bh_ybin != 'null')[0][:]
        if len(ii)!=0:
            self.ybin[ii]=self.bh_ybin[ii]

        ii = np.where(self.mask != 'null')[0][:]
        if len(ii)!=0:
            self.fpu[ii]=self.mask[ii]

        ii = np.where(self.group == '')[0][:]
        if len(ii)!=0:
            self.group[ii]=self.obs_id[ii]   

        ii = np.where(self.charged_time == '')[0][:]
        if len(ii)!=0:
            self.charged_time[ii]='00:00:00'

        #Conditions contraints
        for i in range(len(catinfo.iq)):
            if self.cc[i][0:1] == 'P': self.cc[i] == self.cc[i][8:10]
            if self.cc[i][0:1] == 'A': self.cc[i] =='Any'
            if self.iq[i][0:1] == 'P': self.iq[i] == self.iq[i][8:10]
            if self.iq[i][0:1] == 'A': self.iq[i] =='Any'
            if self.wv[i][0:1] == 'P': self.wv[i] == self.wv[i][8:10]
            if self.wv[i][0:1] == 'A': self.wv[i] == 'Any'



class Gcatfile(object):

    """
    Rough version of a container class for storing catalog information.

    Reads file columns in to a dictionary w/ column names matching headers in OT file.
    
    Attribute naming convention: lowercase column headers w/ underscores (eg.'Obs. Status'=obs_status).

    Issue: repeated column names are not combined into a single attribute name.
    If an column name is repeated, a numerical value is appended to the corresponding attr. name (eg. 'disperser_2').
    """

    def __init__(self,otfile=None):

        cattext = []
        with open(otfile, 'r') as readcattext: #read file into memory. 
            #[print(line.split('\t')) for line in readcattext]
            [cattext.append(line.split('\t')) for line in readcattext] #Split lines where tabs ('\t') are found.
            readcattext.close()

        names = np.array(cattext[8])
        obsall = np.array(cattext[10:])

        #print('\notcat attribute names...',names)
        existing_names = []
        for i in range(0,len(names)):
            #remove special charaters, trim ends of string, replace whitespace with underscore
            string=names[i].replace('.','')
            string=re.sub(r'\W',' ',string)
            string=string.strip()
            string=re.sub(r' +','_',string)
            string=string.lower()
            if string=='class': #change attribute name (python doesn't allow attribute name 'class')
                string = 'obs_class'
            if np.isin(string,existing_names): #add number to end of repeated attribute name
                rename = True
                j = 0    
                while rename:
                    if j>=8: #if number 9 is reached, make next number 10
                        tempstring = string+'_1'+chr(50+j-10)
                    else:
                        tempstring = string+'_'+chr(50+j)
                    if np.isin(tempstring,existing_names): #if name taken, increment number and check again
                        j+=1
                    else: 
                        string = tempstring
                        rename = False

            setattr(self, string, obsall[:,i]) #set attribute name an include corresponding catalog column
            existing_names.append(string) #add name to library of used names
            #print(string)

        #print('\nFound '+str(len(obsall))+' observations in '+str(otfile))