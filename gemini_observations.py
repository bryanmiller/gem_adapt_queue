import re
import numpy as np
import astropy.units as u
from conversions import convertConditions
from astropy import time
from select_obs import selectqueue
from astropy.coordinates import (SkyCoord, FK5)

def hms_to_hr(timestring):
    """
    Convert time string of the form 'HH:MM:SS' to float

    Parameter
    ---------
    timestring : string
        'HH:MM:SS:

    Return
    ---------
    float
        number of hours
    """
    (h, m, s) = timestring.split(':')
    return (np.int(h) + np.int(m)/60 + np.int(s)/3600)

def convertElevConst(elev_const):

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

def convertTimeConst(windows):

    """
    Convert Time Constraints values in catalog to appropriate types

    Parameter
    ----------
    windows : list of strings
        Observation time constraints.
        Format: {1526872703000 1800000 100 140740000}
            value 1 : start time in unix time milliseconds
            value 2 : duration of time window in milliseconds
            value 3 : number of window repeats
            value 4 : period between repeats in milliseconds

    Return
    ---------
    converted time constraints : list of converted elements (dictionaries or 'None')
        {'start':'astropy.time.Time', 'dur':'astropy.units.Quantity', 'reps':int, 'per':'astropy.units.Quantity'}
    """

    newtimeconst = []

    infinity = 3. * 365. * 24. * u.h

    for i in range(0, len(windows)):
        otstr = re.sub('[\[{}\]]', '', windows[i])
        winlist = otstr.split(',')
        if len(winlist) != 0 and winlist[0] != '':
            obs_wins = []
            for string in winlist:
                vals = re.findall(r'[+-]?\d+(?:\.\d+)?', string)

                t0 = float(vals[0]) * u.ms  # start time in unix time milliseconds
                start = time.Time(t0.to_value('s'), format='unix', scale='utc')

                duration = float(vals[1])
                if duration == -1.0:  # -1=forever
                    pass
                    # duration = infinity
                else:
                    duration = duration / 3600000. * u.h

                repeat = int(vals[2])
                if repeat == -1:  # -1=forever
                    repeat = 1000

                period = float(vals[3]) / 3600000. * u.h
                obs_wins.append({'start': start, 'duration': duration, 'repeat': repeat, 'period': period})

            newtimeconst.append(obs_wins)
        else:
            newtimeconst.append(None)
        # print(windows[i])
        # print(new_windows[i])
    return newtimeconst

class Gobservations(object):
    """
    A container class for gemini program information.

    Requires Gcatalog object as input (see gemini_classes.Gcatalog()).
    Automatically converts some variable types,
    performs merging of some columns, and performs computation
    and conversion of elevation constraints, conditions contraints, completed time,
    and total time.


    Attributes
    --------
    prog_ref            (string)                unique program identifier
    obs_id              (string)                unique observation identifier
    pi                  (string)
    inst                (string)                instrument name
    target              (string)                target name
    ra                  (astropy degrees)       right ascension degrees
    dec                 (astropy degrees)       declination degrees
    band                (int)                   integer
    partner             (string)                gemini partner name
    obs_status          (string)                'ready' status of observation
    tot_time            (astropy hours)         total planned observation time
    obs_time            (astropy hours)         completed observation time
    obs_comp            (float)                 fraction of completed/total observation time
    charged_time        (string)                HH:MM:SS (required to compute obs_comp)
    obs_class           (string)                observation class
    iq                  (float)                 image quality constraint (percentile converted to decimal value)
    cc                  (float)                 cloud condition constraint (percentile converted to decimal value)
    bg                  (float)                 sky background constraint (percentile converted to decimal value)
    wv                  (float)                 water vapor constraint (percentile converted to decimal value)
    user_prior          (string)                user priority
    group               (string)                observation group name
    elev_const          (dictionary)            elevation constraint {'type':string,'min':float,'max':float}
    ready               (boolean)               ready status
    disperser           (string)                disperser name
    fpu                 (string)                focal plane unit
    grcwlen             (string)                grating control wavelength
    crwlen              (string)                central wavelength
    filter              (string)                filter name
    mask                (string)                mask name
    xbin                (string)                xbin number
    ybin                (string)                ybin number
    """

    def __init__(self, catinfo, epoch=time.Time.now()):

        timer = False
        if timer:
            import time as t
            timerstart = t.time()  # runtime clock

        # Select observations from catalog input to queue
        i_obs = selectqueue(catinfo=catinfo)

        n_obs = len(i_obs)

        # interpret and store catalog data
        self.prog_ref = catinfo.prog_ref[i_obs]
        self.obs_id = catinfo.obs_id[i_obs]
        self.pi = catinfo.pi[i_obs]
        self.inst = catinfo.inst[i_obs]
        self.target = catinfo.target[i_obs]
        self.ra = np.array(list(map(float, catinfo.ra[i_obs]))) * u.deg
        self.dec = np.array(list(map(float, catinfo.dec[i_obs]))) * u.deg
        self.band = np.array(list(map(int, catinfo.band[i_obs])))
        self.partner = catinfo.partner[i_obs]
        self.obs_status = catinfo.obs_status[i_obs]
        # self.qa_status = catinfo.obs_qa[i_obs]
        # self.dataflow_step = catinfo.dataflow_step[i_obs]
        self.tot_time = catinfo.planned_exec_time[i_obs]
        self.obs_comp = np.zeros(n_obs)
        # self.planned_pi_time = catinfo.planned_pi_time[i_obs]
        self.charged_time = catinfo.charged_time[i_obs]
        self.obs_time = self.charged_time
        self.obs_class = catinfo.obs_class[i_obs]
        self.bg = catinfo.sky_bg[i_obs]
        self.wv = catinfo.wv[i_obs]
        self.cc = catinfo.cloud[i_obs]
        self.iq = catinfo.iq[i_obs]
        self.user_prior = catinfo.user_prio[i_obs]
        # self.ao = catinfo.ao[i_obs]
        self.group = catinfo.group[i_obs]
        # self.group_type = catinfo.gt[i_obs]
        self.elev_const = [convertElevConst(catinfo.elev_const[i]) for i in i_obs]
        self.time_const = convertTimeConst(catinfo.time_const[i_obs])
        self.ready = np.array(list(map(bool, catinfo.ready[i_obs])))
        # self.color_filter = catinfo.color_filter[i_obs]
        # self.nd_filter = catinfo.neutral_density_filter[i_obs]
        # self.binning = catinfo.binning[i_obs]
        # self.windowing = catinfo.windowing[i_obs]
        # self.lens = catinfo.lens[i_obs]
        # self.cass_rotator = catinfo.cass_rotator[i_obs]
        # self.bh_ccdamps = catinfo.ccd_amplifiers[i_obs]
        # self.bh_ccdgain = catinfo.ccd_gain[i_obs]
        # self.bh_ccdspeed = catinfo.ccd_speed[i_obs]
        self.bh_xbin = catinfo.ccd_x_binning[i_obs]
        self.bh_ybin = catinfo.ccd_y_binning[i_obs]
        # self.bh_fibre = catinfo.entrance_fibre[i_obs]
        # self.bh_expmeter_filter = catinfo.exposure_meter_filter[i_obs]
        # self.bh_hartmann = catinfo.hartmann_flap[i_obs]
        # self.bh_issport = catinfo.iss_port[i_obs]
        # self.bh_pslitfilter = catinfo.post_slit_filter[i_obs]
        # self.bh_roi = catinfo.region_of_interest[i_obs]
        self.f2_disperser = catinfo.disperser[i_obs]
        self.f2_filter = catinfo.filter[i_obs]
        # self.f2_readmode = catinfo.read_mode[i_obs]
        # self.f2_lyot = catinfo.lyot_wheel[i_obs]
        self.f2_fpu = catinfo.focal_plane_unit[i_obs]
        self.grcwlen = catinfo.grating_ctrl_wvl[i_obs]
        self.xbin = catinfo.x_bin[i_obs]
        self.ybin = catinfo.y_bin[i_obs]
        # self.roi = catinfo.builtin_roi[i_obs]
        # self.nodshuffle = catinfo.nod_shuffle[i_obs]
        # self.dtax = catinfo.dta_x_offset[i_obs]
        # self.custom_mask = catinfo.custom_mask_mdf[i_obs]
        # self.preimage = catinfo.mos_pre_imaging[i_obs]
        # self.amp_count = catinfo.amp_count[i_obs]
        self.disperser = catinfo.disperser_2[i_obs]
        self.filter = catinfo.filter_2[i_obs]
        self.fpu = catinfo.fpu[i_obs]
        # self.detector = catinfo.detector_manufacturer[i_obs]
        # self.FIELD063 = catinfo.grating_ctrl_wvl_2[i_obs]
        # self.FIELD064 = catinfo.x_bin_2[i_obs]
        # self.FIELD065 = catinfo.y_bin_2[i_obs]
        # self.FIELD066 = catinfo.builtin_roi_2[i_obs]
        # self.FIELD067 = catinfo.nod_shuffle_2[i_obs]
        # self.FIELD068 = catinfo.dta_x_offset_2[i_obs]
        # self.FIELD069 = catinfo.custom_mask_mdf_2[i_obs]
        # self.FIELD070 = catinfo.mos_pre_imaging_2[i_obs]
        # self.FIELD071 = catinfo.amp_count_2[i_obs]
        # self.FIELD072 = catinfo.disperser_3[i_obs]
        # self.FIELD073 = catinfo.filter_3[i_obs]
        # self.FIELD074 = catinfo.fpu_2[i_obs]
        # self.FIELD075 = catinfo.detector_manufacturer_2[i_obs]
        # self.pixel_scale = catinfo.pixel_scale[i_obs]
        # self.FIELD077 = catinfo.disperser_4[i_obs]
        # self.FIELD078 = catinfo.focal_plane_unit_2[i_obs]
        # self.cross_dispersed = catinfo.cross_dispersed[i_obs]
        # self.FIELD080 = catinfo.read_mode_2[i_obs]
        self.crwlen = catinfo.central_wavelength[i_obs]
        # self.iss_port = catinfo.iss_port_2[i_obs]
        # self.FIELD083 = catinfo.well_depth[i_obs]
        # self.FIELD084 = catinfo.filter_4[i_obs]
        # self.readmode = catinfo.read_mode_3[i_obs]
        # self.astrometric = catinfo.astrometric_field[i_obs]
        # self.FIELD087 = catinfo.disperser_5[i_obs]
        # self.adc = catinfo.adc[i_obs]
        # self.observing_mode = catinfo.observing_mode[i_obs]
        # self.coadds = catinfo.coadds[i_obs]
        # self.exptime = catinfo.exposure_time[i_obs]
        # self.FIELD092 = catinfo.disperser_6[i_obs]
        self.mask = catinfo.mask[i_obs]
        # self.eng_mask = catinfo.engineering_mask[i_obs]
        # self.FIELD095 = catinfo.filter_[i_obs]
        # self.order = catinfo.disperser_order[i_obs]
        # self.nici_fpu = catinfo.focal_plane_mask[i_obs]
        # self.nici_pupil = catinfo.pupil_mask[i_obs]
        # self.nici_cassrot = catinfo.cass_rotator_2[i_obs]
        # self.nici_imgmode = catinfo.imaging_mode[i_obs]
        # self.nici_dichroic = catinfo.dichroic_wheel[i_obs]
        # self.nici_fw1 = catinfo.filter_red_channel[i_obs]
        # self.nici_fw2 = catinfo.filter_blue_channel[i_obs]
        # self.nici_welldepth = catinfo.well_depth_2[i_obs]
        # self.nici_dhs = catinfo.dhs_mode[i_obs]
        # self.imaging_mirror = catinfo.imaging_mirror[i_obs]
        # self.FIELD107 = catinfo.disperser_7[i_obs]
        # self.FIELD108 = catinfo.mask_2[i_obs]
        # self.FIELD109 = catinfo.filter_6[i_obs]
        # self.FIELD110 = catinfo.read_mode_4[i_obs]
        # self.camera = catinfo.camera[i_obs]
        # self.FIELD112 = catinfo.disperser_8[i_obs]
        # self.FIELD113 = catinfo.mask_3[i_obs]
        # self.FIELD114 = catinfo.filter_7[i_obs]
        # self.beam_splitter = catinfo.beam_splitter[i_obs]
        # self.FIELD116 = catinfo.read_mode_5[i_obs]
        # self.FIELD117 = catinfo.mask_4[i_obs]
        # self.FIELD118 = catinfo.filter_8[i_obs]
        # self.FIELD119 = catinfo.disperser_9[i_obs]
        # self.FIELD120 = catinfo.disperser_10[i_obs]
        # self.FIELD121 = catinfo.mask_5[i_obs]
        # self.FIELD122 = catinfo.filter_9[i_obs]

        # ---------------- Combine columns ----------------------
        ii = np.where(self.f2_fpu != 'null')[0][:]
        if len(ii) != 0:
            self.fpu[ii] = self.f2_fpu[ii]
        ii = np.where(self.crwlen != 'null')[0][:]
        if len(ii) != 0:
            self.grcwlen[ii] = self.crwlen[ii]
        ii = np.where(self.bh_xbin != 'null')[0][:]
        if len(ii) != 0:
            self.xbin[ii] = self.bh_xbin[ii]
        ii = np.where(self.bh_ybin != 'null')[0][:]
        if len(ii) != 0:
            self.ybin[ii] = self.bh_ybin[ii]
        ii = np.where(self.mask != 'null')[0][:]
        if len(ii) != 0:
            self.fpu[ii] = self.mask[ii]
        ii = np.where(self.group == '')[0][:]
        if len(ii) != 0:
            self.group[ii] = self.obs_id[ii]
        ii = np.where(self.charged_time == '')[0][:]
        if len(ii) != 0:
            self.charged_time[ii] = '00:00:00'

        # ------- Format and convert condition constraints --------
        for i in range(0, n_obs):
            if self.cc[i][0:1] == 'P': self.cc[i] = self.cc[i][8:10]
            if self.cc[i][0:1] == 'A': self.cc[i] = 'Any'
            if self.iq[i][0:1] == 'P': self.iq[i] = self.iq[i][8:10]
            if self.iq[i][0:1] == 'A': self.iq[i] = 'Any'
            if self.wv[i][0:1] == 'P': self.wv[i] = self.wv[i][8:10]
            if self.wv[i][0:1] == 'A': self.wv[i] = 'Any'
        self.iq, self.cc, self.bg, self.wv = convertConditions(self.iq, self.cc, self.bg, self.wv)

        # ---------- Format and convert observation times -----------
        for i in range(0, n_obs):
            charged_time = hms_to_hr(self.charged_time[i])
            tot_time = hms_to_hr(self.tot_time[i])
            if (charged_time > 0.):  # add additional time
                if 'Mirror' in self.disperser[i]:
                    tot_time = tot_time + 0.2
                else:
                    tot_time = tot_time + 0.3
            self.obs_comp[i] = charged_time / tot_time  # completion fraction
            self.tot_time[i] = tot_time  # total time required
            self.obs_time[i] = charged_time  # observed time
        self.obs_time = np.array(self.obs_time, dtype=float) * u.h
        self.tot_time = np.array(self.tot_time, dtype=float) * u.h

        # --------- Get target coordinates for current epoch ---------
        coord_j2000 = SkyCoord(self.ra, self.dec, frame='icrs', unit=(u.deg, u.deg))
        current_epoch = coord_j2000.transform_to(FK5(equinox='J' + str(epoch.jyear)))
        self.ra = current_epoch.ra
        self.dec = current_epoch.dec

        if timer: print('\n\tInitialize Gobservations: ', t.time() - timerstart)  # runtime clock


class Gcatfile(object):
    """
    - Rough version of a container class for reading and storing catalog information.
    - Reads catalog file columns as lists w/ attribute names matching headers in OT file.
    - Attribute naming convention: lowercase column headers w/ underscores (eg.'Obs. Status'=obs_status).
    - Repeated column names are not combined into a single attribute name.  i.e. If an column name is repeated,
        a numerical value is appended to the column header (eg. second disperser column-->'disperser_2').
    """

    def __init__(self, otfile=None):
        cattext = []
        with open(otfile, 'r') as readcattext:  # read file into memory.
            # [print(line.split('\t')) for line in readcattext]
            [cattext.append(line.split('\t')) for line in readcattext]  # Split lines where tabs ('\t') are found.
            readcattext.close()
        names = np.array(cattext[8])
        obsall = np.array(cattext[10:])
        # print('\notcat attribute names...',names)
        existing_names = []
        for i in range(0, len(names)):
            # remove special charaters, trim ends of string, replace whitespace with underscore
            string = names[i].replace('.', '')
            string = re.sub(r'\W', ' ', string)
            string = string.strip()
            string = re.sub(r' +', '_', string)
            string = string.lower()
            if string == 'class':  # change attribute name (python doesn't allow attribute name 'class')
                string = 'obs_class'
            if np.isin(string, existing_names):  # add number to end of repeated attribute name
                rename = True
                j = 0
                while rename:
                    if j >= 8:  # if number 9 is reached, make next number 10
                        tempstring = string + '_1' + chr(50 + j - 10)
                    else:
                        tempstring = string + '_' + chr(50 + j)
                    if np.isin(tempstring, existing_names):  # if name taken, increment number and check again
                        j += 1
                    else:
                        string = tempstring
                        rename = False
            setattr(self, string, obsall[:, i])  # set attribute name an include corresponding catalog column
            existing_names.append(string)  # add name to library of used names
            # print(string)
        # print('\nFound '+str(len(obsall))+' observations in '+str(otfile))