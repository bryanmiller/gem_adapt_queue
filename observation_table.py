from astropy.coordinates import SkyCoord, FK5
from astropy.table import Table, Column
from astropy.time import Time
import astropy.units as u
import numpy as np

from catalog_table import catalog_table
import convert_conditions as convertcond
from convert_elevation import convert_elevation
from hms_to_hr import hms_to_hr
from select_obs import selectqueue


def fixcondstring(string):
    """
    Additional function from Bryan Miller's IDL GQPT used for adjusting certain conditions in ascii catalog file.
    """
    if string[0] == 'P':
        return string[8:10]
    elif string[0] == 'A':
        return 'Any'
    else:
        return string


def observation_table(filename):
    """
    Store gemini observation information in '~astropy.table.Table' object.
    Converts some variable types, performs merging of columns.

    Parameters
    ----------
    cattable : '~astropy.table.Table' of str types
        Table of ot catalog browser output data.

    Returns
    -------
    '~astropy.table.Table'

        Columns
        --------
        prog_ref            (string)                unique program identifier
        obs_id              (string)                unique observation identifier
        pi                  (string)                principle investigator
        inst                (string)                instrument name
        target              (string)                target name
        ra                  (degrees)               right ascension degrees
        dec                 (degrees)               declination degrees
        band                (int)                   integer
        partner             (string)                gemini partner name
        obs_status          (string)                'ready' status of observation
        tot_time            ('astropy.units' hours) total planned observation time
        obs_time            ('astropy.units' hours) completed observation time
        obs_comp            (float)                 fraction of completed/total observation time
        charged_time        (string)                HH:MM:SS (required to compute obs_comp)
        obs_class           (string)                observation class
        iq                  (float)                 image quality constraint (percentile converted to decimal value)
        cc                  (float)                 cloud condition constraint (percentile converted to decimal value)
        bg                  (float)                 sky background constraint (percentile converted to decimal value)
        wv                  (float)                 water vapor constraint (percentile converted to decimal value)
        user_prior          (string)                user priority (Low, Medium, High, Target of Opportunity)
        too_status          (string)                ToO type (Rapid, Standard, None)
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

    # Read OBD ascii catalog file columns into an 'astropy.table.Table' structure.
    # This module will also need to be replaced or changed if attempting to use a observation data format or a
    # different file format.
    cattable = catalog_table(filename)

    # Select observations from catalog to add to queue
    # i_obs = selectqueue(cattable=cattable)

    # for now, take all observations from catalog table.
    # A method for selecting only 'ready' observations could be included here.
    i_obs = np.arange(len(cattable))

    n_obs = len(i_obs)  # number of observations
    obstable = Table()  # initialize table

    # # Add column for time of ToO arrival for simulation purposes
    # if 'arrival' in cattable.colnames:
    #     obstable['arrival'] = cattable['arrival'][i_obs]

    obstable['prog_ref'] = cattable['prog_ref'][i_obs]
    obstable['obs_id'] = cattable['obs_id'][i_obs]
    obstable['pi'] = cattable['pi'][i_obs]
    obstable['inst'] = cattable['inst'][i_obs]
    obstable['target'] = cattable['target'][i_obs]

    # ------ Get current epoch coordinates ------
    epoch = Time.now()
    coord_j2000 = SkyCoord(cattable['ra'], cattable['dec'], frame='icrs', unit=(u.deg, u.deg))
    current_epoch = coord_j2000.transform_to(FK5(equinox='J' + str(epoch.jyear)))
    obstable['ra'] = Column(current_epoch.ra.value, unit='deg')
    obstable['dec'] = Column(current_epoch.dec.value, unit='deg')

    # ------ Format condition constraints -------
    iq = np.array(list(map(fixcondstring, cattable['iq'][i_obs])))
    cc = np.array(list(map(fixcondstring, cattable['cloud'][i_obs])))
    wv = np.array(list(map(fixcondstring, cattable['wv'][i_obs])))
    obstable['iq'], obstable['cc'], obstable['bg'], obstable['wv'] = \
        convertcond.convertcond(iq, cc, cattable['sky_bg'][i_obs], wv)

    obstable['band'] = list(map(int, cattable['band'][i_obs]))
    obstable['partner'] = cattable['partner'][i_obs]
    obstable['obs_status'] = cattable['obs_status'][i_obs]
    obstable['obs_class'] = cattable['obs_class'][i_obs]
    obstable['user_prior'] = cattable['user_prio'][i_obs]
    obstable['qc_prior'] = np.ones(len(i_obs))

    # -- ToO status --
    too_status = []
    for user_prior in obstable['user_prior']:
        if 'Rapid' in user_prior:
            tootype = 'Rapid'
        elif 'Standard' in user_prior:
            tootype = 'Standard'
        else:
            tootype = 'None'
        too_status.append(tootype)
    obstable['too_status'] = tootype

    obstable['group'] = cattable['group'][i_obs]
    obstable['elev_const'] = [convert_elevation(cattable['elev_const'][i]) for i in i_obs]
    obstable['time_const'] = cattable['time_const'][i_obs]
    obstable['ready'] = np.array(list(map(bool, cattable['ready'][i_obs])))
    obstable['f2_disperser'] = cattable['disperser'][i_obs]
    obstable['f2_filter'] = cattable['filter'][i_obs]
    obstable['grcwlen'] = cattable['grating_ctrl_wvl'][i_obs]
    obstable['xbin'] = cattable['x_bin'][i_obs]
    obstable['ybin'] = cattable['y_bin'][i_obs]
    obstable['disperser'] = cattable['disperser_2'][i_obs]
    obstable['filter'] = cattable['filter_2'][i_obs]
    obstable['fpu'] = cattable['fpu'][i_obs]

    f2_fpu = cattable['focal_plane_unit'][i_obs]
    crwlen = cattable['central_wavelength'][i_obs]
    bh_xbin = cattable['ccd_x_binning'][i_obs]
    bh_ybin = cattable['ccd_y_binning'][i_obs]
    mask = cattable['mask'][i_obs]

    # ------ Combine columns ------
    ii = np.where(f2_fpu != 'null')[0][:]
    if len(ii) != 0:
        obstable['fpu'][ii] = f2_fpu[ii]

    ii = np.where(crwlen != 'null')[0][:]
    if len(ii) != 0:
        obstable['grcwlen'][ii] = crwlen[ii]

    ii = np.where(bh_xbin != 'null')[0][:]
    if len(ii) != 0:
        obstable['xbin'][ii] = bh_xbin[ii]

    ii = np.where(bh_ybin != 'null')[0][:]
    if len(ii) != 0:
        obstable['ybin'][ii] = bh_ybin[ii]

    ii = np.where(mask != 'null')[0][:]
    if len(ii) != 0:
        obstable['fpu'][ii] = mask[ii]

    ii = np.where(obstable['group'] == '')[0][:]
    if len(ii) != 0:
        obstable['group'][ii] = obstable['obs_id'][ii]

    # --- Convert observation times ---
    obs_comp = []
    tot_time = []
    obs_time = []
    ii = np.where(cattable['charged_time'][i_obs] == '')[0][:]
    if len(ii) != 0:
        cattable['charged_time'][i_obs][ii] = '00:00:00'
    for i in range(0, n_obs):
        charged = hms_to_hr(cattable['charged_time'][i_obs][i])
        total = hms_to_hr(cattable['planned_exec_time'][i_obs][i])
        if charged > 0.:  # add additional time
            if 'Mirror' in obstable['disperser'][i]:
                total = total + 0.2
            else:
                total = total + 0.3
        obs_comp.append(charged / total)  # completion fraction
        tot_time.append(total)  # total time required
        obs_time.append(charged)  # observed time
    obstable['tot_time'] = Column(np.array(tot_time, dtype=float), unit='hr')
    obstable['obs_time'] = Column(np.array(obs_time, dtype=float), unit='hr')
    obstable['obs_comp'] = obs_comp

    # unused columns from catalog file...
    # obstable['qa_status'] = cattable['obs_qa'][i_obs]
    # obstable['dataflow_step'] = cattable['dataflow_step'][i_obs]
    # obstable['planned_pi_time'] = cattable.planned_pi_time[i_obs]
    # obstable['ao'] = cattable.ao[i_obs]
    # obstable['group_type'] = cattable.gt[i_obs]
    # obstable['color_filter'] = cattable.color_filter[i_obs]
    # obstable['nd_filter'] = cattable.neutral_density_filter[i_obs]
    # obstable['binning'] = cattable.binning[i_obs]
    # obstable['windowing'] = cattable.windowing[i_obs]
    # obstable['lens'] = cattable.lens[i_obs]
    # obstable['cass_rotator'] = cattable.cass_rotator[i_obs]
    # obstable['bh_ccdamps'] = cattable.ccd_amplifiers[i_obs]
    # obstable['bh_ccdgain'] = cattable.ccd_gain[i_obs]
    # obstable['bh_ccdspeed'] = cattable.ccd_speed[i_obs]
    # obstable['bh_fibre'] = cattable.entrance_fibre[i_obs]
    # obstable['bh_expmeter_filter'] = cattable.exposure_meter_filter[i_obs]
    # obstable['bh_hartmann'] = cattable.hartmann_flap[i_obs]
    # obstable['bh_issport'] = cattable.iss_port[i_obs]
    # obstable['bh_pslitfilter'] = cattable.post_slit_filter[i_obs]
    # obstable['bh_roi'] = cattable.region_of_interest[i_obs]
    # obstable['f2_readmode'] = cattable.read_mode[i_obs]
    # obstable['f2_lyot'] = cattable.lyot_wheel[i_obs]
    # obstable['roi'] = cattable.builtin_roi[i_obs]
    # obstable['nodshuffle'] = cattable.nod_shuffle[i_obs]
    # obstable['dtax'] = cattable.dta_x_offset[i_obs]
    # obstable['custom_mask'] = cattable.custom_mask_mdf[i_obs]
    # obstable['preimage'] = cattable.mos_pre_imaging[i_obs]
    # obstable['amp_count'] = cattable.amp_count[i_obs]
    # obstable['detector'] = cattable.detector_manufacturer[i_obs]
    # obstable['FIELD063'] = cattable.grating_ctrl_wvl_2[i_obs]
    # obstable['FIELD064'] = cattable.x_bin_2[i_obs]
    # obstable['FIELD065'] = cattable.y_bin_2[i_obs]
    # obstable['FIELD066'] = cattable.builtin_roi_2[i_obs]
    # obstable['FIELD067'] = cattable.nod_shuffle_2[i_obs]
    # obstable['FIELD068'] = cattable.dta_x_offset_2[i_obs]
    # obstable['FIELD069'] = cattable.custom_mask_mdf_2[i_obs]
    # obstable['FIELD070'] = cattable.mos_pre_imaging_2[i_obs]
    # obstable['FIELD071'] = cattable.amp_count_2[i_obs]
    # obstable['FIELD072'] = cattable.disperser_3[i_obs]
    # obstable['FIELD073'] = cattable.filter_3[i_obs]
    # obstable['FIELD074'] = cattable.fpu_2[i_obs]
    # obstable['FIELD075'] = cattable.detector_manufacturer_2[i_obs]
    # obstable['pixel_scale'] = cattable.pixel_scale[i_obs]
    # obstable['FIELD077'] = cattable.disperser_4[i_obs]
    # obstable['FIELD078'] = cattable.focal_plane_unit_2[i_obs]
    # obstable['cross_dispersed'] = cattable.cross_dispersed[i_obs]
    # obstable['FIELD080'] = cattable.read_mode_2[i_obs]
    # obstable['iss_port'] = cattable.iss_port_2[i_obs]
    # obstable['FIELD083'] = cattable.well_depth[i_obs]
    # obstable['FIELD084'] = cattable.filter_4[i_obs]
    # obstable['readmode'] = cattable.read_mode_3[i_obs]
    # obstable['astrometric'] = cattable.astrometric_field[i_obs]
    # obstable['FIELD087'] = cattable.disperser_5[i_obs]
    # obstable['adc'] = cattable.adc[i_obs]
    # obstable['observing_mode'] = cattable.observing_mode[i_obs]
    # obstable['coadds'] = cattable.coadds[i_obs]
    # obstable['exptime'] = cattable.exposure_time[i_obs]
    # obstable['FIELD092'] = cattable.disperser_6[i_obs]
    # obstable['eng_mask'] = cattable.engineering_mask[i_obs]
    # obstable['FIELD095'] = cattable.filter_[i_obs]
    # obstable['order'] = cattable.disperser_order[i_obs]
    # obstable['nici_fpu'] = cattable.focal_plane_mask[i_obs]
    # obstable['nici_pupil'] = cattable.pupil_mask[i_obs]
    # obstable['nici_cassrot'] = cattable.cass_rotator_2[i_obs]
    # obstable['nici_imgmode'] = cattable.imaging_mode[i_obs]
    # obstable['nici_dichroic'] = cattable.dichroic_wheel[i_obs]
    # obstable['nici_fw1'] = cattable.filter_red_channel[i_obs]
    # obstable['nici_fw2'] = cattable.filter_blue_channel[i_obs]
    # obstable['nici_welldepth'] = cattable.well_depth_2[i_obs]
    # obstable['nici_dhs'] = cattable.dhs_mode[i_obs]
    # obstable['imaging_mirror'] = cattable.imaging_mirror[i_obs]
    # obstable['FIELD107'] = cattable.disperser_7[i_obs]
    # obstable['FIELD108'] = cattable.mask_2[i_obs]
    # obstable['FIELD109'] = cattable.filter_6[i_obs]
    # obstable['FIELD110'] = cattable.read_mode_4[i_obs]
    # obstable['camera'] = cattable.camera[i_obs]
    # obstable['FIELD112'] = cattable.disperser_8[i_obs]
    # obstable['FIELD113'] = cattable.mask_3[i_obs]
    # obstable['FIELD114'] = cattable.filter_7[i_obs]
    # obstable['beam_splitter'] = cattable.beam_splitter[i_obs]
    # obstable['FIELD116'] = cattable.read_mode_5[i_obs]
    # obstable['FIELD117'] = cattable.mask_4[i_obs]
    # obstable['FIELD118'] = cattable.filter_8[i_obs]
    # obstable['FIELD119'] = cattable.disperser_9[i_obs]
    # obstable['FIELD120'] = cattable.disperser_10[i_obs]
    # obstable['FIELD121'] = cattable.mask_5[i_obs]
    # obstable['FIELD122'] = cattable.filter_9[i_obs]

    return obstable

def test_fixcondstring():
    string1 = fixcondstring('Partly cloudy')
    string2 = fixcondstring('Apparently it\'s cloudy but I have not been outside today')
    string3 = fixcondstring('Not cloudy at all')
    print(string1)
    print(string2)
    print(string3)
    assert string1 == 'lo'
    assert string2 == 'Any'
    assert string3 == 'Not cloudy at all'
    print('Test succesful!')
    return

if __name__=='__main__':
    test_fixcondstring()
