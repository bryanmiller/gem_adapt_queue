import numpy as np

def selectQueue(catinfo):

    """
    Select observations from catalog to add to Queue.

    Parameters
    ----------
    catinfo : '~gemini_otcat.Gcatfile'
        OT catalog info

    Returns
    -------
    numpy integer array
        indices of observations selected for queue
    """
    return np.where(np.logical_and(np.logical_or(catinfo.obs_status == 'Ready', catinfo.obs_status == 'Ongoing'),
                                   np.logical_or(catinfo.obs_class == 'Science',
                                   np.logical_and(np.logical_or(catinfo.inst == 'GMOS', catinfo.inst == 'bHROS'),
                                   np.logical_or(catinfo.obs_class == 'Nighttime Partner Calibration',
                                                 catinfo.obs_class == 'Nighttime Program Calibration')))))[0][:]

def selectObs(obs, inst_calender):

    """
    Select observations to add to tonight's queue.

    Parameters
    ----------
    obs : '~gemini_otcat.Gobservations'
        All observations in queue

    inst_calender :
        Instrument and component calender

    Returns
    -------
    numpy integer array
        indices of observations selected for tonight's queue
    """

    return i_obs