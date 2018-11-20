from astropy.table import Table, Column
from astropy.time import Time
import astropy.units as u
from joblib import Parallel, delayed
import numpy as np

from gcirc import gcirc
from airmass import airmass
from calc_zd_ha_az import calc_zd_ha_az


def target_table(i_obs, latitude, lst, utc, obs_id, obs_ra, obs_dec, moon_ra, moon_dec):
    """
    Compute Target data for a single scheduling window (eg. a night).
    Return '~astropy.table' object.

    Parameters
    ----------
    i_obs : int array
        List or array of observation indices

    latitude : '~astropy.units.quantity.Quantity'
        Observatory site latitude.

    lst : '~astropy.units.quantity.Quantity' np.array
        Array of local sidereal times along time grid of scheduling window.

    utc : '~astropy.time.core.Time' array
        Array of UTC time grid spaces for scheduling period (i.e. night) in format accepted by 'astropy.time.Time'

    moon_ra : '~astropy.units.quantity.Quantity'
        Array of Moon right ascesion values along time grid of scheduling window.

    moon_dec : '~astropy.units.quantity.Quantity'
        Array of Moon declination values along time grid of scheduling window.

    obs_id : str
        Array of observation identifiers.

    obs_ra : '~astropy.units.quantity.Quantity'
        2d array of right ascension values along time grid of scheduling window for several observations.

    obs_dec : '~astropy.units.quantity.Quantity'
        2d array of declination values along time grid of scheduling window for several observations.

    Returns
    -------
    targets : '~astropy.table.Table'
        Target data throughout night with columns:
            'id'        str                                                    identifiers
            'ZD'        np.array of '~astropy.units.quantity.Quantity'         zenith distance angle
            'HA'        np.array of '~astropy.units.quantity.Quantity'         hour angle
            'AZ'        np.array of '~astropy.units.quantity.Quantity'         azimuth angle
            'AM'        np.array of float                                      airmass
            'mdist'     np.array of '~astropy.units.quantity.Quantity'         moon angular separation

    """

    verbose = False

    if verbose:
        print('i_obs', i_obs)
        print('latitude', latitude)
        print('lst', lst)
        print('obs_id', obs_id)
        print('obs_ra', obs_ra)
        print('obs_dec', obs_dec)
        print('moon_ra', moon_ra)
        print('moon_dec', moon_dec)

    if len(i_obs) == 0:
        return Table()

    n_obs = len(i_obs)

    targettable = Table()

    if n_obs == 0:
        return targettable

    else:
        targettable['i'] = Column([i_obs[i] for i in range(n_obs)])

        targettable['id'] = Column([obs_id[i_obs[i]] for i in range(n_obs)])

        ZDHAAZ = [calc_zd_ha_az(lst=lst,
                                latitude=latitude,
                                ra=obs_ra[i_obs[i]],
                                dec=obs_dec[i_obs[i]])
                  for i in range(n_obs)]

        targettable['ZD'] = Column([ZDHAAZ[i][0] for i in range(n_obs)], unit='deg')

        targettable['HA'] = Column([ZDHAAZ[i][1] for i in range(n_obs)], unit='hourangle')

        targettable['AZ'] = Column([ZDHAAZ[i][2] for i in range(n_obs)], unit='deg')

        targettable['AM'] = Column([airmass(ZDHAAZ[i][0]) for i in range(n_obs)])

        targettable['mdist'] = Column([gcirc(moon_ra, moon_dec, obs_ra[i_obs[i]], obs_dec[i_obs[i]], degree=True)
                                       for i in range(n_obs)], unit='deg')

        # dt = deltat(time_strings=utc[0:2])  # time grid spacing
        # targettable['i_wins'] = Column([time_window_indices(utc=utc, time_wins=time_windows[i], dt=dt)
        #                                 for i in range(n_obs)])

    if verbose:
        print(targettable)

    return targettable
