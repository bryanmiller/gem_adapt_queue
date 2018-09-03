# Matt Bonnyman 18 July 2018

import numpy as np
from astropy.table import Table, Column


def wind_table(size, direction, velocity, site_name, random=False):
    """
    Generate '~astropy.table.Table' of wind conditions.

    Parameters
    ----------

    site_name : '~astroplan.Observer'
        observatory site name 'gemini_south', 'CP', 'gemini_north', or 'MK'

    size : int
        length of time grid.

    direction : float
        Wind direction in degrees

    velocity : float
        Wind velocity in km/h

    random : boolean
        If True, the wind will be selected using the mean and standard deviation
        of the wind for the provided observatory location in site_name.

    Returns
    -------
    '~astropy.table.Table' with direction and velocity of wind.
    """

    verbose = False

    if random:
        # set means and standard deviations of normal distributions
        if site_name.lower() == 'gemini_south' or site_name.lower() == 'cp':
            meanvel = 5
            stdvel = 3
            meandir = 330
            stddir = 20
        elif site_name.lower() == 'gemini_north' or site_name.lower() == 'mk':
            meanvel = 5
            stdvel = 3
            meandir = 330
            stddir = 20
        else:
            meanvel = 5
            stdvel = 3
            meandir = 330
            stddir = 20

        vel = round(np.random.normal(meanvel, stdvel), 1)
        dir = round(np.random.normal(meandir, stddir), 1)
        if verbose:
            print('mean,sigma velocity', meanvel, stdvel)
            print('mean,sigma direction', meandir, stddir)
    else:
        vel = velocity
        dir = direction

    if verbose:
        print(vel)
        print(dir)

    if vel < 0:
        vel = 0.
    if dir < 0:
        dir = dir + 360
    if dir > 360:
        dir = dir - 360

    vels = np.full(size, vel)
    dirs = np.full(size, dir)

    return Table((Column(vels, name='vel', unit='m/s'), Column(dirs, name='dir', unit='deg')))


def test_windtable():
    from astroplan import Observer
    gs = Observer.at_site('gemini_south')
    gn = Observer.at_site('gemini_north')

    size = 3
    dir = 270
    vel = 5

    # -- Reset random number seed --
    np.random.seed(1000)
    gswind = wind_table(size, direction=dir, velocity=vel, site_name=gs.name, random=True)

    # -- Reset random number seed --
    np.random.seed(1000)
    gnwind = wind_table(size, direction=dir, velocity=vel, site_name=gn.name, random=True)

    print('\nGemini South wind test conditions (random)\n', gswind)
    print('\nGemini North wind test conditions (random)\n', gnwind)

    assert gswind['vel'][0] == 2.6
    assert gswind['dir'][0] == 336.4
    assert gnwind['vel'][0] == 2.6
    assert gnwind['dir'][0] == 336.4

    # -- Reset random number seed --
    np.random.seed(1000)
    gswind = wind_table(size, direction=dir, velocity=vel, site_name=gs.name, random=False)

    # -- Reset random number seed --
    np.random.seed(1000)
    gnwind = wind_table(size, direction=dir, velocity=vel, site_name=gn.name, random=False)

    print('\nGemini South wind test conditions (dir=270deg, vel=5m/s)\n', gswind)
    print('\nGemini North wind test conditions (dir=270deg, vel=5m/s)\n', gnwind)

    assert gswind['vel'][0] == 5
    assert gswind['dir'][0] == 270
    assert gnwind['vel'][0] == 5
    assert gnwind['dir'][0] == 270

    print(' Test successful!')
    return


if __name__ == '__main__':
    test_windtable()
