# Matt Bonnyman 18 July 2018

import numpy as np
from astropy.table import Table

from convert_conditions import convertcond


def checkdist(conddist):
    """
    Check user input type if distribution type selected for generating conditions.

    Parameters
    ----------
    conddist : str or NoneType
        condition distribution name

    Returns
    -------
    str or None
    """
    if  conddist is None:
        return None
    elif conddist.lower() == 'none' or conddist == '':
        return None
    elif conddist.lower() == 'r' or conddist.lower() == 'random':
        return 'random'
    elif conddist.lower() == 'v' or conddist.lower() == 'variant':
        return 'variant'
    else:
        print(' ValueError: Condition distribution type \'{}\' not recognized.'.format(conddist))
        print(' Conditions distributions must be "r", "random", "v", or "variant".')
        raise ValueError


def condition_table(size, iq, cc, wv, conddist=None):
    """
    Generates an '~astropy.table.Table' of sky conditions.

    Columns: iq (image quality), cc (cloud condition), wv (water vapour).

    If conddist is NoneType, then the conditions will use iq, cc, wv.
    If conddist is 'gaussian', conditions will be randomly generated for each night.
    If conddist is 'variant', a random variant will be selected for each night.

    Examples
    --------



    Parameters
    ----------
    size : int
        length of time grid for observing window (eg. a night or portion of a night)

    iq : array of floats
        image quality (if conddist is None)

    cc : array of floats
        cloud condition (if conddist is None)

    wv : array of floats
        water vapor (if conddist is None)

    conddist : str or None
        condition distribution type to generate conditions from (or None to use iq, cc, and wv values)

    Returns
    -------
    '~astropy.table.Table' of iq, cc, and wv condition values.
    """

    if conddist is None:  # Use conditions input
        pass

    elif conddist.lower() == 'variant' or conddist.lower() == 'v':  # Randomly select a variant
        variantnum = np.random.randint(4) + 1
        if variantnum == 1:
            iq = '20%'
            cc = '50%'
            wv = '50%'
        elif variantnum == 2:
            iq = '70%'
            cc = '70%'
            wv = '80%'
        elif variantnum == 3:
            iq = '85%'
            cc = '70%'
            wv = 'Any'
        else:
            iq = 'Any'
            cc = '70%'
            wv = 'Any'

    elif conddist.lower() == 'random' or conddist.lower() == 'r': # Generate random conditions
        iq = str(np.random.uniform(0, 100))
        cc = str(np.random.uniform(0, 100))
        wv = str(np.random.uniform(0, 100))

    else:
        raise ValueError(' Condition type \'{}\' not recognized.'.format(conddist))

    # Convert conditions to decimals and construct table columns.
    bg = 'Any'
    iq, cc, bg, wv = convertcond(iq, cc, bg, wv)  # convert string or float to decimal
    iq = np.full(size, iq)
    cc = np.full(size, cc)
    wv = np.full(size, wv)
    return Table([iq, cc, wv], names=('iq', 'cc', 'wv'))


def test_condtable():
    size = 3

    conds = condition_table(size, '20%', '50%', '85%', conddist=None)
    assert conds['iq'][0] == 0.2
    assert conds['cc'][0] == 0.5
    assert conds['wv'][0] == 0.85
    print('\nInput iq=20%, cc=50%, wv=85%\n', conds)

    # -- Reset random number seed --
    np.random.seed(1000)  # Should produce variant 4 each time.
    conds = condition_table(size, '20%', '50%', '85%', conddist='variant')
    assert conds['iq'][0] == 1.
    assert conds['cc'][0] == 0.7
    assert conds['wv'][0] == 1.
    print('\nInput \'variant\'\n', conds)

    # -- Reset random number seed --
    np.random.seed(1000)
    conds = condition_table(size, '20%', '50%', '85%', conddist='random')
    assert conds['iq'][0] == 0.7
    assert conds['cc'][0] == 0.2
    assert conds['wv'][0] == 1.
    print('\nInput \'gaussian\'\n', conds)

    print('Test successful!')
    return

if __name__ == '__main__':
    test_condtable()
    exit()
