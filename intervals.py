# Matt Bonnyman 18 July 2018

import numpy as np


def intervals(indices):
    """
    Find the number and properties of contiguous intervals for an array of indices.

    Parameters
    ----------
    indices : np.ndarray of ints
        array of indices

    Returns
    -------
    indx : np.ndarray of ints

    """

    ni = len(indices)
    cvec = np.zeros(ni, dtype=int)
    nint = 1
    cvec[0] = nint
    for j in range(1, ni):
        if (indices[j] != (indices[j - 1] + 1)):
            nint = nint + 1
        cvec[j] = nint

    indx = np.digitize(cvec, bins=np.arange(ni) + 1)

    return indx


def test_intervals():
    print('\nTest intervals.py...')
    print('Get intervals of continuous -1s...')

    indices = np.array([0, 0, 0, 0, -1, -1, -1, -1, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, -1, -1, -1, -1], dtype=int)
    ints = np.zeros(len(indices), dtype=int)
    i_negones = np.where(indices == -1)[0][:]
    ints[i_negones] = intervals(i_negones)

    print('input:', indices)
    print('output: ', ints)

    assert ((ints == np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2])).all())
    print('Test successful!')


if __name__ == '__main__':
    test_intervals()
