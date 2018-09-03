# Matt Bonnyman 23 July 2018

import copy

def convindex(plan, i_obs):
    """
    Convert indices in plan (which should correspond to nightly target table rows)
    to row indices of corresponding observations in the observation table.

    -2 --> Empty time slot in plan
    >= 0 --> Target table row indices

    Example
    -------
    >>> plan = [-2, -2, 1, 1, 1, 5, 5, 5, 5, 7, 7, -2, -2]
    >>> i_obs = [5, 33, 45, 66, 74, 78, 89, 114, 130]
    >>> print(convindex(plan, i_obs))
    [-2, -2, 33, 33, 33, 78, 78, 78, 78, 114, 114, -2, -2]

    Parameters
    ----------
    plan : int array
        Observing plan for current night.

    i_obs : int array
        corresponding observation indices.

    Returns
    -------
    new_plan : int array

    """
    new_plan = copy.deepcopy(plan)
    for i in range(len(new_plan)):
        if new_plan[i] >= 0:
            new_plan[i] = i_obs[new_plan[i]]
    return new_plan


def test_convindex():
    plan = [3, 3, 3, 5, 5, 5, 1, 1, 1, 7, 7, 7]
    obs_inds = [0, 2, 4, 6, 8, 10, 12, 14, 16]
    result = convindex(plan, obs_inds)
    print('Test convindex()...')
    print('plan:', plan)
    print('obs_inds:', obs_inds)
    print('convindex(plan, obs_inds):', convindex(plan, obs_inds))
    assert result == [6, 6, 6, 10, 10, 10, 2, 2, 2, 14, 14, 14]
    print('Test successful!')
    return

if __name__=='__main__':
    test_convindex()
