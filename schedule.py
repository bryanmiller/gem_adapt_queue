# Matt Bonnyman 19 July 2018

import numpy as np
import astropy.units as u
from astropy.time import Time

from dt import deltat
from intervals import intervals


def priority(plan, obs, targets, dt):
    """
    Add an observation to the current plan using the priority scheduling algorithm.
    Return the current plan if no observations can be added.

    2018-08-01 addition: ToOs will be scheduled as early in the night as possible.

    Input
    ---------
    plan : list or np.array of integers
        Current plan.

    obs : '~astropy.table.Table'
        Observation information table created by observation_table.py

    targets : '~astropy.table.Table'
        Target info table created by target_table.py and with additional columns of vsb (visible sky background
        magnitude), bg(sky background constraint), and weights.

    dt : '~astropy.units.quantity.Quantity'
        Size of time grid spacing in hours.

    Returns
    ---------
    plan : list or np.array of integers
        new plan
    """

    verbose = False
    verbose_add_to_plan = False  # print only the final results of the algorithm

    # -- Add an observation to the plan --
    while True:

        ii = np.where(plan == -1)[0][:]  # empty time slots in schedule
        if len(ii) != 0:

            nt = len(plan)
            n_obs = len(obs)
            indx = intervals(ii)  # intervals of empty time slots
            iint = ii[np.where(indx == 1)[0][:]]  # first interval of indx

            if verbose:
                print('ii:', ii)
                print('indx', indx)
                print('iint: ', iint)

            # -- Try to schedule an observation --
            gow = True
            while gow:

                # -- Select an observation --
                nminuse = 10  # min obs. block
                maxweight = 0.
                iimax = -1  # index of target with max. weight
                for i in np.arange(n_obs):
                    ipos = np.where(targets['weight'][i][iint] > 0.)[0][:]  # in. of weights>0 in first interval
                    if len(ipos) == 0:
                        continue  # skip to next observation if no non-zero weights in current window
                    else:
                        iwin = iint[ipos]  # indices with pos. weights within first empty window
                        if verbose:
                            print('iwin', iwin)
                        if len(iwin) >= 2:  # if window >=
                            if verbose:
                                print('i, weights:', i, targets['weight'][i][iint])
                            i_wmax = iwin[np.argmax(targets['weight'][i][iwin])]  # in. of max weight
                            wmax = targets['weight'][i][i_wmax]  # maximum weight
                        else:  # if window size of 1
                            wmax = targets['weight'][i][iwin]  # maximum weight

                        if wmax > maxweight:
                            maxweight = wmax
                            iimax = i
                            iwinmax = iwin

                        if verbose:
                            print('maxweight', maxweight)
                            print('max obs: ', targets['id'][iimax])
                            print('iimax', iimax)

                # -- Determine observation window and length --
                if iimax == -1:
                    gow = False
                else:
                    # Boundaries of available window
                    wstart = iwinmax[0]  # window start
                    wend = iwinmax[-1]  # window end
                    nobswin = wend - wstart + 1

                    # Calibration time
                    if verbose:
                        print('Inst: ', obs['inst'][iimax])
                        print('Disperser: ', obs['disperser'][iimax])
                    ntcal = _timecalibrate(inst=obs['inst'][iimax], disperser=obs['disperser'][iimax])
                    if verbose:
                        print('ntcal = ', ntcal)

                    # Remaining time (including calibration)
                    ttime = ((obs['tot_time'].quantity[iimax] -
                              obs['obs_time'].quantity[iimax] + 0.05 * u.h).round(1))
                    nttime = int(np.round(ttime / dt) + ntcal)  # number of spots in time grid

                    # Alter min. schedule block size for small observations
                    if nttime - nminuse <= nminuse:
                        nminuse = nttime

                    if verbose:
                        print('ID of chosen ob.', targets['id'][iimax])
                        print('weights of chosen ob.', targets['weight'][iimax])
                        print('Current plan', plan)
                        print('wstart', wstart)
                        print('wend', wend)
                        print('dt', dt)
                        print('tot_time', obs['tot_time'].quantity[iimax])
                        print('obs_time', obs['obs_time'].quantity[iimax])
                        print('ttime', ttime)
                        print('nttime', nttime)
                        print('nobswin', nobswin)
                        print('nminuse', nminuse)

                    # Decide whether or not to add to schedule
                    if np.logical_or(nttime <= nobswin, nobswin >= nminuse):  # Schedule observation
                        gow = False
                    else:  # Do not schedule observation
                        targets['weight'][iimax][iint] = 0.
                        if verbose:
                            print('Block too short to schedule...')

            # -- Place observation in schedule
            if iimax == -1:
                plan[iint] = -2
            else:

                # -- Place observation within available window --
                if np.logical_and(nttime <= nobswin, nttime != 0):

                    jj = np.where(plan == iimax)[0][:]  # check if already scheduled
                    if len(jj) == 0:

                        if 'Interrupt' in obs['user_prior'][iimax]:  # Schedule interrupt ToO at beginning of window
                            jstart = wstart
                            jend = wstart + nttime - 1
                        else:
                            # Determine schedule placement for maximum integrated weight
                            maxf = 0.0
                            if nttime > 1:
                                # NOTE: integrates over one extra time slot...
                                # ie. if nttime = 14, then the program will choose 15
                                # x values to do trapz integration (therefore integrating
                                # 14 time slots).
                                if verbose:
                                    print('\nIntegrating max obs. over window...')
                                    print('wstart', wstart)
                                    print('wend', wend)
                                    print('nttime', nttime)
                                    print('j values', np.arange(wstart, wend - nttime + 2))
                                for j in range(wstart, wend - nttime + 2):
                                    f = sum(targets['weight'][iimax][j:j + nttime])
                                    if verbose:
                                        print('j range', j, j + nttime - 1)
                                        print('obs wieght', targets['weight'][iimax][j:j + nttime])
                                        print('integral', f)
                                    if f > maxf:
                                        maxf = f
                                        jstart = j
                                jend = jstart + nttime - 1
                            else:
                                jstart = np.argmax(targets['weight'][iimax][iwinmax])
                                maxf = np.amax(targets['weight'][iimax][jstart])
                                jend = jstart + nttime - 1

                            if verbose:
                                print('max integral of weight func (maxf)', maxf)
                                print('index jstart', jstart)
                                print('index jend', jend)

                            # shift to start or end of night if within minimum block time from boundary
                            if jstart < nminuse:
                                if plan[0] == -1 and targets['weight'][iimax][0] > 0.:
                                    jstart = 0
                                    jend = jstart + nttime - 1
                            elif (nt - jend) < nminuse:
                                if plan[-1] == -1 and targets['weight'][iimax][-1] > 0.:
                                    jend = nt - 1
                                    jstart = jend - nttime + 1

                            # Shift to window boundary if within minimum block time of edge.
                            # If near both boundaries, choose boundary with higher weight.
                            wtstart = targets['weight'][iimax][wstart]  # weight at start
                            wtend = targets['weight'][iimax][wend]  # weight at end
                            dstart = jstart - wstart - 1  # difference between start of window and block
                            dend = wend - jend + 1    # difference between end of window and block
                            if dstart < nminuse and dend < nminuse:
                                if wtstart > wtend and wtstart > 0.:
                                    jstart = wstart
                                    jend = wstart + nttime - 1
                                elif wtend > 0.:
                                    jstart = wend - nttime + 1
                                    jend = wend
                            elif dstart < nminuse and wtstart > 0.:
                                jstart = wstart
                                jend = wstart + nttime - 1
                            elif dend < nminuse and wtstart > 0.:
                                jstart = wend - nttime + 1
                                jend = wend

                    # If observation is already in plan, shift to side of window closest to existing obs.
                    else:
                        if jj[0] < wstart:  # Existing obs in plan before window. Schedule at beginning of window.
                            jstart = wstart
                            jend = wstart + nttime - 1
                        else:  # Existing obs in plan after window. Schedule at end of window.
                            jstart = wend - nttime + 1
                            jend = wend

                else:  # if window smaller than observation length
                    jstart = wstart
                    jend = wend

                if verbose:
                    print('Chosen index jstart', jstart)
                    print('Chosen index jend', jend)
                    print('Current obs time: ', obs['obs_time'].quantity[iimax])
                    print('Current tot time: ', obs['tot_time'].quantity[iimax])

                plan[jstart:jend + 1] = iimax  # Add observation to plan
                ntmin = np.minimum(nttime - ntcal, nobswin)  # number of spots in time grid used(excluding calibration)

                obs['obs_time'].quantity[iimax] = obs['obs_time'].quantity[iimax] + dt * ntmin  # update time
                obs['obs_comp'][iimax] = obs['obs_comp'][iimax] + dt * ntmin / obs['tot_time'].quantity[iimax]  # update completion fraction

                # Adjust weights of scheduled observation
                if obs['obs_comp'][iimax] >= 1.:  # if completed set all to zero.
                    targets['weight'][iimax] = targets['weight'][iimax] * 0
                else:  # if observation not fully completed, set only scheduled portion to zero. Increase remaining.
                    targets['weight'][iimax][jstart:jend + 1] = targets['weight'][iimax][jstart:jend + 1] * 0
                    wpositive = np.where(targets['weight'][iimax] >= 0)[0][:]
                    targets['weight'][iimax][wpositive] = targets['weight'][iimax][wpositive] * 1.5

                # Add to total time if observation not fully completed
                if obs['obs_time'].quantity[iimax] < obs['tot_time'].quantity[iimax]:
                    obs['tot_time'].quantity[iimax] = obs['tot_time'].quantity[iimax] + \
                                                      _acqoverhead(obs['disperser'][iimax])

                # increase weights of observations in program
                # ii_obs = np.where(obs.obs_id == obs.prog_ref[iimax])[0][:]  # indices of obs. in same program
                if verbose:
                    print('Current plan: ', plan)
                    print('New obs. weights: ', targets['weight'][iimax])
                    print('nttime - ntcal , nobswin: ', nttime - ntcal, nobswin)
                    print('ntmin: ', ntmin)
                    print('Tot time: ', obs['tot_time'].quantity[iimax])
                    print('New obs time: ', obs['obs_time'].quantity[iimax])
                    print('New comp time: ', obs['obs_comp'][iimax])

                if verbose_add_to_plan:
                    print('\tScheduled: ', iimax, targets['name'][iimax], 'from jstart =', jstart, 'to jend =', jend)
                    print(targets[iimax].weight)

                break  # successfully added an observation to the plan
        else:
            break  # No available spots in plan

    return plan


def optimize(plan, targets, jj=None):
    """
    Attempt to rearrange plan and maximize the sum of the target weighting functions.

    Parameters
    ----------
    plan : np.ndarray of ints
        array of observation indices in plan

    targets : '~astropy.table.Table'
        Target information table created by target_table.py

    jj : np.ndarray of ints
        indices of section of 'plan' to be optimized.  Other parts of the plan
        will remain unchanged.

    Returns
    -------
    plan : np.ndarray of ints
        array of observation indices in plan
    """

    verbose = False

    # select whole plan to optimize
    if jj is None:
        jj = np.arange(len(plan))

    newplan = np.full(len(plan), -2)  # empty plan

    i_obs = np.unique(plan[jj])  # obs in plan[jj]

    nid = np.zeros(len(i_obs), dtype=int)  # number of time slots per obs
    plan_weight = 0.

    if verbose:
        print('Full plan: ', plan)
        print('Plan section to optimize (plan[jj]): ', plan[jj])
        print('jj: ', jj)
        print('i_obs: ', i_obs)

    # -- Compute total weight of plan[jj] --
    for i in range(0, len(i_obs)):
        ii = jj[np.where(plan[jj] == i_obs[i])[0][:]]
        nid[i] = int(len(ii))
        if i_obs[i] >= 0:
            plan_weight = plan_weight + sum(abs(targets['weight'][i_obs[i]][ii]))

    # Attempt to re-arrange observations and achieve higher total weight
    nt = len(jj)
    i = jj[0]
    while i < nt:
        if plan[i] >= 0:
            if verbose:
                print('i, i_obs, nid: ', i, i_obs, nid)
            imax = -1
            wmax = 0.
            idmax = 0.
            for j in range(0, len(i_obs)):
                if i_obs[j] >= 0:
                    temp_wmax = abs(targets['weight'][i_obs[j]][i])
                    if temp_wmax > wmax:
                        wmax = temp_wmax
                        imax = j
                        idmax = i_obs[j]
            if verbose:
                print('wmax, imax, idmax: ', wmax, imax, idmax)
            if wmax > 0.:
                if i+nid[imax] <= nt:
                    newplan[i:(i+nid[imax])] = idmax
                    i = i + nid[imax]
                else:
                    newplan[i:nt] = idmax
                    i = nt
                if verbose:
                    print('nid[imax]: ', nid[imax])
                    print('delete j from i_obs: ', imax, i_obs)
                    print('newplan: ', newplan)
                i_obs = np.delete(i_obs, imax, None)
                nid = np.delete(nid, imax, None)
            else:
                i = i+1
        else:
            i = i+1

    # return original plan if no changes were made
    if newplan[jj].all() == plan[jj].all():
        return plan

    # Get total weight of new plan
    newplan_weight = 0.
    new_nid = np.zeros(len(i_obs))
    for i in range(0, len(i_obs)):
        ii = jj[np.where(newplan[jj] == i_obs[i])[0][:]]
        new_nid[i] = int(len(ii))
        if i_obs[i] >= 0:
            newplan_weight = newplan_weight + sum(abs(targets['weight'][i_obs[i]][ii]))

    if newplan_weight >= plan_weight:
        plan[jj] = newplan[jj]

    if verbose:
        print('Original plan[jj]: ', plan[jj])
        print('Weight: ', plan_weight)
        print('New plan[jj]: ', newplan)
        print('Weight: ', newplan_weight)

    return plan


def update_obs_progs(plan, obs, progs, dt):
    """
    Update times and data in observation and program tables for all or part of a plan.

    Parameters
    ----------
    plan : np.ndarray of ints
        all or part of a nightly plan (array of observation row indices)

    obs : '~astropy.table.Table'
        Observation information table from observation_table.py

    progs : '~astropy.table.Table'
        Gemini program information table from program_table.py

    dt : '~astropy.units' hours
        size of time grid spacing

    Returns
    -------
    obs : '~astropy.table.Table'
        Updated observation information table.

    progs : '~astropy.table.Table'
        Updated program information table.
    """
    verbose = False

    index = np.unique(plan)  # indices of obs in plan

    if verbose:
        print('\nUPDATE_OBS_PROGS()...')
        print('plan input', plan)
        print('obs indices in plan', index)

    for ind in index:
        if ind >= 0:
            ntcal = _timecalibrate(inst=obs['inst'][ind], disperser=obs['disperser'][ind])

            ii = np.where(plan == ind)[0][:]  # indices of obs in plan
            intr = intervals(ii)  # continuous intervals of obs in plan
            iintr = np.unique(intr)  # count independent exposures

            if verbose:
                print('index', ind)
                print('obs_id', obs['obs_id'][ind])
                print('ii', ii)
                print('intr', intr)
                print('iintr', iintr)

            for j in iintr:  # cycle through exposures for the current obs
                jj = np.where(intr == j)[0][:]  # indices of scheduled block
                nttime = len(jj)  # number of time slots in plan

                if verbose:
                    print('j', j)
                    print('nttime, ntcal: ', nttime, ntcal)
                    print('old obs_time', obs['obs_time'][ind])
                    print('old obs_comp', obs['obs_comp'][ind])
                    print('old tot_time', obs['tot_time'][ind])
                    print('old prog_time', progs['prog_time'][obs['i_prog'][ind]])
                    print('old prog_comp', progs['prog_comp'][obs['i_prog'][ind]])
                    print('old alloc_time', progs['alloc_time'][obs['i_prog'][ind]])

                # add observed time to observation
                obs['obs_time'].quantity[ind] = obs['obs_time'].quantity[ind] + (nttime - ntcal) * dt

                # add observed time to program
                progs['prog_time'].quantity[obs['i_prog'][ind]] = progs['prog_time'].quantity[obs['i_prog'][ind]] + (nttime - ntcal) * dt

                # add acquisition overhead time to observation and program total if observation not fully completed
                if obs['obs_time'].quantity[ind] < obs['tot_time'].quantity[ind]:
                    acqtime = _acqoverhead(obs['disperser'][ind])  # acquisition overhead time
                    progs['alloc_time'].quantity[obs['i_prog'][ind]] = \
                        progs['alloc_time'].quantity[obs['i_prog'][ind]] + acqtime
                    obs['tot_time'].quantity[ind] = obs['tot_time'].quantity[ind] + acqtime
                    obs['obs_comp'][ind] = obs['obs_time'].quantity[ind] / obs['tot_time'].quantity[ind]
                else:
                    obs['obs_comp'][ind] = 1.

                # update program completion fraction
                progs['prog_comp'].quantity[obs['i_prog'].data[ind]] = \
                    progs['prog_time'].quantity[obs['i_prog'].data[ind]]/\
                    progs['alloc_time'].quantity[obs['i_prog'].data[ind]]

                if verbose:
                    print('new obs_time', obs['obs_time'][ind])
                    print('new obs_comp', obs['obs_comp'][ind])
                    print('new tot_time', obs['tot_time'][ind])
                    print('new prog_time', progs['prog_time'][obs['i_prog'][ind]])
                    print('new prog_comp', progs['prog_comp'][obs['i_prog'][ind]])
                    print('new alloc_time', progs['alloc_time'][obs['i_prog'][ind]])


    return obs, progs


def updateweights(plan, targets, obs_comp):
    """
    Update weights of targets for all or part of the plan.
    If observation is completed (i.e. obs_comp >= 1.0), multiply weights by 0.
    If observation in plan but not complete, multiply weights by 1.5.

    Parameters
    ----------
    plan : np.ndarray of ints
        all or part of a nightly plan (array of observation indices)

    targets : '~astropy.table.Table'
        Target information table created by target_table.py

    obs_comp : list or np.array of floats
        Subset of observation completion fraction column corresponding to observations in 'targets'
         in observation information table.

    Returns
    -------
    targets : '~astropy.table.Table'
        Target information table with updated values
    """
    verbose = False

    index = np.unique(plan)  # indices of targets in plan

    if verbose:
        print('plan input', plan)
        print('target indices in plan', index)

    for ind in index:
        if ind >= 0:

            if verbose:
                print('ind', ind)
                print('old weights', targets['weight'][ind])

            if obs_comp[ind] >= 1:
                targets['weight'][ind] = targets['weight'][ind] * 0
            else:
                targets['weight'][ind] = targets['weight'][ind] * 1.5

            if verbose:
                print('new weights', targets['weight'][ind])
    return targets


def nightstats(stats, plan, timetable):
    """
    Update the total available and used observating time.
    """

    dt = deltat(time_strings=timetable['utc'][0][0:2])

    night_length = (Time(timetable['utc'][0][-1]) - Time(timetable['local'][0][0]) + dt).to(u.hr).round(2)
    used_time = (len(np.where(plan >= 0)[0][:]) * dt).round(2)

    stats['tot_time'] = stats['tot_time'] + night_length
    stats['used_time'] = stats['used_time'] + used_time

    return stats


def _timecalibrate(inst, disperser):
    """
    Get number of time slots needed for calibration

    Parameters
    ----------
    inst : string
        instrument

    disperser : string

    Returns
    -------
    int
    """
    ntcal = 0
    if 'GMOS' not in inst:
        if 'Mirror' not in disperser and 'null' not in disperser:
            ntcal = 3
    return ntcal


def _acqoverhead(disperser):
    """
    Get acquisition overhead time

    Parameters
    ----------
    disperser : str

    Returns
    -------
    '~astropy.unit.quantity.Quantity'
    """
    if 'Mirror' in disperser:
        return 0.2*u.h
    else:
        return 0.3*u.h
