import scipy
import numpy as np
import astropy.units as u
from intervals import intervals

def _optimize_plan(plan, istart, targetinfo):

    verbose = False

    newplan = np.full(len(plan), -2)

    id = np.unique(plan)
    if len(id)==0: return

    nid = np.zeros(len(id),dtype=int)
    plan_weight = 0.

    for i in range(0,len(id)): # get number of slots per obs. and sum plan weight
        ii = np.where(plan == id[i])[0][:]
        nid[i] = int(len(ii))
        if id[i] >= 0:
            if verbose: print('id, ii, weight: ',id[i],ii,targetinfo[id[i]].weight[ii])
            plan_weight = plan_weight + sum(abs(targetinfo[id[i]].weight[ii]))

    nt = len(plan)
    i = 0
    idsel = np.array(id)
    while i < nt:
        if plan[i] >= 0:
            if verbose: print('i, idsel, nid: ', i, idsel, nid)
            imax = -1
            wmax = 0.
            temp_wmax = 0.
            idmax = 0.
            for j in range(0, len(idsel)):
                if idsel[j] >= 0:
                    temp_wmax = abs(targetinfo[idsel[j]].weight[i])
                    if temp_wmax > wmax:
                        wmax = temp_wmax
                        imax = j
                        idmax = idsel[j]
            if verbose: print('wmax, imax, idmax: ', wmax, imax, idmax)
            if wmax > 0.:
                if (i+nid[imax] <= nt):
                    newplan[i:(i+nid[imax])] = idmax
                    i = i + nid[imax]
                else:
                    newplan[i:nt] = idmax
                    i = nt
                if verbose: print('nid[imax]: ',nid[imax])
                if verbose: print('delete j from id: ', imax, idsel)
                if verbose: print('newplan: ', newplan)
                idsel = np.delete(idsel, imax, None)
                nid = np.delete(nid, imax, None)
            else:
                i = i+1
        else:
            i = i+1

    if (newplan == plan).all():
        return plan

    newplan_weight = 0.
    new_nid = np.zeros(len(id))
    for i in range(0, len(id)):  # get number of slots per obs. and sum plan weight
        ii = np.where(newplan == id[i])[0][:]
        new_nid[i] = int(len(ii))
        if id[i] >= 0:
            newplan_weight = newplan_weight + sum(abs(targetinfo[id[i]].weight[ii]))

    if verbose:
        print('Original plan: ',plan)
        print('Weight: ', plan_weight)
        print('New plan: ', newplan)
        print('Weight: ', newplan_weight)

    if newplan_weight >= plan_weight:
        return newplan
    else:
        return plan


def _plan_details(plan, targetinfo):

    # get stats while building final plan
    obslist = []  # observation names
    i_start = []  # observing index start
    i_end = []  # observing index end

    i = 0
    nt = len(plan)
    while i < nt:
        if plan[i] > 0:
            n = 1
            next = True
            iobs = plan[i]  # current obs index in plan
            obslist.append(targetinfo[iobs].name)
            i_start.append(i)
            while next:
                if plan[i+1] != plan[i] and i <= nt:
                    i_end.append(i)
                    next = False
                i = i + 1
        else:
            i = i+1
    return obslist, np.array(i_start, dtype=int), np.array(i_end, dtype=int)



class Gschedule(object):

    @u.quantity_input(tot_time=u.h,night_length=u.h)
    def __init__(self, plantype=None, plan=None, obslist=None, i_start=None, i_end=None, hours=None, cplt=None,
                 tot_time=0.*u.h, night_length=0.*u.h):
        """
        Parameters
        ----------
        type : string
            algorithm type used to generate plan

        plan : int
            array of observation indices (corresponding to observation
            order in Gobservation object) at times in TimeInfo object

        obslist : string
            unique gemini observation identifiers of scheduled observations
            in the order they were added to the plan

        i_start : int
            plan indices of observation starts corresponding to identifiers in oblist

        i_end : int
            plan indices of observation ends corresponding to identifiers in oblist

        hours : astropy.units.Quantity
            observation lengths corresponding to identifiers in oblist

        cplt : boolean
            completion status at end of observation corresponding to identifiers in oblist

        tot_time : astropy.units.Quantity
            total number of hours scheduled

        night_length : astropy.units.Quantity
            total number of in night (or similar observing window)
        """

        self.type = plantype  # define plan type
        self.plan = plan
        self.obs_id = obslist
        self.i_start = i_start
        self.i_end = i_end
        self.hours = hours
        self.cplt = cplt
        self.tot_time = tot_time
        self.night_length = night_length #thtimeinfo.night_length

    def __repr__(self):

        """
        String representation of the gemini_schedulers.Gschedule object.
        """

        class_name = self.__class__.__name__
        attr_names = ['type','night_length','tot_time','obs_id','i_start','i_end','hours','cplt','plan']
        attr_values = [getattr(self, attr) for attr in attr_names]
        attributes_strings = []
        for name, value in zip(attr_names, attr_values):
            if value is not None:
                value = "'{}'".format(value)
                attributes_strings.append("{}={}".format(name, value))
        return "<{}: {}>".format(class_name, ",\n    ".join(attributes_strings))

    @classmethod
    def priority(cls,obs,timeinfo,targetinfo):
        """
        Generate a plan with the priority scheduling algorithm

        Input
        ---------
        obs : '~gemini_classes.Gobservations'
            Gobservations object

        timeinfo : '~gemini_classes.TimeInfo'
            TimeInfo object

        targetinfo : '~gemini_classes.TargetInfo'
            TargetInfo object

        Returns
        ---------
        '~gemini_schedule.Gschedule'
            Gschedule object

        obs : '~gemini_classes.Gobservations'
            inputted Gobservation object with times
            of scheduled observations updated

        """
        verbose = False
        schedule_order = False

        plantype = 'Priority'  # define plan type
        plan = np.full(timeinfo.nt, -1)


        dt = timeinfo.dt
        nt = timeinfo.nt
        n_obs = len(obs.obs_id)
        ntcal = 0
        nsel = 0
        ii = np.where(plan == -1)[0][:]  # unscheduled time indices

        while len(ii)!=0:
            if verbose: print('\nIteration: ',nsel+1)

            indx = intervals(ii)  # group adjacent time slots
            iint = ii[np.where(indx == 1)[0][:]]  # first interval of unscheduled times

            if verbose:
                print('ii:',ii)
                print('iint: ',iint)
                # print('indx',indx)

            gow = True
            while gow: # schedule observation with maximum weight in time interval

                nminuse = 10  # min obs. block
                maxweight = 0.
                iimax = -1  # index of target with max. weight
                for i in np.arange(n_obs):

                    # ===== Observation windows ====
                    # if observation weights are zero during current interval, skip remainder of loop.
                    iwin = iint[np.where(targetinfo[i].weight[iint] > 0.)[0][:]]  # ind. of weights>0 in current window
                    if len(iwin)==0:
                        continue  # go to next observation if no non-zero weights in current window
                    else:
                        if verbose: print('iwin', iwin)
                        if (len(iwin) >= 2):  # if window >=
                            if verbose: print('i, obs. weights:',i,targetinfo[i].weight[iint])
                            i_wmax = iwin[np.argmax(targetinfo[i].weight[iwin])]  # index of max weight
                            wmax = targetinfo[i].weight[i_wmax]  # maximum weight
                        else:  # if window size of 1
                            wmax = targetinfo[i].weight[iwin]  # maximum weight

                        if (wmax > maxweight):
                            maxweight = wmax
                            iimax = i
                            iwinmax = iwin

                        if verbose:
                            print('maxweight',maxweight)
                            print('iimax',iimax)

                if (iimax == -1):
                    gow = False
                else:

                    # Time interval limited by observing window or plan window?
                    istart = iwinmax[0]
                    iend = iwinmax[-1]
                    nobswin = iend - istart + 1

                    if verbose:
                        print('Inst: ',obs.inst[iimax])
                        print('Disperser: ',obs.disperser[iimax])

                    if (obs.inst[iimax]!='GMOS'):
                        if np.logical_and(obs.disperser[iimax]!='Mirror',obs.disperser[iimax]=='null'):
                            ntcal = 3
                            if verbose: print('add ntcall=3')

                    # total time remaining in observation, include calibration time
                    ttime = np.round((obs.tot_time[iimax] - obs.obs_time[iimax])*10. + 0.5*u.h) /10
                    nttime = int(np.round(ttime / dt) + ntcal) #number of steps

                    # Try not to leave little pieces of programs
                    if (nttime - nminuse) <= nminuse:
                        nminuse = nttime

                    # Set weights to zero if observation window is too short
                    # to schedule selected observation.  Otherwise, continue.
                    if np.logical_or(nttime<nobswin,nobswin>nminuse):
                        gow = False
                    else:
                        targetinfo[iimax].weight[iint] = 0.
                        if verbose: print('Block too short to schedule...')

                    if verbose:
                        print('ID of chosen ob.',targetinfo[iimax].name)
                        print('weights of chosen ob.',targetinfo[iimax].weight)
                        print('Current plan',plan)
                        print('istart',istart)
                        print('iend',iend)
                        print('ttime',ttime)
                        print('nttime',nttime)
                        print('nobswin',nobswin)
                        print('nminuse',nminuse)

            if iimax == -1:
                plan[iint] = -2
            else:
                # pick optimal observing window by integrating weight function
                if np.logical_and(nttime < nobswin , nttime != 0):

                    maxf = 0.0

                    if nttime > 1:
                        # NOTE: integrates over one extra time slot...
                        # ie. if nttime = 14, then the program will choose 15
                        # x values to do trapz integration (therefore integrating
                        # 14 time slots).
                        if verbose:
                            print('\nIntegrating max obs. over window...')
                            print('istart',istart)
                            print('iend',iend)
                            print('nttime',nttime)
                            print('j values',np.arange(istart,iend-nttime+2))
                        for j in range(istart,iend-nttime+2):
                            f = sum(targetinfo[iimax].weight[j:j+nttime])
                            if verbose:
                                print('j range',j,j+nttime-1)
                                print('obs wieght',targetinfo[iimax].weight[j:j+nttime])
                                print('integral',f)
                            if f>maxf:
                                maxf = f
                                jstart = j

                        jend = jstart + nttime - 1

                    else:
                        jstart = np.argmax(targetinfo[iimax].weight[iwinmax])
                        maxf = np.amax(targetinfo[iimax].weight[jstart])
                        jend = jstart + nttime - 1

                    if verbose:
                        print('max integral of weight func (maxf)',maxf)
                        print('index jstart',jstart)
                        print('index jend',jend)

                    if jstart < nminuse:  # shift block if near start or end of night
                        if np.logical_and(plan[0]==-1 , targetinfo[iimax].weight[0]>0. ):
                            jstart = 0
                            jend = jstart + nttime - 1
                    elif (nt-jend) < nminuse:
                        if np.logical_and(plan[-1]==-1 , targetinfo[iimax].weight[-1]>0. ):
                            jend = nt - 1
                            jstart = jend - nttime + 1

                    # dstart = jstart - istart - 1
                    # wstart = obslist[iimax]['weight'][istart]
                    # dend = iend - jend + 1
                    # wend = obslist[iimax]['weight'][iend]
                    # if np.logical_and(dstart < nminuse , dend < nminuse):
                    #     if np.logical_and(wstart > wend , wstart > 0.):
                    #         jstart = istart
                    #         jend = istart + nttime - 1
                    #     elif wend > 0.:
                    #         jstart = iend - nttime + 1
                    #         jend = iend
                    # elif np.logical_and(dstart < nminuse , wstart > 0.):
                    #     jstart = istart
                    #     jend = istart + nttime - 1
                    # elif np.logical_and(dend < nminuse , wstart > 0.):
                    #     jstart = iend - nttime + 1
                    #     jend = iend
                    # else:
                    #     if (jj[0] < istart):
                    #         jstart = istart
                    #         jend = istart + nttime - 1
                    #     else:
                    #         jstart = iend - nttime + 1
                    #         jend = iend

                else:
                    jstart = istart
                    jend = iend
                    #set weights of scheduled times to negatives

                if verbose:
                    print('Chosen index jstart',jstart)
                    print('Chosen index jend',jend)
                    print('Current obs time: ', obs.obs_time[iimax])
                    print('Current tot time: ', obs.tot_time[iimax])

                nsel = nsel + 1  # increment number of selected observations
                plan[jstart:jend+1] = iimax  # set plan slots to index of scheduled observation
                ntmin = np.minimum(nttime - ntcal, nobswin)  # observation time slots scheduled (minus cal)
                obs.obs_time[iimax] = obs.obs_time[iimax] + dt * ntmin  # update observed time
                # ii_obs = np.where(obs.obs_id == obs.prog_ref[iimax])[0][:]  # indices of obs. in same program
                obs.comp_time[iimax] = obs.comp_time[iimax] + dt * ntmin / obs.tot_time[iimax]  # update comp_time

                if obs.comp_time[iimax]>=1.:  # completed observation
                    targetinfo[iimax].weight = targetinfo[iimax].weight * -1  # set all target weights to neg.
                else:  # incomplete observation
                    targetinfo[iimax].weight[jstart:jend+1] = targetinfo[iimax].weight[jstart:jend+1] * -1  # set scheduled target weights to neg.
                    wpositive = np.where(targetinfo[iimax].weight >= 0)[0][:]
                    targetinfo[iimax].weight[wpositive] = targetinfo[iimax].weight[wpositive] * 1.5  # increase remaining weights.

                if verbose:
                    print('New obs. weights: ', targetinfo[iimax].weight)
                    print('nttime - ntcal , nobswin: ',nttime - ntcal , nobswin)
                    print('ntmin: ',ntmin)
                    print('Obs tot time: ', obs.tot_time[iimax])

                # add time to total if observation was not fully completed
                if obs.obs_time[iimax] < obs.tot_time[iimax]:
                    obs.tot_time[iimax] = obs.tot_time[iimax] + 0.3*u.h

                if verbose:
                    print('Current program: ',plan)
                    print('New obs time: ', obs.obs_time[iimax])
                    print('New comp time: ', obs.comp_time[iimax])

                if schedule_order:
                    print('\tScheduled: ',iimax,targetinfo[iimax].name,'from',timeinfo.utc[jstart].iso,'to',timeinfo.utc[jend].iso)
                    print(targetinfo[iimax].weight)

            ii = np.where(plan == -1)[0][:]  # get indices of remaining time slots

        # finalize plan and retrieve details
        plan = _optimize_plan(plan, istart, targetinfo)  # Final plan
        obslist, i_start, i_end = _plan_details(plan, targetinfo)  # plan details
        hours = (i_end-i_start+1)*dt  # lengths of scheduled observations
        ii = np.where(plan >= 0)[0][:] # scheduled time slots
        tot_time = len(ii) * dt  # total scheduled time
        night_length = timeinfo.night_length
        cplt = np.full(len(obslist), False, dtype=bool)  # completion status of scheduled obs.
        for i in range(0, len(i_start)):
            print(plan[i_start[i]], obs.comp_time[plan[i_start[i]]])
            if obs.comp_time[plan[i_start[i]]] >= 1.:
                cplt[i] = True

        return cls(plantype=plantype, plan=plan, obslist=obslist, i_start=i_start, i_end=i_end, hours=hours,
                   cplt=cplt, tot_time=tot_time, night_length=night_length),obs