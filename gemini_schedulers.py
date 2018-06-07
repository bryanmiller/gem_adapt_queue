import scipy
import numpy as np
import astropy.units as u
from intervals import intervals

def priority_scheduler(i_obs, n_obs, obslist, plan, prog_status, otcat):

    verbose = False
    
    dt = plan['dt']
    n_timesteps = plan['n_timesteps']
    ntcal = 0
    plan['isel'][:] = -1
    nsel = 0
    ii = np.where(plan['isel'] == -1)[0][:]
    
    while len(ii)!=0:
        
        if verbose: print('\nIteration: ',nsel+1)

        #define min block
        nminuse = 10

        #bin indices of adjacent timeslots
        indx = intervals(ii)

        #get indices of first unscheduled time block 
        iint = ii[np.where(indx==1)[0][:]]
        
        if verbose:
            print('ii:',ii)
            print('iint: ',iint)
            # print('indx',indx)
            # print('iint',iint)

        gow = True
        while gow: #schedule observation with maximum weight in time interval
            maxweight = 0.
            iimax = -1

            for i in np.arange(n_obs):
                if (len(ii) >= 2):
                    if verbose: print('i, obs. weights:',i,obslist[i]['weight'][iint])
                    i_wmax = np.argmax(obslist[i]['weight'][iint])+iint[0]
                    wmax = obslist[i]['weight'][i_wmax]
                else:
                    wmax = obslist[i]['weight'][iint]

                if (wmax > maxweight):
                    maxweight = wmax
                    iimax = i

                if verbose:
                    print('maxweight',maxweight)
                    print('iimax',iimax)

            if (iimax == -1):
                gow = False
            else:
                if verbose:
                    print('iobswin',obslist[iimax]['iobswin'])
                    print('iint',iint)
                # Time interval limited by observing window or plan window?
                istart = np.maximum(obslist[iimax]['iobswin'][0],iint[0])
                iend = np.minimum(obslist[iimax]['iobswin'][1],iint[-1])
                nobswin = iend - istart + 1

                if verbose:
                    print('nobswin',nobswin)
                    print('Inst: ',otcat.inst[i_obs[iimax]])
                    print('Disperser: ',otcat.disperser[i_obs[iimax]])

                if (otcat.inst[i_obs[iimax]]!='GMOS'):
                    if np.logical_and(otcat.disperser[i_obs[iimax]]!='Mirror',otcat.disperser[i_obs[iimax]]=='null'):
                        ntcal = 3
                        if verbose: print('add ntcall=3')

                # total time remaining in observation, include calibration time
                ttime = np.round((prog_status.tot_time[iimax] - prog_status.obs_time[iimax]) *10. + 0.5*u.h ) /10
                nttime = np.int(np.round(ttime / dt) + ntcal) #number of steps

                # Try not to leave little pieces of programs
                # if (nttime - nminuse) <= nminuse:
                #     nminuse = nttime

                #Set weights to zero if observation can be done within window
                if np.logical_or(nttime<nobswin,nobswin>nminuse):
                    gow = False
                else:
                    obslist[iimax]['weight'][iint] = 0.
                    if verbose: print('Block too short to schedule...')

                if verbose:
                    print('ID of chosen ob.',obslist[iimax]['id'])
                    print('weights of chosen ob.',obslist[iimax]['weight'])
                    print('istart',istart)
                    print('iend',iend)
                    print('ttime',ttime)
                    print('nttime',nttime)
                    print('nobswin',nobswin)
                    print('nminuse',nminuse)

        if iimax == -1:
            plan['isel'][iint] = -2
        else:
            # pick optimal observing window by integrating weight function
            if np.logical_and(nttime < nobswin , nttime != 0):
                
                #indices where obs. is already scheduled
                jj = np.where(plan['isel']==iimax)[0][:]

                x = np.arange(nttime,dtype=float)
                maxf = 0.0
                jstart = 0
            
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
                        # print('obs weights all: ',obslist[iimax]['weight'])
                        print('j values',np.arange(istart,iend-nttime+2))
                    for j in range(istart,iend-nttime+2):
                        f = scipy.integrate.trapz(obslist[iimax]['weight'][j:j+nttime],x)
                        if verbose:
                            print('j',j)
                            print('obs wieght',obslist[iimax]['weight'][j:j+nttime])
                            print('x',x)
                            print('integral',f)
                        if f>maxf:
                            maxf = f
                            jstart = j

                    jend = jstart + nttime - 1

                else:
                    jstart = np.argmax(obslist[iimax]['weight'][iint])
                    maxf = np.amax(obslist[iimax]['weight'][jstart])
                    jend = jstart + nttime - 1

                if verbose:
                    print('maxf',maxf)    
                    print('jstart',jstart)
                    print('jend',jend)

                if jstart < nminuse:
                    if np.logical_and( plan['isel'][0]==-1 , obslist[iimax]['weight'][0]>0. ):
                        jstart = 0
                        jend = jstart + nttime - 1
                elif (n_timesteps-jend) < nminuse:
                    if np.logical_and( plan['isel'][n_timesteps-1]==-1 , obslist[iimax]['weight'][n_timesteps-1]>0. ):
                        jend = n_timesteps - 1
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
                    
                #set weights of scheduled times to negatives
                obslist[iimax]['weight'] = obslist[iimax]['weight'] * -1

            else:
                jstart = istart
                jend = iend
                #set weights of scheduled times to negatives
                obslist[iimax]['weight'] = obslist[iimax]['weight'] * -1

            if verbose:
                print('Final jstart',jstart)
                print('Final jend',jend)
                print('New obs. weights: ',obslist[iimax]['weight'])
            
            #increment number of selected observations
            nsel = nsel + 1

            #set timeslots in plan to index of scheduled observation 
            plan['isel'][jstart:jend] = iimax
            
            if verbose:
                print('Current obs time: ',prog_status.obs_time[iimax])
                print('Current tot time: ',prog_status.tot_time[iimax])

            #update time observed
            ntmin = np.minimum(nttime - ntcal , nobswin)
            if verbose:
                print('nttime - ntcal , nobswin: ',nttime - ntcal , nobswin)
                print('ntmin: ',ntmin)

            prog_status.obs_time[iimax] = prog_status.obs_time[iimax] + dt*ntmin
            if verbose:
                print('New obs time: ',prog_status.obs_time[iimax])
                print('Obs tot time: ',prog_status.tot_time[iimax])

            #update time status for observations in program
            ii_obs = np.where(prog_status.prog_id == otcat.prog_ref[i_obs[iimax]])[0][:]
            prog_status.comp_time[iimax] = prog_status.comp_time[iimax] + dt*ntmin/prog_status.tot_time[iimax]
            if verbose: print('New comp time: ',prog_status.comp_time[iimax])

            #add time to total if observation was not fully completed
            if prog_status.obs_time[iimax] < prog_status.tot_time[iimax]:
                prog_status.tot_time[iimax] = prog_status.tot_time[iimax] + 0.3*u.h

            # #if ntmin == nttime, observation is complete. Set all weights to negative
            # if prog_status['obs_time'][iimax] >= prog_status['tot_time'][iimax]:
            #     if (jstart!=0):
            #         obslist[iimax]['weight'][0:(jstart-1)] = obslist[iimax]['weight'][0:(jstart-1)] * -1
            #     if (jend!=0):
            #         obslist[iimax]['weight'][(jend+1):n_timesteps-1] = obslist[iimax]['weight'][(jend+1):n_timesteps-1] * -1    
        
            # if verbose: 
            # print('Obs. added to program:',prog_status['obs_id'][iimax],obslist[iimax])
        
        if verbose: print('Current program: ',plan['isel'])

        ii = np.where(plan['isel'] == -1)[0][:]

    # print('Runtime = ',t.time()-starttime) 

    return plan, obslist, prog_status
