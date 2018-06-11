import numpy as np
import astropy

def printplan(i_obs,obslist,plan,prog_status,otcat,time_diff_utc):
    
    verbose = False

    sprint = '\t{0:20.20s}{1:14.12s}{2:7.5s}{3:12.8}{4:12.8}{5:8.6s}{6:8.6s}{7:8.6s}{8:8.6s}' 
    fprint = '\t{0:20.20s}{1:14.12s}{2:7.5s}{3:12.8}{4:12.8}{5:<8.4s}{6:<8.2f}{7:<8.2f}{8:<8.5s}'
    print('\n\tPlan type: '+plan['type']+' ')
    print(sprint.format('Obs. ID','Target','Instr','Local','UT','LST','Hrs','AirM','HA'))
    print(sprint.format('-------','------','-----','-----','--------','---','---','----','--'))


    sel_obs,ii,ri,count = np.unique(plan['isel'],return_index=True, return_inverse=True, return_counts=True)
    ii_sort = np.sort(ii)

    if verbose:
        print(sel_obs,ii,ri,count)
        print('ii_sort',ii_sort)

    for i in range(0,len(ii_sort)):
        
        time_index = int(ii_sort[i]) #get obs_index of next obs start time
        obs_index = int(sel_obs[ri[time_index]]) #get obs obs_index corresponding to plan['isel']
        
        n_slots = 1
        while (time_index + (n_slots)) < len(ri):
            if sel_obs[ri[time_index + (n_slots)]] == obs_index:
                n_slots = n_slots + 1
            else:
                break
            
        if verbose:
            print('time_index',time_index)
            print('obs_index',obs_index)

        if obs_index>0: #Do not print info for empty time blocks

            obs_name = prog_status.obs_id[obs_index]
            targ_name = prog_status.target[obs_index]
            inst_name = otcat.inst[i_obs[obs_index]]
            local_time = str((plan['UT'][time_index]+time_diff_utc).iso)[11:-4]
            lst_time = plan['lst'][time_index]
            utc_start = str(plan['UT'][time_index].iso)[11:-4]
            airmass = obslist[obs_index]['AM'][time_index]
            HA = '{:.2f}'.format(obslist[obs_index]['HA'][time_index])
            duration = n_slots * 1/10
            print(fprint.format(obs_name,targ_name,inst_name,local_time,utc_start,lst_time,duration,airmass,HA))
            # print(type(local_time),type(lst_time),type(utc_start),type(airmass),type(HA))

    return