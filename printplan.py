import numpy as np
import astropy

def printplan(i_obs,obslist,plan,prog_status,otcat,time_diff_utc):
    
    verbose = False

    sprint = '{0:20s}{1:40s}{2:20s}{3:>30s}{4:>30s}{5:>10s}{6:>10s}{7:>10s}' 
    fprint = '{0:20s}{1:40s}{2:20s}{3:>30s}{4:>30s}{5:10.2f}{6:10.2f}{7:10.2f}'
    print(sprint.format('Obs. ID','Target','Instrument','Local','UT start','LST','AirM','HA'))
    print(sprint.format('-------','------','---------','-----','--------','---','----','--'))


    sel_obs,ii,ri,count = np.unique(plan['isel'],return_index=True, return_inverse=True, return_counts=True)
    ii_sort = np.sort(ii)

    if verbose:
        print(sel_obs,ii,ri,count)
        print('ii_sort',ii_sort)

    for i in range(0,len(ii_sort)):
        
        time_index = int(ii_sort[i]) #get obs_index of next obs start time
        obs_index = int(sel_obs[ri[time_index]]) #get obs obs_index corresponding to plan['isel']
        if verbose:
            print('time_index',time_index)
            print('obs_index',obs_index)

        if obs_index>0: #Do not print info for empty time blocks

            obs_name = prog_status['obs_id'][obs_index]
            targ_name = prog_status['target'][obs_index]
            inst_name = otcat.inst[i_obs[obs_index]]
            local_time = (plan['UT'][time_index]+time_diff_utc).iso
            lst_time = plan['lst'][time_index]
            utc_start = plan['UT'][time_index].iso
            airmass = obslist[obs_index]['AM'][time_index]
            HA = obslist[obs_index]['HA'][time_index]
            # print(type(local_time),type(lst_time),type(utc_start),type(airmass),type(HA))
            print(fprint.format(obs_name,targ_name,inst_name,local_time,utc_start,lst_time,airmass,HA))
     

    return