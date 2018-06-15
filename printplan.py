import numpy as np
import astropy

def printplan(plan,obs,timeinfo,targetinfo):
    
    verbose = False

    sprint = '\t{0:14.12s}{1:12.10s}{2:7.5s}{3:10.8s}{4:8.6s}{5:10.8s}{6:10.8s}{7:8.6s}{8:8.6s}{9:8.6s}'
    # print('\n\tPlan type: '+plan['type']+' ')
    print('\n\t'+plan.type+' schedule:')
    print(sprint.format('Obs. ID','Target','Instr','UTC','LST','Start','End','Hrs','AM','HA'))
    print(sprint.format('-------','------','-----','---','---','-----','---','---','--','--'))

    order = np.argsort(plan.i_start)

    # sel_obs,ii,ri,count = np.unique(plan,return_index=True, return_inverse=True, return_counts=True)
    # ii_sort = np.sort(ii)
    if verbose:
        # print(sel_obs,ii,ri,count)
        print('order',order)

    for i in order:

        i_obs = np.where(obs.obs_id==plan.obs_id[i])[0][0]

        obs_name = plan.obs_id[i][-10:]
        targ_name = obs.target[i_obs]
        inst_name = obs.inst[i_obs]
        utc_start = str(timeinfo.utc[plan.i_start[i]].iso)[11:19]
        lst_start = str(timeinfo.lst[plan.i_start[i]])
        local_start = str((timeinfo.local[plan.i_start[i]]).iso)[11:19]
        local_end = str((timeinfo.local[plan.i_start[i]]).iso)[11:19]
        airmass = str('{:.2f}'.format(targetinfo[i_obs].AM[plan.i_start[i]]))
        HA = str('{:.2f}'.format(targetinfo[i_obs].HA[plan.i_start[i]]))
        hours = str('{:.2f}'.format(plan.hours[i]))
        print(sprint.format(obs_name,targ_name,inst_name,utc_start,lst_start,
                            local_start,local_end,hours,airmass,HA))

    return