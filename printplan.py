import numpy as np

def printPlanTable(plan, i_obs, obs, timeinfo, targetinfo):
    verbose = False

    sprint = '\t{0:14.12s}{1:12.10s}{2:7.5s}{3:7.5s}{4:7.5s}{5:7.5s}{6:8.6s}{7:8.6s}{8:7.5s}{9:8.6s}{10:10.10}'
    table = []

    table.append(str('\n\t'+plan.type+' schedule:'))
    table.append(str(sprint.format('Obs. ID','Target','Instr','UTC','LST','Start','End','Hrs','AM','HA','Completed')))
    table.append(str(sprint.format('-------','------','-----','---','---','-----','---','---','--','--','---------')))

    order = np.argsort(plan.i_start)

    if verbose:
        print('order',order)

    for i in order:

        i_select = np.where(obs.obs_id[i_obs] == plan.obs_id[i])[0][0]

        obs_name = plan.obs_id[i][-10:]
        targ_name = obs.target[i_obs[i_select]]
        inst_name = obs.inst[i_obs[i_select]]
        utc_start = str(timeinfo.utc[plan.i_start[i]].iso)[11:16]
        lst_start = str('{:.2f}'.format(timeinfo.lst[plan.i_start[i]].hour))
        local_start = str((timeinfo.local[plan.i_start[i]]).iso)[11:16]
        local_end = str((timeinfo.local[plan.i_start[i]]).iso)[11:16]
        airmass = str('{:.2f}'.format(targetinfo[i_select].AM[plan.i_start[i]]))
        HA = str('{:.2f}'.format(targetinfo[i_select].HA[plan.i_start[i]]))
        hours = str('{:.2f}'.format(plan.hours[i]))
        cplt = str(plan.cplt[i])
        table.append(str(sprint.format(obs_name,targ_name,inst_name,utc_start,lst_start,
                            local_start,local_end,hours,airmass,HA,cplt)))

    return table