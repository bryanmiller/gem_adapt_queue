import numpy as np
import matplotlib.pyplot as plt

def amplot(plan,targetinfo,timeinfo,mooninfo):

    thk=4
    thn=1

    plt.plot_date(timeinfo.utc.plot_date,mooninfo.AM,linestyle=':',label='Moon',linewidth=thn,color='grey',markersize=0)

    for i in range(0,len(plan.i_start)):
        i_obs = plan.plan[plan.i_start[i]]
        if i_obs>0:
            jstart = plan.i_start[i] # observation start time index
            jend = plan.i_end[i] + 1 # observation end time index
            plt.plot_date(timeinfo.utc.plot_date,targetinfo[i_obs].AM,linestyle='-',linewidth=thn,color='black',markersize=0)

    for i in range(0,len(plan.i_start)):
        i_obs = plan.plan[plan.i_start[i]]
        if i_obs>0:
            jstart = plan.i_start[i] # observation start time index
            jend = plan.i_end[i] + 1 # observation end time index
            plt.plot_date(timeinfo.utc[jstart:jend].plot_date,targetinfo[i_obs].AM[jstart:jend],linestyle='-',linewidth=thk,markersize=0,label=plan.obs_id[i][-10:])

    date = str(timeinfo.local[0].iso)[0:10]
    plt.title(date+' schedule')
    plt.ylim(2.1, 0.9)
    plt.ylabel('Airmass')
    plt.xlabel('UTC')
    plt.legend(loc=8,ncol=4,fontsize=8,markerscale=0.5)
    plt.savefig(date+'_amplot.png')
    plt.clf()

    return