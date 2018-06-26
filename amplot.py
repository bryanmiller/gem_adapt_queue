import matplotlib.pyplot as plt

def amplot(plan,targetinfo,timeinfo,mooninfo):

    thk=4
    thn=1

    plt.plot_date(timeinfo.utc.plot_date,mooninfo.AM,linestyle=':',label='Moon',linewidth=thn,color='grey',markersize=0)

    for i in range(0,len(plan.i_start)):
        i_obs = plan.plan[plan.i_start[i]]
        if i_obs>0:
            plt.plot_date(timeinfo.utc.plot_date,targetinfo[i_obs].AM,linestyle='-',linewidth=thn,color='black',markersize=0)

    for i in range(0,len(plan.i_start)):
        i_obs = plan.plan[plan.i_start[i]]
        if i_obs>0:
            jstart = plan.i_start[i] # observation start time index
            jend = plan.i_end[i] + 1 # observation end time index
            plt.plot_date(timeinfo.utc[jstart:jend+1].plot_date,targetinfo[i_obs].AM[jstart:jend+1],linestyle='-',linewidth=thk,markersize=0,label=plan.obs_id[i][-10:])

    date = str(timeinfo.local[0].iso)[0:10]
    plt.title(date+' schedule')
    plt.ylim(2.1, 0.9)
    plt.xlim(timeinfo.utc[0].plot_date,timeinfo.utc[-1].plot_date)
    plt.xticks(rotation=20)
    plt.ylabel('Airmass')
    plt.xlabel('UTC')
    plt.legend(loc=8,ncol=4,fontsize=8,markerscale=0.5)
    plt.tight_layout()
    plt.savefig('amplot'+date+'.png')
    plt.clf()

    return

