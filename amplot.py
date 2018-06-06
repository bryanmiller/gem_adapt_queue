import numpy as np
import matplotlib.pyplot as plt

def amplot(obslist,plan,prog_status):

    thk=4
    thn=1

    uniq_id,uniq_rr = np.unique(plan['isel'], return_inverse=True)

    plt.plot_date(plan['UT'].plot_date,plan['mAM'],linestyle=':',label='Moon',linewidth=thn,color='grey',markersize=0)

    for i in range(len(uniq_id)):
        iimax = int(uniq_id[i])
        if iimax>0:
            ii = np.where(plan['isel']==iimax)[0][:]
            np.append(ii,np.amax(ii)+1)
            temp = plan['UT'][ii]
            # print('Plotting: ',prog_status.obs_id[iimax])
            plt.plot_date(plan['UT'].plot_date,obslist[iimax]['AM'],linestyle='-',linewidth=thn,color='black',markersize=0)
            plt.plot_date(plan['UT'][ii].plot_date,obslist[iimax]['AM'][ii],linestyle='-',linewidth=thk,markersize=0)

    plt.ylim(2.1, 0.9)
    plt.ylabel('Airmass')
    plt.xlabel('UTC')
    plt.show()
    
    return