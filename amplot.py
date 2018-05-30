import numpy as np
import matplotlib.pyplot as plt

def amplot(obslist,plan,prog_status):

    thk=4
    thn=1

    uniq_id,uniq_rr = np.unique(plan['isel'], return_inverse=True)
    print('uniq_id',uniq_id)
    print('uniq_rr',uniq_rr)

    plt.plot_date(plan['UT'].plot_date,plan['mAM'],linestyle=':',label='Moon',linewidth=thn,color='grey')

    for i in range(len(uniq_id)):
        iimax = int(uniq_id[i])
        if iimax>0:
            ii = np.where(plan['isel']==iimax)[0][:]
            np.append(ii,np.amax(ii)+1)
            # print('ii',ii)
            temp = plan['UT'][ii]
            # print('temp',temp)
            # print('plan[\'UT\'][ii].plot_date',plan['UT'][ii].plot_date)
            # print('obslist[iimax][\'AM\'][ii]',obslist[iimax]['AM'][ii])
            plt.plot_date(plan['UT'].plot_date,obslist[iimax]['AM'],linestyle='-',linewidth=thn,color='black')
            plt.plot_date(plan['UT'][ii].plot_date,obslist[iimax]['AM'][ii],linestyle='-',linewidth=thk)
            
            print('Obs ID: ',prog_status['obs_id'][iimax])
            print('Observation information: ',obslist[iimax])

    plt.ylim(2, 1)
    plt.show()
    exit()