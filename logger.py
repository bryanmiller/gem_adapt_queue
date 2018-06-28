# Logging functions for gqpt statistics and results
# Matt Bonnyman
# 2018-06-27

import os
import numpy as np
import astropy.units as u
from astropy import coordinates
from printplan import printPlanTable

def initLogFile(filename, catalogfile, site, start, n_nights, dst):
    aprint = '\n\t{0:<15s}{1}'  # print two strings
    bprint = '\n\t{0:<15s}{1:<.4f}'  # print string and number
    with open(filename, 'w') as log:
        log.write('\n Program: gqpt.py')
        log.write('\n Run from directory: ' + os.getcwd())
        log.write('\n Observation information retrieved from '+catalogfile)
        log.write('\n\n Dates: '+str(start)[0:10]+' to '+str(start+(n_nights-1)*u.d)[0:10])
        log.write('\n Number of nights: '+str(n_nights))
        log.write('\n Daylight savings time: ' + str(dst))
        log.write('\n\n Observing site: ')
        log.write(aprint.format('Site: ', site.name))
        log.write(bprint.format('Height: ', site.location.height))
        log.write(bprint.format('Longitude: ', coordinates.Angle(site.location.lon)))
        log.write(bprint.format('Latitude: ', coordinates.Angle(site.location.lat)))
        log.close()
    return

def logPlanStats(filename, obs, plan, timeinfo, suninfo, mooninfo, targetinfo, acond):

    with open(filename, 'a') as log:
        log.write('\n\n -----------------------------------------------------------')
        log.write('\n\n Night schedule:')
        log.write('\n\tSky conditions (iq,cc,wv): {0} , {1} , {2}'.format(acond[0], acond[1], acond[3]))
        [log.write('\n'+line) for line in timeinfo.table()]
        [log.write('\n' + line) for line in suninfo.table()]
        [log.write('\n' + line) for line in mooninfo.table()]
        [log.write('\n' + line) for line in plan.table()]
        [log.write('\n' + line) for line in printPlanTable(plan=plan, obs=obs, timeinfo=timeinfo, targetinfo=targetinfo)]
        log.close()

    return

def logProgStats(filename, obs, semesterinfo, description):
    """
    Computes and appends observations statistics to a file.

    Example
    -------

    >>> _log_progstat(filename='myfile.log', obs=Gobservations(catinfo=Gcatfile(otfile),i_obs=i_obs))

     Program             Completion     Total time
     -------             ----------     ----------
     GS-2018A-A-1        100%           1.02 h
     GS-2018A-B-1        2.8%           2.68 h
     GS-2018A-B-2        0.03%          6.68 h
     GS-2018A-C-1        1.02%          12.7 h
     GS-2018A-C-2        100%           0.4 h
     GS-2018A-C-3        30.32%         14.84 h
     GS-2018A-C-4        43.74%         2.51 h
     GS-2018A-C-5        44.96%         0.63 h
     GS-2018A-C-6        0.96%          21.91 h
     GS-2018A-C-7        83.34%         1.82 h
     GS-2018A-C-8        93.1%          12.03 h
     GS-2018A-C-9        65.69%         14.61 h
     GS-2018A-C-10       9.68%          4.14 h
     GS-2018A-C-11       51.86%         9.17 h

     Number of programs: 28
     Total program time: 341.63 h
     Total time completion: 10.35%

     Completed: 2
     Observed time: 1.43 h

     Partially completed: 12
     Observed time: 33.77 h
     Remaining time: 69.95 h

     Not started: 14
     Remaining time: 236.48 h

     Parameter
     ---------
     filename : string
         log file name including extension type (eg. 'logfile2018-01-01.log')

     obs : 'gemini_classes.Gobservations'
         Observation information object
    """

    progname, rr, ri, rc = np.unique(obs.prog_ref, return_index=True, return_inverse=True, return_counts=True)
    prog_comp_time = 0. * u.h  # total time of completed programs
    prog_start_time = 0. * u.h  # total time of partially completed programs
    obs_start_time = 0. *u.h  # total observed time for partially completed program
    prog_unstart_time = 0. * u.h  # total time of unstarted programs
    num_prog_comp = 0  # number of completed programs
    num_prog_started = 0  # number of partially completed programs
    num_prog_unstarted = 0  # number of not yet started programs

    aprint = '\n {0:<20}{1:<15}{2:<15}'
    with open(filename, 'a') as log:

        log.write('\n\n -----------------------------------------------------------')
        log.write('\n\n ' + description)
        log.write('\n\n'+aprint.format('Program', 'Completion', 'Total time'))
        log.write(aprint.format('-------', '----------', '----------'))
        for i in range(len(rr)):
            jj = np.where(obs.obs_comp[rr[i]:rr[i]+rc[i]]>0.)[0][:]
            if len(jj) > 0: # started or completed obs in program
                prog_time = sum(obs.tot_time[rr[i]:rr[i]+rc[i]])
                sum_obs_time = sum(obs.obs_time[rr[i]:rr[i]+rc[i]])
                frac_comp = sum_obs_time/prog_time
                if frac_comp >=1.:
                    perc_comp = '100%'
                    prog_comp_time = prog_comp_time + prog_time
                    num_prog_comp = num_prog_comp + 1
                else:
                    perc_comp = str((100*frac_comp).round(2))+'%'
                    obs_start_time = obs_start_time + sum_obs_time
                    prog_start_time = prog_start_time + prog_time
                    num_prog_started = num_prog_started + 1
                log.write(aprint.format(progname[i], perc_comp, str(prog_time.round(2))))
            else:
                prog_unstart_time = prog_unstart_time + sum(obs.tot_time[rr[i]:rr[i] + rc[i]])
                num_prog_unstarted = num_prog_unstarted + 1

        tot_prog_time = sum(obs.tot_time)
        tot_obs_time = sum(obs.obs_time)
        log.write('\n\n Number of programs: ' + str(len(progname)))
        log.write('\n Total program time: ' + str(tot_prog_time.round(2)))
        log.write('\n Total time completion: ' + str((100 * tot_obs_time / tot_prog_time).round(2)) + '%')

        log.write('\n\n Total observable time: '+str(semesterinfo['night_time'].round(2)))
        log.write('\n Total scheduled time: ' + str(semesterinfo['used_time'].round(2)))

        log.write('\n\n Completed: ' + str(num_prog_comp))
        log.write('\n Observed time: '+str(prog_comp_time.round(2)))

        log.write('\n\n Partially completed: ' + str(num_prog_started))
        log.write('\n Observed time: ' + str(obs_start_time.round(2)))
        log.write('\n Remaining time: ' + str((prog_start_time-obs_start_time).round(2)))

        log.write('\n\n Not started: ' + str(num_prog_unstarted))
        log.write('\n Remaining time: ' + str(prog_unstart_time.round(2)))

        log.close()
    return