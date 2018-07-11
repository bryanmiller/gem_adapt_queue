import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from sb import sb
import calc_weight
from gcirc import gcirc
from intervals import intervals
from printplan import printPlanTable
from gemini_classes import TargetInfo

ttimer = False
if ttimer: import time as t

def plotweightfunc(id,timeinfo,weights):
    """
    Show plot of target weighting function with respect to time.

    Parameters
    ----------
    id : str
        Target/Observation identifier

    timeinfo : '~gemini_classes.TimeInfo'
        Times object containing array of UTC times and solar midnight as
        '~astropy.time.Time' object types

    weight : np.array of floats
        weighting function values. Must be same shape as time array in timeinfo.
    """
    timedelt = (timeinfo.utc - timeinfo.midnight).value * 24

    ii = np.where(weights>0)[0][:]
    if len(ii)!=0:
        i_intervals = intervals(ii)
        wmin = np.min(weights[ii])
        wmax = np.max(weights[ii])
    else:
        print(' No non-zero weights.')
        return

    n_int = np.unique(i_intervals)
    for n in n_int:
        iint = ii[np.where(i_intervals == n)[0][:]]
        plt.plot(timedelt[iint], weights[iint], color='black')

    plt.title(str(timeinfo.utc[0].iso)[0:10]+': '+id)
    plt.xlim(timedelt[0], timedelt[-1])
    plt.ylim(wmin,wmax)
    plt.ylabel('Weight')
    plt.xlabel(r'$\Delta time$ from solar midnight (hrs)')
    plt.tight_layout()
    plt.show(block=False)
    input('\n Press enter to close window...')
    plt.close()
    plt.clf()
    return

def plotcomponents(id, timeinfo, function, dec, AM, HA, AZ, band, user_prior, prstatus, latitude, cond, acond, obstatus, wra, elev):
    """
    Plot or print the individual components of the weighting function.
    Plot values if time dependent, otherwise print to console.

    Parameters
    ----------
    function : str
        String representation of weighting function.  Components will be selected from this string.

    cond : dict
        required sky conditions

    acond : dict
        actual sky conditions

    dec : 'astropy.coordinates.angles.Latitude'
        target declination

    AM : array of floats
        Target airmass at times throughout observing window.

    HA   - target hour angles

    AZ   - target azimuths

    latitude : 'astropy.coordinates.angles.Latitude'
        observer latitude

    band : int
        ranking band (1, 2, 3, 4)

    user_prior : str
        user priority (Low, Medium, High)

    pstatus : float
        program completion as fraction

    ostatus : float
        observation completion as fraction

    wind : list of '~astropy.unit.Quantity'
        wind conditions (speed (m/s) and direction)

    wra - RA distribution weights

    elev - elevation constraint
    """
    print('\n\t{} weights\n\t-------------------------'.format(id))
    aprint = '\t{0:<25s}{1}'  # print two strings
    f = '{} ({})'

    if 'twcond' in function:
        func = 'iq={}, cc={}, bg={}, wv={}'.format(cond['iq'],cond['cc'],cond['bg'],cond['wv'])
        twcond = calc_weight.weight_tot_cond(cond=cond)
        print(aprint.format('total cond: ',f.format(str(twcond),func)))
    if 'wra' in function:
        print(aprint.format('ra: ', wra))
    if 'wband' in function:
        func = 'band {}'.format(band)
        wband = calc_weight.weight_band(band=band)
        print(aprint.format('Band: ', f.format(str(wband), func)))
    if 'wprior' in function:
        func = '{} priority'.format(user_prior)
        wprior = calc_weight.weight_userpriority(user_prior=user_prior)
        print(aprint.format('user priority: ', f.format(str(wprior), func)))
    if 'wstatus' in function:
        func = 'Partially complete: prog={}, obs={}'.format(prstatus, obstatus>0)
        wstatus = calc_weight.weight_status(prstatus=prstatus, obstatus=obstatus)
        print(aprint.format('status: ',f.format(str(wstatus),func)))
    # Not yet added
    # if 'wbal' in function:
    #     func = '{} partner balance'.format(bal)
    #     f = '{} ({})'
    #     wprior = calc_weight.weight_userpriority(user_prior=user_prior)
    #     print(aprint.format('wstatus: ', f.format(str(wprior), func)))

    if 'wha' in function:
        wha = calc_weight.weight_ha(latitude=latitude, dec=dec, HA=HA)
    else:
        wha = np.ones(len(HA))
    if 'cmatch' in function:
        cmatch = calc_weight.weight_cond_match(cond=cond, acond=acond, negHA=min(HA) < 0.*u.hourangle)
    else:
        cmatch = np.ones(len(HA))
    if 'wam' in function:
        wam = calc_weight.weight_am(AM=AM, HA=HA, elev=elev)
    else:
        wam = np.ones(len(HA))
    if 'wwind' in function:
        wwind = calc_weight.weight_wind(AZ=AZ)
    else:
        wwind = np.ones(len(AZ))


    timedelt = (timeinfo.utc - timeinfo.midnight).value * 24

    plt.subplot(221)
    plt.plot(timedelt, wha, color='black', label='wha')
    plt.title('Hour angle')
    plt.xlabel(r'$\Delta t_{mid}$ (hrs)')
    plt.ylabel('Weight')

    plt.subplot(222)
    plt.plot(timedelt, cmatch, color='black', label='cmatch')
    plt.title('Conditions')
    plt.ylim(-0.1, 1.1)
    plt.xlabel(r'$\Delta t_{mid}$ (hrs)')
    plt.ylabel('Weight')

    plt.subplot(223)
    plt.plot(timedelt, wam, color='black', label='wam')
    plt.title('Air mass')
    plt.ylim(-0.1, 1.1)
    plt.xlabel(r'$\Delta t_{mid}$ (hrs)')
    plt.ylabel('Weight')

    plt.subplot(224)
    plt.plot(timedelt, wwind, color='black', label='wwind')
    plt.title('Wind')
    plt.ylim(-0.1, 1.1)
    plt.xlabel(r'$\Delta t_{mid}$ (hrs)')
    plt.ylabel('Weight')


    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
    plt.tight_layout()
    plt.show(block=False)
    input('\n Press enter to close window...')
    plt.close()
    plt.clf()
    return

def getobsindex(id, obs):
    ii = np.where(obs.obs_id == id)[0][:]
    if len(ii)==1:
        return ii[0]
    elif len(ii)==0:
        print(' \'{}\' not found.'.format(id))
        raise ValueError
    elif len(ii)>1:
        print('Found '+str(len[ii])+' observations with identifier '+id)
        raise ValueError
    else:
        print('Something went wrong.')
        raise ValueError

def listobs(list_ids, obs_ids, prog_refs, groups, sprint):
    print('')
    print(sprint.format('Observation', 'Program', 'Group'))
    print(sprint.format('-----------', '-------', '-----'))
    for obs_id in list_ids:
        i = np.where(obs_ids == obs_id)[0][0]
        print(sprint.format(obs_ids[i], prog_refs[i], groups[i]))
    return

def weightplotmode(site, timeinfo, suninfo, mooninfo, targetinfo, i_obs, obs, plan, acond):

    aprint = '\t{0:<35s}{1}'  # print two strings
    fprint = '\t{0:5s}{1:40s}{2:10}'  # format options menu
    sprint = '\t{0:25s}{1:23s}{2:25s}'

    date = str(timeinfo.utc[0].iso)[0:10]
    function = calc_weight.obsweight(dec='', AM='', HA='', AZ='', band='', user_prior='', prstatus='', latitude='',
                                        cond='', acond='', obstatus='', elev='', getfunc=True)

    print('\n\n\t---------- Weight function plotting mode ----------\n')
    print(aprint.format('Plan date:', date))

    # listobs(list_ids=plan.obs_id, obs_ids=obs.obs_id, prog_refs=obs.prog_ref, groups=obs.group, sprint=sprint)

    if ttimer: timer = t.time()
    wra = calc_weight.weight_ra(ras=obs.ra, tot_time=obs.tot_time, obs_time=obs.obs_time)
    if ttimer: print('\n\tTimer weight_ra = ', t.time() - timer)

    if ttimer: timer = t.time()
    # ttime = np.round((obs.tot_time[i_obs] - obs.obs_time[i_obs]) * 10) / 10  # remaining time in observations
    prstatus = calc_weight.getprstatus(prog_ref=obs.prog_ref, obs_time=obs.obs_time)
    if ttimer: print('\n\tTimer prstatus = ', t.time() - timer)

    pltcomp = True
    while True:

        # =============================================== Menu =========================================================
        print(aprint.format('Weight function:', function))
        print('\n\tOptions:')
        print('\t--------')
        print(fprint.format('1.', 'See plan','-'))
        print(fprint.format('2.', 'See plan observations','-'))
        print(fprint.format('3.', 'See plan candidates','-'))
        print(fprint.format('4.', 'See all observations','-'))
        print(fprint.format('5.', 'Plot weighting function components',str(pltcomp)))
        print()

        userinput = input(' Select option from list or input observation identifier: ')

        if userinput == '1':
            [print(line) for line in
             printPlanTable(plan=plan, i_obs=i_obs, obs=obs, timeinfo=timeinfo, targetinfo=targetinfo)]
        elif userinput == '2':
            listobs(list_ids=plan.obs_id, obs_ids=obs.obs_id, prog_refs=obs.prog_ref, groups=obs.group, sprint=sprint)
        elif userinput == '3':
            list_ids = obs.obs_id[i_obs]
            listobs(list_ids=list_ids, obs_ids=obs.obs_id, prog_refs=obs.prog_ref, groups=obs.group, sprint=sprint)
        elif userinput == '4':
            list_ids = obs.obs_id
            listobs(list_ids=list_ids, obs_ids=obs.obs_id, prog_refs=obs.prog_ref, groups=obs.group, sprint=sprint)
        elif userinput == '5':
            if pltcomp:
                pltcomp = False
            else:
                pltcomp = True
        elif userinput.lower() == 'quit' or userinput.lower() == 'exit' or userinput.lower() == 'q':
            break
        else:
            try:
                i = getobsindex(id=userinput, obs=obs)
            except ValueError:
                continue

            if ttimer: timer = t.time()
            target = TargetInfo(ra=obs.ra[i], dec=obs.dec[i], name=obs.obs_id[i], site=site, utc_times=timeinfo.utc)
            if ttimer: print('\n\tTimer target = ', t.time() - timer)

            if ttimer: timer = t.time()
            mdist = gcirc(mooninfo.ra, mooninfo.dec, target.ra, target.dec)
            if ttimer: print('\n\tTimer mdist = ', t.time() - timer)

            if ttimer: timer = t.time()
            vsb = sb(mpa=mooninfo.phase, mdist=mdist, mZD=mooninfo.ZD, ZD=target.ZD, sZD=suninfo.ZD, cc=acond[1])
            if ttimer: print('\n\tTimer sb = ', t.time() - timer)

            if ttimer: timer = t.time()
            sbcond = calc_weight.vsb_to_sbcond(vsb=vsb)
            if ttimer: print('\n\tTimer sb = ', t.time() - timer)

            if ttimer: timer = t.time()
            weightfunc = calc_weight.obsweight(dec=target.dec,
                               AM=target.AM,
                               HA=target.HA,
                               AZ=target.AZ,
                               band=obs.band[i],
                               user_prior=obs.user_prior[i],
                               prstatus=prstatus[i],
                               latitude=site.location.lat,
                               cond={'iq': obs.iq[i], 'cc': obs.cc[i], 'bg': obs.bg[i], 'wv': obs.wv[i]},
                               acond={'iq': acond[0], 'cc': acond[1], 'bg': sbcond, 'wv': acond[3]},
                               obstatus=obs.obs_time[i],
                               wra=wra[i],
                               elev=obs.elev_const[i])
            if ttimer: print('\n\tTimer obs_weight = ', t.time() - timer)
            plotweightfunc(id=userinput, timeinfo=timeinfo, weights=weightfunc)

            if pltcomp:
                try:
                    if ttimer: timer = t.time()
                    plotcomponents(id=userinput,
                                   timeinfo=timeinfo,
                                   function=function,
                                   dec=target.dec,
                                   AM=target.AM,
                                   HA=target.HA,
                                   AZ=target.AZ,
                                   band=obs.band[i],
                                   user_prior=obs.user_prior[i],
                                   prstatus=prstatus[i],
                                   latitude=site.location.lat,
                                   cond={'iq': obs.iq[i], 'cc': obs.cc[i], 'bg': obs.bg[i], 'wv': obs.wv[i]},
                                   acond={'iq': acond[0], 'cc': acond[1], 'bg': sbcond, 'wv': acond[3]},
                                   obstatus=obs.obs_time[i],
                                   wra=wra[i],
                                   elev=obs.elev_const[i])
                    if ttimer: print('\n\tTimer obs_weight = ', t.time() - timer)
                except AttributeError:
                    pass


    return