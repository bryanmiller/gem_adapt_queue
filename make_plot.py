# Matt Bonnyman 19 July 2018

import numpy as np
import astropy.units as u
from astropy.time import Time
import matplotlib.pyplot as plt

import weights as weights
from hms_to_hr import hms_to_hr
from intervals import intervals


def airmass(date, plan, obs_id, am, local, moonam=None, obslabels=False, description='', istart=0, iend=-1,
            savefig=False):
    """
    Create air mass plot of current plan.

    Parameters
    ----------
    date : string
        date of current plan

    plan : numpy integer array or None
        indices of targets from 'targets' along time grid.

    obs_id : np.array of string
        observation identifiers of targets.

    am : np.arrays of float
        target airmasses along time grid.

    local : np.array of string
        local times along time grid in format accepted by 'astropy.time.Time'.

    moonam : array of floats, optional
        moon air mass values. Default = None

    obslabels : boolean, optional
        annotate plot with observation identifiers. Default = False

    description : string, optional
        descriptor appended to plot title (eg. '2018-08-12 schedule"description"'). Default = ''.

    savefig : boolean, optional
        save figure as png. Default = False.

    istart : integer, optional
        starting index of plan window to include in plot. Default = 0

    iend : integer, optional
        ending index of plan window to include in plot. Default = -1
    """

    verbose = False

    thk = 4
    thn = 1

    if iend == -1:
        iend = len(plan) - 1

    pplan = plan[istart:iend + 1]

    hours = _hour_from_midnight(local)

    if moonam is not None:
        plt.plot(hours.value, moonam, linestyle=':', label='Moon', linewidth=thn, color='grey', markersize=0)

    index = np.unique(pplan)  # indices of obs in plan

    for ind in index:
        if ind >= 0:
            plt.plot(hours.value, am[ind], linestyle='-', linewidth=thn, color='black', markersize=0)

    for ind in index:
        if ind >= 0:
            ii = np.where(pplan == ind)[0][:]
            intvl = intervals(ii)
            iint = np.unique(intvl)

            if verbose:
                print('ii', ii)
                print('intvl', intvl)
                print('iint', iint)

            for i in iint:
                jj = np.where(intvl == i)[0][:]

                if verbose:
                    print('jj', jj)

                plt.plot(hours[ii[jj[0]]:ii[jj[-1]]+1].value, am[ind][ii[jj[0]]:ii[jj[-1]]+1], linestyle='-',
                         linewidth=thk, markersize=0)  # , label=targets['id'][ind])

                if obslabels:
                    arrowprops = dict(arrowstyle="<-", connectionstyle="arc3")
                    plt.annotate(obs_id[ind][-11:], xy=(hours[ii[jj[0]]].value, am[ind][ii[jj[0]]]),
                                 xytext=(hours[ii[jj[0]]].value, am[ind][ii[jj[0]]] + 0.2), arrowprops=arrowprops)

    plt.title(date+' schedule'+description)
    plt.ylim(2.1, 0.9)
    plt.xlim(hours[0].value, hours[-1].value)
    plt.ylabel('Airmass')
    plt.xlabel(r'$\Delta t$ from local midnight (hrs)')
    plt.legend(loc=8, ncol=4, fontsize=8, markerscale=0.5)
    plt.tight_layout()

    if savefig:
        plt.savefig('amplot'+date+'.png')

    plt.show(block=True)
    # input(' Press enter to close plot window...')
    plt.close()
    plt.clf()

    return


def altaz(date, plan, obs_id, az, zd, moonaz=None, moonzd=None, obslabels=False, description='', savefig=False,
          istart=0, iend=-1):
    """
    Create air mass plot of current plan.

    Parameters
    ----------
    date : string
        date of current schedule

    plan : numpy integer array or None
        indices of targets from 'targets' along time grid.

    obs_id : np.array of string
        observation identifiers of targets.

    az : arrays of 'astropy.units' degrees
        target azimuth angles along time grid

    zd : arrays of 'astropy.units' degrees
        target zenith distance angles along time grid

    moonaz : array of 'astropy.units' degrees, optional
        moon azimuth angles along time grid. Default = None

    moonzd : arrays of 'astropy.units' degrees, optional
        moon zenith distance angles along time grid. Default = None

    obslabels : boolean, optional
        annotate plot with observation identifiers. Default = False

    description : string, optional
        descriptor appended to plot title (eg. '2018-08-12 schedule"description"'). Default = ''.

    savefig : boolean, optional
        save figure as png. Default = False.

    istart : integer, optional
        starting index of plan window to include in plot. Default = 0

    iend : integer, optional
        ending index of plan window to include in plot. Default = -1
    """

    verbose = False

    # Line thicknesses
    thk = 4
    thn = 1

    if iend == -1:
        iend = len(plan) - 1

    pplan = plan[istart:iend]

    ax = plt.subplot(111, polar=True)

    if moonaz is not None and moonzd is not None:
        kk = np.where(moonzd <= 90*u.deg)
        ax.plot(moonaz[kk].to(u.rad), moonzd[kk], linestyle=':', label='Moon', linewidth=thn,
                color='grey', markersize=0)

    index = np.unique(pplan)  # indices of obs in plan

    for ind in index:
        if ind >= 0:
            ii = np.where(zd[ind] <= 90*u.deg)
            ax.plot(az[ind][ii].to(u.rad), zd[ind][ii], linestyle='-', linewidth=thn,
                    color='grey', markersize=0)

    for ind in index:
        if ind >= 0:
            ii = np.where(pplan == ind)[0][:]
            intvl = intervals(ii)
            iint = np.unique(intvl)

            if verbose:
                print('ii', ii)
                print('intvl', intvl)
                print('iint', iint)

            for i in iint:
                jj = np.where(intvl == i)[0][:]

                if verbose:
                    print('jj', jj)

                ax.plot(az[ind][ii[jj[0]]:ii[jj[-1]]+1].to(u.rad),
                        zd[ind][ii[jj[0]]:ii[jj[-1]]+1], linestyle='-', linewidth=thk)

                if obslabels:
                    arrowprops = dict(arrowstyle="<-", connectionstyle="arc3")
                    ax.annotate(obs_id[ind][-11:], (az[ind][ii[jj[0]]], zd[ind][ii[jj[0]]]), arrowprops=arrowprops)

                if verbose:
                    print(zd[ind][ii[jj[0]]:ii[jj[-1]]+1])

    ax.set_title(str(date)+' schedule'+description)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rlim(0, 90)
    plt.legend(loc=8, ncol=4, fontsize=8, markerscale=0.5)

    if savefig:
        plt.savefig('altazplot'+date+'.png')

    plt.tight_layout()
    plt.show(block=True)
    # input(' Press enter to close plot window...')
    plt.close()
    plt.clf()

    return


def vsb(vsb, local_time, date, obs_id, savefig=False):
    """
    Plot V sky brightness
    :param vsb:
    :param local_time:
    :return:
    """
    plt.title(obs_id + ' on ' + date + ': sky brightness')

    hours = _hour_from_midnight(local_time)

    plt.plot(hours, vsb, label='sky brightness', alpha=0.7, linestyle='--')

    # plt.legend(loc='upper right', fontsize=8, markerscale=0.5)
    plt.ylabel('V sky brightness')
    plt.xlabel(r'$\Delta t_{mid}$ (hrs)')
    plt.gca().invert_yaxis()

    if savefig:
        plt.savefig('vsbplot' + date + '.png')

    plt.tight_layout()
    plt.show(block=True)
    # input(' Press enter to close plot window...')
    plt.close()
    plt.clf()
    return

def skyconditions(skycond, local_time, date, bg=None, savefig=False, verbose = True):
    """
    Plot sky conditions percentiles

    Parameters
    ----------
    skycond : 'astropy.table.Table'
        Sky conditions table with columns 'iq', 'cc', 'wv'.

    local_time : array of strings or 'astropy.time.core.Time' array
        Time grid of local times in format accepted by 'astropy.time.Time'

    date : string
        Plot date

    bg : array of floats (optional)
        Sky background conditions for target
    """

    plt.title(date + ' sky conditions')

    hours = _hour_from_midnight(local_time)

    if verbose:
        print(date)
        print(Time(local_time[-1].iso[0:10]))
        print(hours)
        print(skycond['iq'].quantity)
        print(skycond['cc'].quantity)
        print(skycond['wv'].quantity)
        print(bg)

    plt.plot(hours, skycond['iq'].quantity, label='image quality', alpha=0.7, linestyle='--')
    plt.plot(hours, skycond['cc'].quantity, label='cloud condition', alpha=0.7, linestyle='--')
    plt.plot(hours, skycond['wv'].quantity, label='water vapor', alpha=0.7, linestyle='--')
    if bg is not None:
        plt.plot(hours, bg, label='sky background', alpha=0.7, linestyle='--')
    plt.legend(loc='upper right', fontsize=8, markerscale=0.5)
    plt.ylabel('Decimal percentile')
    plt.xlabel(r'$\Delta t_{mid}$ (hrs)')

    if savefig:
        plt.savefig('condplot' + date + '.png')

    plt.tight_layout()
    plt.show(block=True)
    # input(' Press enter to close plot window...')
    plt.close()
    plt.clf()
    return


def windconditions(wind, local_time, date, savefig=False):
    """
    Plot sky conditions percentiles

    Parameters
    ----------
    wind : 'astropy.table.Table'
        Wind conditions table with columns dir, vel.

    local_time : array of strings
        time grid of local times in format accepted by 'astropy.time.Time'

    date : string
        Date of time window
    """

    verbose = False

    plt.title(date + ' wind conditions')

    hours = _hour_from_midnight(local_time)

    if verbose:
        print(date)
        print(Time(local_time[-1][0:10]))
        print(wind['vel'].quantity.value)
        print(wind['dir'].quantity.value)

    # Add 90 degrees so that 0 degrees points North.
    # Invert x-component so that degrees increase in clockwise direction.
    r = 0.5
    xcomp = r * np.cos(wind['dir'].quantity + 90*u.deg) * -1
    ycomp = r * np.sin(wind['dir'].quantity + 90*u.deg)

    plt.quiver(hours.value, wind['vel'].quantity.value, xcomp, ycomp)
    plt.xlabel(r'$\Delta t_{mid}$ (hrs)')
    plt.ylabel(r'$velocity$ (m/s)')
    plt.xlim(hours[0].value, hours[-1].value)
    plt.ylim(-0.5, np.max(wind['vel'].quantity.value) + 2)

    if savefig:
        plt.savefig('windplot'+date+'.png')

    plt.tight_layout()
    plt.show(block=True)
    # input(' Press enter to close plot window...')
    plt.close()
    plt.clf()
    return


def weightfunction(obs_id, local_time, date, weight):
    """
    Show plot of target weighting function with respect to time.

    Parameters
    ----------
    obs_id : string
        Unique observation identifier

    local_time : array of strings or 'astropy.time.core.Time' array
        Time grid of local times in format accepted by 'astropy.time.Time'

    date : string
        Plot date

    weight : np.array of floats
        weighting function values. Must have same length as columns in timetable.
    """
    hours = _hour_from_midnight(Time(local_time))

    #  Get non-zero parts of weight function
    ii = np.where(weight > 0)[0][:]
    if len(ii) != 0:
        i_intervals = intervals(ii)
        wmin = np.min(weight[ii])
        wmax = np.max(weight[ii])
    else:
        print(' No non-zero weights.')
        return

    #  Plot non-zero portions of weight function
    n_int = np.unique(i_intervals)
    for n in n_int:
        iint = ii[np.where(i_intervals == n)[0][:]]
        plt.plot(hours[iint].value, weight[iint], color='black')

    plt.title(obs_id + ' on ' + date)
    plt.xlim(hours[0].value, hours[-1].value)
    plt.ylim(wmin, wmax)
    plt.ylabel('Weight')
    plt.xlabel(r'$\Delta t$ from local midnight (hrs)')
    plt.tight_layout()
    plt.show(block=True)
    # input('\n Press enter to close window...')
    plt.close()
    plt.clf()
    return


def weightcomponents(obs_id, ra, dec, iq, cc, bg, wv, elev_const, i_wins, band, user_prior, AM, HA, AZ, latitude,
                     prog_comp, obs_comp, skyiq, skycc, skybg, skywv, winddir, windvel, wra, localtimes):
    """
    Calculate observation weights.

    Parameters
    ----------
    obs_id : string
        observation identifier (only needed if printing output)

    ra : 'astropy.units' degrees
        observation right ascension

    dec : 'astropy.units' degrees
        observation declination

    iq : float
        observation image quality constraint percentile

    cc : float
        observation cloud condition constraint percentile

    bg : float
        observation sky background constraint percentile

    wv : float
        observation water vapour constraint percentile

    elev_const : dictionary
        observation elevation constraint (type, min, max).

        Example
        -------
        elev_const = {type='Hour Angle', min='-2.00', max='2.00'}

    i_wins : list of integer pair(s)
        indices of observation time window(s) along time grid.

        Example
        -------
        an observation with two time windows would look something like...
        i_wins = [
                  [0,80],
                  [110, 130],
                 ]

    band : int
        observation ranking band (1, 2, 3, 4)

    user_prior : string
        observation user priority ('Low', 'Medium', 'High', 'Target of Opportunity')

    obs_comp : float
        fraction of observation completed

    AM : np.array of floats
        target airmasses along time grid

    HA : np.array of 'astropy.units' hourangles
        target hour angles along time grid

    AZ : np.array of 'astropy.units' radians
        target azimuth angles along time grid

    skyiq : np.array of float
        sky image quality percentile along time grid

    skycc : np.array of float
        sky cloud condition percentile along time grid

    skywv : np.array of float
        sky water vapour percentile along time grid

    skybg : array of floats
        target sky background percentiles along time grid

    latitude : '~astropy.coordinates.angles.Latitude' or '~astropy.unit.Quantity'
        observatory latitude

    prog_comp : float
        Completion fraction of program

    winddir : np.array of 'astropy.units' degrees
        wind direction along time grid

    windvel : np.array of 'astropy.units' kilometers/hour
        wind velocity along time grid

    wra : np.ndarray of floats
        RA time distribution weighting factor
    """

    print('\n\t{} weights\n\t-------------------------'.format(obs_id))

    aprint = '\t{0:<25s}{1}'  # print two strings
    f = '{} ({})'

    func = 'iq={}, cc={}, bg={}, wv={}'.format(iq, cc, bg, wv)
    twcond = weights.total_cond(iq=iq, cc=cc, bg=bg, wv=wv)
    print(aprint.format('Total cond: ', f.format(str(twcond), func)))

    print(aprint.format('RA: ', wra))

    func = 'Band {}'.format(band)
    wband = weights.rankingband(band=band)
    print(aprint.format('Band: ', f.format(str(wband), func)))

    func = '{} priority'.format(user_prior)
    wprior = weights.userpriority(user_prior=user_prior)
    print(aprint.format('User priority: ', f.format(str(wprior), func)))

    func = 'Partially complete: prog={}, obs={}'.format(prog_comp > 0, obs_comp > 0)
    wstatus = weights.status(prog_comp=prog_comp, obs_comp=obs_comp)
    print(aprint.format('Status: ', f.format(str(wstatus), func)))

    # Not yet added
    # if 'wbal' in function:
    #     func = '{} partner balance'.format(bal)
    #     f = '{} ({})'
    #     wprior = calc_weight.weight_userpriority(user_prior=user_prior)
    #     print(aprint.format('wstatus: ', f.format(str(wprior), func)))

    wha = weights.hourangle(latitude=latitude, dec=dec, ha=HA)

    cmatch = weights.cond_match(iq=iq, cc=cc, bg=bg, wv=wv,
                                skyiq=skyiq, skycc=skycc, skywv=skywv, skybg=skybg, negha=min(HA) < 0. * u.hourangle,
                                user_prior=user_prior)

    wam = weights.airmass(am=AM, ha=HA, elev=elev_const)

    wwins = weights.time_wins(grid_size=len(skyiq), i_wins=i_wins)

    wwind = weights.windconditions(dir=winddir, vel=windvel, az=AZ)

    hours = _hour_from_midnight(local_time=localtimes)

    plt.subplot(221)
    plt.plot(hours, wha, color='black', label='wha')
    plt.title('Hour angle')
    plt.xlabel(r'$\Delta t_{mid}$ (hrs)')
    plt.ylabel('Weight')

    plt.subplot(222)
    plt.plot(hours, cmatch, color='black', label='cmatch')
    plt.title('Conditions constraints')
    plt.ylim(-0.1, 1.1)
    plt.xlabel(r'$\Delta t_{mid}$ (hrs)')
    plt.ylabel('Weight')

    plt.subplot(223)
    plt.plot(hours, wam, color='black', label='wam')
    plt.title('Air mass')
    plt.ylim(-0.1, 1.1)
    plt.xlabel(r'$\Delta t_{mid}$ (hrs)')
    plt.ylabel('Weight')

    plt.subplot(224)
    plt.plot(hours, wwind, color='black', label='wwind')
    plt.title('Wind')
    plt.ylim(-0.1, 1.1)
    plt.xlabel(r'$\Delta t_{mid}$ (hrs)')
    plt.ylabel('Weight')

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
    plt.tight_layout()
    plt.show(block=True)
    # input('\n Press enter to close window...')
    plt.close()
    plt.clf()

    plt.subplot(221)
    plt.plot(hours, wwins, color='black', label='wwins')
    plt.title('Time windows')
    plt.ylim(-0.1, 1.1)
    plt.xlabel(r'$\Delta t_{mid}$ (hrs)')
    plt.ylabel('Weight')

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
    plt.tight_layout()
    plt.show(block=True)
    # input('\n Press enter to close window...')
    plt.close()
    plt.clf()

    # All in one
    plt.plot(hours, wha, label='Hour Angle')
    plt.plot(hours, cmatch, label='Conditions')
    plt.plot(hours, wam, label='Airmass')
    plt.plot(hours, wwind, label='Wind')
    plt.plot(hours, wwins, label='Timing windows')
    plt.xlabel(r'$\Delta t_{mid}$ (hrs)')
    plt.ylabel('Weight')
    plt.legend()
    plt.show(block=True)
    # input('\n Press enter to close window...')
    plt.close()
    plt.clf()

    return


def _hour_from_midnight(local_time):
    """
    Get array of the time difference from local midnight for each time in array 'localtimes'.

    Parameters
    ----------
    local_time : array of strings
        local time in iso format (i.e. 'YYYY-MM-DD hh:mm:ss.sss')

    Returns
    -------
    array of floats
    """
    hr = hms_to_hr(local_time[0].iso[11:]) * u.h
    if hr < 12. * u.h:
        midnight = Time(local_time[0].iso[0:10])
    else:
        midnight = Time(local_time[0].iso[0:10]) + 1. * u.d
    return (Time(local_time) - midnight).to(u.h)


def test_hour_from_midnight():
    print(' Test _hour_from_midnight()...')
    print('\n Case that time array begins before midnight:')
    print(' Input times = [\'2018-07-25 19:00:40.529\', \'2018-07-26 06:36:40.529\']')
    print(' Difference from midnight =',
          _hour_from_midnight(Time(['2018-07-25 19:00:40.529', '2018-07-26 06:36:40.529'])))

    print('\n Case that time array begins after midnight:')
    print(' Input times = [\'2018-07-26 2:00:40.529\', \'2018-07-26 06:36:40.529\']')
    print(' Difference from midnight =',
          _hour_from_midnight(Time(['2018-07-26 2:00:40.529', '2018-07-26 06:36:40.529'])))

    assert _hour_from_midnight(Time(['2018-07-25 19:00:40.529', '2018-07-26 06:36:40.529'])).value.all() \
           == (np.array([-4.98874194, 6.61125806])*u.h).value.all()
    assert _hour_from_midnight(Time(['2018-07-26 2:00:40.529', '2018-07-26 06:36:40.529'])).value.all() \
           == (np.array([ 2.01125806,  6.61125806])*u.h).value.all()

    print(' Test successful!')

if __name__ == '__main__':
    test_hour_from_midnight()
