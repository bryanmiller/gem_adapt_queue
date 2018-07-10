import numpy as np
import astropy.units as u
from calc_ZDHA import calc_ZDHA
from astropy.coordinates import (EarthLocation, get_sun, get_moon)
from astroplan import Observer
from astropy.time import Time

def airmass(ZD):
    """
    Calculate airmasses

    Parameter
    ---------
    ZD : array of 'astropy.units.Quantity'
        zenith distance angle

    Return
    ---------
    AM : array of floats
        airmass
    """
    AM = np.full(len(ZD),20.)
    ii = np.where(ZD < 87.*u.deg)[0][:]
    sec_z = 1. / np.cos(ZD[ii])
    AM[ii] = sec_z - 0.0018167 * ( sec_z - 1 ) - 0.002875 * ( sec_z - 1 )**2 - 0.0008083 * ( sec_z - 1 )**3 #compute moon AMs
    return AM

def sun_horizon(site):
    """
    Calculate angle between sun and horizon at sunset at a location.

    Parameter
    ---------
    site : '~astroplan.Observer'
        class with observer's longitude, latitude, elevation.

    Return
    --------
    '~astropy.units.Quantity'
        degree angle of sun at sunset
    """
    sun_horiz = -.83 * u.degree
    equat_radius = 6378137. * u.m
    return sun_horiz - np.sqrt(2. * site.location.height / equat_radius) * (180. / np.pi) * u.degree

def _checkinput_site(site,latitude,longitude,elevation):
    """
    Check that observer location is 'astroplan.Observer' class or
    initialize and return 'astroplan.Observer' if longitude, latitude, elevation provided.
    """
    if site is None and (latitude is not None and
                         longitude is not None):
        return Observer(location=EarthLocation.from_geodetic(longitude, latitude,
                                                             elevation))
    elif isinstance(site, Observer):
        return site
    else:
        raise TypeError('Site location must be specified with either'
                        '(1) an instance of astroplan.Observer'
                        '(2) astropy.coordinates.EarthLocation'
                        '(3) latitude,longitude in degrees as '
                        'accepted by astropy.coordinates.Latitude and '
                        'astropy.coordinates.Latitude.')

def _checkinput_lst(variable,varname):
    if isinstance(variable, u.quantity.Quantity):
        return
    else:
        raise TypeError('Input sidereal time \''+varname+'\' must be specified with'
                        'a list of instances of'
                        'astropy.coordinates.angles.Longitude.')

def _checkinput_time(variable,varname):
    if isinstance(variable, Time):
        return variable
    elif isinstance(variable, str):
        return Time(variable)
    else:
        raise TypeError('Input time \''+varname+'\' must be specified with either'
                        '(1) an instance of astropy.time.core.Time'
                        '(2) string as accepted by astropy.time.Time.')

def _checkinput_coord(ra,dec):
    if isinstance(ra,u.quantity.Quantity) and isinstance(dec,u.quantity.Quantity):
        return ra,dec
    else:
        raise TypeError('Coordinates must be specified as...'
                  '\nra: an instance of astropy.coordinates.angles.Longitude'
                  '\ndec: an instance of astropy.coordinates.angles.Latitude.')

class TimeInfo(object):

    """
    A container class for time interval information
    throughout a given night (or similar time window).
    """

    @u.quantity_input(elevation=u.m,utc_to_local=u.h)
    def __init__(self,site=None,starttime=None,endtime=None,dt=0.1,
                 latitude=None,longitude=None,elevation=0.0*u.m,utc_to_local=0.*u.h):

        """
        Parameters
        ----------
        site : astroplan.Observer (or astropy.coordinates.Earthlocation)

        starttime : string, astropy.time.core.Time object
            Time must be in utc and ~18:00 local time on observation night
            in order to obtain correct twilight times.  If string, must
            be accepted by astropy.time.Time().

        dt (optional): float
            size of time intervals in hours (Default is 0.1).

        latitude (optional) : float, str, 'astropy.units.Quantity'
            The latitude of the observing location (if site not provided).

        longitude (optional) : float, str, 'astropy.units.Quantity'
            The longitude of the observing location (if site not provided).

        elevation (optional) : 'astropy.units.Quantity' (default = 0 meters)
            The elevation of the observing location (if site not provided).


        Returns
        ---------
        '~gemini_classes.TimeInfo'
            TimeInfo Object

        Parameters
        ----------

        dt : 'astropy.units.quantity.Quantity'
            size of time interval (defaults to 0.1 hours).

        nt : int
            number of time intervals in observation window

        utc : array of 'astropy.time.core.Time' objects
            utc times at time intervals

        lst : array of 'astropy.units.quantity.Quantity's
            lst hourangle times at intervals

        evening_twilight : astropy.time.Time
            utc evening nautical twilight

        morning_twilight : astropy.time.Time
            utc morning nautical twilight

        start : astropy.time.Time
            start of observation window
            (defaults to evening_twilight and
            limited to be within twilights)

        end : astropy.time.Time
            end of observation window
            (defaults to morning_twilight and
            limited to be within twilights)

        midnight : astropy.time.Time
            utc time at solar midnight

        night_length : 'astropy.units.quantity.Quantity'
            hours between twilights
        """

        timer = False
        if timer:
            import time as t
            timerstart = t.time()  # runtime clock

        self.site = _checkinput_site(site=site, latitude=latitude, longitude=longitude, elevation=elevation)

        start_time = _checkinput_time(variable=starttime,varname='starttime')

        # nautical twilight times (sun at -12 degrees)
        evening_twilight = site.twilight_evening_nautical(start_time, which='nearest')
        morning_twilight = site.twilight_morning_nautical(start_time, which='next')

        # set time window boundaries
        if evening_twilight-start_time > 0.*u.d:
            start_time = evening_twilight
        if endtime is not None:
            end_time = _checkinput_time(variable=endtime, varname='endtime')
            if end_time - morning_twilight > 0. * u.d:
                end_time = morning_twilight
        else:
            end_time = morning_twilight


        # hours between start and end times
        night_length = (end_time - start_time) * 24. / u.d  # get number of hours between twilights
        nt = int(round(float(night_length) / dt, 1)) + 1  # number of time intervals between twilights
        dt = dt * u.h  # apply units to dt
        stepnum = np.arange(0, nt)  # assign integer to time intervals

        utc_times = (stepnum * dt + start_time)  # array of utc astropy.time.core.Time objects
        lst_times = site.local_sidereal_time(utc_times)  # array of lst hourangles as astropy.units.Quantity
        local_times = utc_times + utc_to_local

        self.dt = dt
        self.nt = nt
        self.utc_to_local = utc_to_local
        self.night_length = night_length*u.h
        self.midnight = site.midnight(start_time, which='nearest')
        self.utc = utc_times
        self.lst = lst_times
        self.local = local_times
        self.evening_twilight = evening_twilight
        self.morning_twilight = morning_twilight
        self.start = start_time
        self.end = end_time

        if timer: print('\n\tInitialize TimeInfo: ', t.time() - timerstart)  # runtime clock

    def __repr__(self):

        """
        String representation of the gemini_classes.TimeInfo object.

        Example
        ---------
        >>> from gemini_classes import TimeInfo
        >>> import astropy
        >>> import astroplan
        >>> site_gs = astroplan.Observer.at_site('gemini_south')
        >>> local_1800 = astropy.time.Time('2018-01-01 22:00:00') #18:00 local time
        >>> times = TimeInfo(site=site_gs,utc_time=local_1800,dt=0.5)
        >>> print(times)
        <TimeInfo: dt='0.5 h',
            nt='16',
            utc='[ 2458120.53394173  2458120.55477506  2458120.5756084   2458120.59644173
          2458120.61727506  2458120.6381084   2458120.65894173  2458120.67977506
          2458120.7006084   2458120.72144173  2458120.74227506  2458120.7631084
          2458120.78394173  2458120.80477506  2458120.8256084   2458120.84644173]',
            lst='[  2.8732448    3.37461381   3.87598283   4.37735184   4.87872085
           5.38008986   5.88145887   6.38282788   6.88419689   7.3855659
           7.88693491   8.38830392   8.88967293   9.39104194   9.89241095
          10.39377996] hourangle',
            evening_twilight='2458120.5339417304',
            morning_twilight='2458120.864519293',
            night_length='7.93386150151491'>
        """

        class_name = self.__class__.__name__
        attr_names = ['site','dt','nt','start','end','evening_twilight',
                      'morning_twilight','midnight','night_length','utc','lst']
        attr_values = [getattr(self, attr) for attr in attr_names]
        attributes_strings = []
        for name, value in zip(attr_names, attr_values):
            if value is not None:
                value = "'{}'".format(value)
                attributes_strings.append("{}={}".format(name, value))
        return "<{}: {}>".format(class_name, ",\n    ".join(attributes_strings))


    def table(self,showall=False):

        """
        Table representation of gemini_classes.TimeInfo object
        """

        table = []

        sattr = '\t\t{0:<25s}{1}'  # print string and string
        fattr = '\t\t{0:<25s}{1:.2f}'  # print string and float

        class_name = self.__class__.__name__
        attr_names = ['dt', 'nt', 'evening_twilight', 'morning_twilight',
                      'start', 'end', 'midnight', 'night_length']
        table.append(str('\n\t'+class_name+':'))
        attr_values = [getattr(self, attr) for attr in attr_names]
        for name, value in zip(attr_names, attr_values):
            if value is not None:
                if name=='evening_twilight' or name=='morning_twilight'\
                    or name=='start' or name=='end' or name=='midnight':
                    table.append(str(sattr.format(name, value.iso)))
                elif name=='night_length':
                    table.append(str(fattr.format(name, value)))
                else:
                    table.append(str(sattr.format(name, value)))

        if showall:
            fheader = '\t\t{0:<25.22s}{1:<25.20s}'
            fvals = '\t\t{0:<25.22}{1:<7.5}'
            attr_names = ['utc', 'lst']
            table.append(str(''))
            table.append(str(fheader.format('utc','lst')))
            table.append(str(fheader.format('--------','--------')))
            attr_values = [getattr(self, attr) for attr in attr_names]
            for i in range(0, len(attr_values[0])):
                table.append(str(fvals.format(attr_values[0][i].iso, attr_values[1][i])))

        return table


class SunInfo(object):

    """
    A container class for storing all required sun parameters 
    for a given night (or similar time window).
    """

    @u.quantity_input(elevation=u.m)
    def __init__(self,site=None,utc_times=None,
                 latitude=None,longitude=None, elevation=0.0 * u.m):
        """
        Input
        ----------
        site : astroplan.Observer (or astropy.coordinates.Earthlocation)
            Observer location information

        utc_times : array of astropy.time.core.Time objects
            time intervals throughout observation window

        latitude (optional) : float, str, astropy.units.Quantity
            The latitude of the observing location (if site not provided).

        longitude (optional) : float, str, astropy.units.Quantity
            The longitude of the observing location (if site not provided).

        elevation (optional) : astropy.units.Quantity (default = 0 meters)
            The elevation of the observing location (if site not provided).

        Returns
        ---------
        '~gemini_classes.SunInfo'
            SunInfo Object

        Parameters
        ----------
        set : 'astropy.time.core.Time'
            UTC time of sun set at observer location

        rise : 'astropy.time.core.Time'
            UTC time of sun rise at observer location

        ra : 'astropy.coordinates.angles.Longitude'
            right ascension at solar midnight

        dec : 'astropy.coordinates.angles.Latitude'
            declination at solar midnight

        ZD : 'astropy.units.quantity.Quantity'
            zenith distance angle at times in utc_times

        HA : 'astropy.coordinates.angles.Angle'
            hour angle at times in utc_times

        AZ : 'astropy.units.quantity.Quantity'
            azimuth angle at times in utc_times

        """

        timer = False
        if timer:
            import time as t
            timerstart = t.time()  # runtime clock

        self.site = _checkinput_site(site=site,latitude=latitude,longitude=longitude, elevation=elevation)
        _checkinput_time(variable=utc_times,varname='utc_times')

        # sunset/sunrise times
        sun_horiz = sun_horizon(site)  # compute sun set/rise angle from zenith
        self.set = site.sun_set_time(utc_times[0], which='nearest', horizon=sun_horiz)
        self.rise = site.sun_rise_time(utc_times[0], which='next', horizon=sun_horiz)
        # self.set = None
        # self.rise = None
        # Sun info (assuming sun coordinates constant throughout night)
        solar_midnight = site.midnight(utc_times[0], which='nearest')  # get local midnight in utc time
        sun_pos = get_sun(solar_midnight)
        self.ra = sun_pos.ra
        self.dec = sun_pos.dec
        lst_times = site.local_sidereal_time(utc_times)
        self.ZD,self.HA,self.AZ = calc_ZDHA(lst=lst_times, latitude=site.location.lat, ra=sun_pos.ra, dec=sun_pos.dec)

        if timer: print('\n\tInitialize SunInfo: ', t.time() - timerstart)  # runtime clock

    def __repr__(self):

        """
        String representation of the gemini_classes.SunInfo object.
        """

        class_name = self.__class__.__name__
        attr_names = ['ra', 'dec', 'set', 'rise', 'ZD', 'HA', 'AZ']
        attr_values = [getattr(self, attr) for attr in attr_names]
        attributes_strings = []
        for name, value in zip(attr_names, attr_values):
            if value is not None:
                value = "'{}'".format(value)
                attributes_strings.append("{}={}".format(name, value))
        return "<{}: {}>".format(class_name, ",\n    ".join(attributes_strings))

    def table(self,showall=False):

        """
        Table representation of gemini_classes.TimeInfo object
        """

        sattr = '\t\t{0:<25s}{1}'  # print string and string
        fattr = '\t\t{0:<25s}{1:.2f}'  # print string and float
        table = []

        class_name = self.__class__.__name__
        attr_names = ['ra', 'dec', 'set', 'rise']
        table.append(str('\n\t'+class_name+':'))
        attr_values = [getattr(self, attr) for attr in attr_names]
        for name, value in zip(attr_names, attr_values):
            if value is not None:
                if name=='set' or name=='rise':
                    table.append(str(sattr.format(name, value.iso)))
                else:
                    table.append(str(fattr.format(name, value)))

        if showall:
            fheader = '\t\t{0:<20s}{1:<20s}{2:<20s}'
            fvals = '\t\t{0:<20.10}{1:<20.15}{2:<20.10}'
            attr_names = ['ZD', 'HA', 'AZ']
            attr_values = [getattr(self, attr) for attr in attr_names]
            table.append(str(''))
            table.append(str(fheader.format('ZD', 'HA', 'AZ')))
            table.append(str(fheader.format(attr_values[0][0].unit,attr_values[1][0].unit,attr_values[2][0].unit)))
            table.append(str(fheader.format('--------', '--------', '--------')))
            for i in range(0, len(attr_values[0])):
                table.append(str(fvals.format(str(attr_values[0][i]), str(attr_values[1][i]), str(attr_values[2][i]))))
        return table


class MoonInfo(object):

    """
    A container class for storing all required moon parameters
    for a given night (or similar time window).
    """

    @u.quantity_input(elevation=u.m)
    def __init__(self,site=None, utc_times=None,
                 latitude=None, longitude=None, elevation=0.0 * u.m):
        """
        Input
        ----------
        site : astroplan.Observer (or astropy.coordinates.Earthlocation)
            Observer location information

        utc_times : array of astropy.time.core.Time objects
            time intervals throughout observation window

        latitude (optional) : float, str, astropy.units.Quantity
            The latitude of the observing location (if site not provided).

        longitude (optional) : float, str, astropy.units.Quantity
            The longitude of the observing location (if site not provided).

        elevation (optional) : astropy.units.Quantity (default = 0 meters)
            The elevation of the observing location (if site not provided).

        Returns
        ---------
        '~gemini_classes.MoonInfo'
            MoonInfo Object

        Parameters
        ----------
        set : 'astropy.time.core.Time'
            UTC time of sun set at observer location

        rise : 'astropy.time.core.Time'
            UTC time of sun rise at observer location

        fraction : float
            fraction of moon illumated at time of
            solar midnight

        phase : 'astropy.units.quantity.Quantity'
            lunar phase angle of at solar midnight
            (full moon at 0 radians, new moon at pi radians)

        ramid : 'astropy.coordinates.angles.Longitude'
            right ascension of moon at solar midnight

        decmid : 'astropy.coordinates.angles.Latitude'
            declination of moon at solar midnight

        ra : 'astropy.coordinates.angles.Longitude'
            right ascension at times in utc_times

        dec : 'astropy.coordinates.angles.Latitude'
            declination at times in utc_times

        ZD : 'astropy.units.quantity.Quantity'
            zenith distance angle at times in utc_times

        HA : 'astropy.coordinates.angles.Angle'
            hour angle at times in utc_times

        AZ : 'astropy.units.quantity.Quantity'
            azimuth angle at times in utc_times

        AM : float
            airmass at times in utc_times

        """

        timer = False
        if timer:
            import time as t
            timerstart = t.time()  # runtime clock

        self.site = _checkinput_site(site=site, latitude=latitude, longitude=longitude, elevation=elevation)
        _checkinput_time(variable=utc_times, varname='utc_times')
        if timer:
            print('\n\tCheck times: ', t.time() - timerstart)  # runtime clock
            timerstart = t.time()

        sun_horiz = sun_horizon(site)  # compute sun set/rise angle from zenith
        if timer:
            print('\n\tInitialize MoonInfo horiz: ', t.time() - timerstart)  # runtime clock
            timerstart = t.time()

        self.set = site.moon_set_time(utc_times[0], which='next', horizon=sun_horiz)
        if timer:
            print('\tInitialize MoonInfo set: ', t.time() - timerstart)  # runtime clock
            timerstart = t.time()

        self.rise = site.moon_rise_time(utc_times[0], which='nearest', horizon=sun_horiz)
        if timer:
            print('\tInitialize MoonInfo rise: ', t.time() - timerstart)  # runtime clock
            timerstart = t.time()
        # self.set = None
        # self.rise = None

        solar_midnight = site.midnight(utc_times[0], which='nearest')  # get local midnight in utc time
        if timer:
            print('\tInitialize MoonInfo midnight: ', t.time() - timerstart)  # runtime clock
            timerstart = t.time()

        self.fraction = site.moon_illumination(solar_midnight)
        if timer:
            print('\tInitialize MoonInfo illum: ', t.time() - timerstart)  # runtime clock
            timerstart = t.time()

        self.phase = site.moon_phase(solar_midnight)
        if timer:
            print('\tInitialize MoonInfo phase: ', t.time() - timerstart)  # runtime clock
            timerstart = t.time()

        moon_pos = get_moon(solar_midnight,  location=self.site.location)
        self.ramid = moon_pos.ra
        self.decmid = moon_pos.dec
        if timer:
            print('\tInitialize MoonInfo ra,dec at midnight: ', t.time() - timerstart)  # runtime clock
            timerstart = t.time()

        # moon position at time intervals
        moon_pos = get_moon(utc_times, location=self.site.location)
        self.ra = moon_pos.ra
        self.dec = moon_pos.dec
        if timer:
            print('\tInitialize MoonInfo ras,decs: ', t.time() - timerstart)  # runtime clock
            timerstart = t.time()

        lst_times = site.local_sidereal_time(utc_times)
        if timer:
            print('\tInitialize MoonInfo lst: ', t.time() - timerstart)  # runtime clock
            timerstart = t.time()

        self.ZD,self.HA,self.AZ = calc_ZDHA(lst=lst_times, latitude=site.location.lat, ra=moon_pos.ra, dec=moon_pos.dec)
        self.AM = airmass(self.ZD)

        if timer:
            print('\n\tInitialize MoonInfo: ', t.time() - timerstart)  # runtime clock

    def __repr__(self):

        """
        String representation of the gemini_classes.MoonInfo object.
        """

        class_name = self.__class__.__name__
        attr_names = ['rise','set','fraction','phase','ramid','decmid','ZD','HA','AZ','AM']
        attr_values = [getattr(self, attr) for attr in attr_names]
        attributes_strings = []
        for name, value in zip(attr_names, attr_values):
            if value is not None:
                value = "'{}'".format(value)
                attributes_strings.append("{}={}".format(name, value))
        return "<{}: {}>".format(class_name, ",\n    ".join(attributes_strings))

    def table(self,showall=False):

        """
        Table representation of gemini_classes.TimeInfo object
        """

        sattr = '\t\t{0:<25s}{1}'  # print string and float
        fattr = '\t\t{0:<25s}{1:.2f}'  # print string and float
        table = []

        class_name = self.__class__.__name__
        # attr_names = ['fraction','phase','rise','set']
        attr_names = ['ramid', 'decmid', 'fraction', 'phase', 'rise', 'set']
        table.append(str('\n\t'+class_name+':'))
        attr_values = [getattr(self, attr) for attr in attr_names]
        for name, value in zip(attr_names, attr_values):
            if value is not None:
                if name=='set' or name=='rise':
                    table.append(str(sattr.format(name, value.iso)))
                elif name=='ramid':
                    table.append(str(fattr.format('ra', value)))
                elif name == 'decmid':
                    table.append(str(fattr.format('dec', value)))
                else:
                    table.append(str(fattr.format(name, value)))

        if showall:
            fheader = '\t\t{0:<12s}{1:<12s}{2:<12s}{3:<17s}{4:<12s}{5:<12s}'
            fvals = '\t\t{0:<12.8}{1:<12.8}{2:<12.8}{3:<17.13}{4:<12.8}{5:<12.8}'
            attr_names = ['ra','dec','ZD','HA','AZ','AM']
            attr_values = [getattr(self, attr) for attr in attr_names]
            table.append(str(''))
            table.append(str(fheader.format('ra','dec','ZD', 'HA', 'AZ', 'AM')))
            table.append(str(fheader.format(attr_values[0][0].unit,attr_values[1][0].unit,attr_values[2][0].unit,\
                                 attr_values[3][0].unit,attr_values[4][0].unit,'')))
            table.append(str(fheader.format('--------', '--------', '--------','--------', '--------', '--------')))
            for i in range(0, len(attr_values[0])):
                table.append(str(fvals.format(str(attr_values[0][i]),str(attr_values[1][i]),\
                                    str(attr_values[2][i]),str(attr_values[3][i]),\
                                    str(attr_values[4][i]), str(attr_values[5][i]))))
        return table



class TargetInfo(object):

    """
    A container class for storing all required target parameters
    for a given night (or similar time window).
    """

    @u.quantity_input(elevation=u.m)
    def __init__(self, site=None, utc_times=None, name='', ra=None, dec=None,
                 latitude=None, longitude=None, elevation=0.0 * u.m):
        """
        Input
        ----------
        site : astroplan.Observer (or astropy.coordinates.Earthlocation)
            Observer location information

        name : string
            Unique gemini observation identifier

        utc_times : array of astropy.time.core.Time objects
            time intervals throughout observation window

        latitude (optional) : float, str, astropy.units.Quantity
            The latitude of the observing location (if site not provided).

        longitude (optional) : float, str, astropy.units.Quantity
            The longitude of the observing location (if site not provided).

        elevation (optional) : astropy.units.Quantity (default = 0 meters)
            The elevation of the observing location (if site not provided).

        Returns
        ---------
        '~gemini_classes.MoonInfo'
            MoonInfo Object

        Parameters
        ----------
        set : 'astropy.time.core.Time'
            UTC time of sun set at observer location

        rise : 'astropy.time.core.Time'
            UTC time of sun rise at observer location

        fraction : float
            fraction of moon illumated at time of
            solar midnight

        phase : 'astropy.units.quantity.Quantity'
            lunar phase angle of at solar midnight
            (full moon at 0 radians, new moon at pi radians)

        ramid : 'astropy.coordinates.angles.Longitude'
            right ascension of moon at solar midnight

        decmid : 'astropy.coordinates.angles.Latitude'
            declination of moon at solar midnight

        ra : 'astropy.coordinates.angles.Longitude'
            right ascension

        dec : 'astropy.coordinates.angles.Latitude'
            declination

        ZD : 'astropy.units.quantity.Quantity'
            zenith distance angle at times in utc_times

        HA : 'astropy.coordinates.angles.Angle'
            hour angle at times in utc_times

        AZ : 'astropy.units.quantity.Quantity'
            azimuth angle at times in utc_times

        AM : float
            airmass at times in utc_times

        mdist : 'astropy.units.quantity.Quantity'
            angular separation between target and moon
            at times in utc_times

        vsb : float
            visible sky background magnitude of target
            location at times in utc_times

        bg : None
            attribute for storing sky background condition
            values at times in utc_times

        weight : None
            attribute for string observation weights computed
            at times in utc_times
        """
        timer = False
        if timer:
            import time as t
            timerstart = t.time()  # runtime clock

        _checkinput_site(site=site, latitude=latitude, longitude=longitude, elevation=elevation)
        _checkinput_time(variable=utc_times, varname='utc_times')

        self.ra,self.dec = _checkinput_coord(ra=ra,dec=dec)

        self.name = name

        lst_times = site.local_sidereal_time(utc_times)
        self.ZD, self.HA, self.AZ = calc_ZDHA(lst=lst_times, latitude=site.location.lat, ra=ra, dec=dec)
        self.AM = airmass(self.ZD)
        self.mdist = None
        self.vsb = None
        self.bg = None
        self.weight = None

        if timer: print('Initialize TargetInfo: ', t.time() - timerstart)  # runtime clock

    def __repr__(self):

        """
        String representation of the gemini_classes.TargetInfo object.
        """

        class_name = self.__class__.__name__
        attr_names = ['name','ra','dec','ZD','HA','AZ','AM','mdist','vsb','bg','weight']
        attr_values = [getattr(self, attr) for attr in attr_names]
        attributes_strings = []
        for name, value in zip(attr_names, attr_values):
            if value is not None:
                value = "'{}'".format(value)
                attributes_strings.append("{}={}".format(name, value))
        return "<{}: {}>".format(class_name, ",\n    ".join(attributes_strings))

