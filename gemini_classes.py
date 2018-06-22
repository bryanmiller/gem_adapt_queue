import re
import numpy as np
import astropy.units as u
from conversions import conditions
from calc_ZDHA import calc_ZDHA
from astropy import time
from astropy.coordinates import (EarthLocation, get_sun, get_moon, angles)
from astroplan import (Observer)

def hms_to_hr(timestring): # convert 'HH:MM:SS' string to hours
    (h, m, s) = timestring.split(':')
    return (np.int(h) + np.int(m)/60 + np.int(s)/3600)

def airmass(ZD): # calculate airmasses
    AM = np.full(len(ZD),20.)
    ii = np.where(ZD < 87.*u.deg)[0][:]
    sec_z = 1. / np.cos(ZD[ii])
    AM[ii] = sec_z - 0.0018167 * ( sec_z - 1 ) - 0.002875 * ( sec_z - 1 )**2 - 0.0008083 * ( sec_z - 1 )**3 #compute moon AMs
    return AM

def sun_horizon(site): # calculate angle of sun at sunset for observer site
    sun_horiz = -.83 * u.degree
    equat_radius = 6378137. * u.m
    return sun_horiz - np.sqrt(2. * site.location.height / equat_radius) * (180. / np.pi) * u.degree

def _checkinput_site(site,latitude,longitude,elevation):
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
    if isinstance(variable, time.core.Time):
        return variable
    elif isinstance(variable, str):
        return time.Time(variable)
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

        latitude (optional) : float, str, astropy.units.Quantity
            The latitude of the observing location (if site not provided).

        longitude (optional) : float, str, astropy.units.Quantity
            The longitude of the observing location (if site not provided).

        elevation (optional) : astropy.units.Quantity (default = 0 meters)
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

        if dt == None:
            dt = 0.1  # 1/10 hr default step size
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

        sun_horiz = sun_horizon(site)  # compute sun set/rise angle from zenith
        solar_midnight = site.midnight(utc_times[0], which='nearest')  # get local midnight in utc time

        self.set = site.moon_set_time(utc_times[0], which='next', horizon=sun_horiz)
        self.rise = site.moon_rise_time(utc_times[0], which='nearest', horizon=sun_horiz)
        self.fraction = site.moon_illumination(solar_midnight)
        self.phase = site.moon_phase(solar_midnight)
        moon_pos = get_moon(solar_midnight,  location=self.site.location)
        self.ramid = moon_pos.ra
        self.decmid = moon_pos.dec

        # moon position at time intervals
        moon_pos = get_moon(utc_times, location=self.site.location)
        self.ra = moon_pos.ra
        self.dec = moon_pos.dec
        lst_times = site.local_sidereal_time(utc_times)
        self.ZD,self.HA,self.AZ = calc_ZDHA(lst=lst_times, latitude=site.location.lat, ra=moon_pos.ra, dec=moon_pos.dec)
        self.AM = airmass(self.ZD)

        if timer: print('\n\tInitialize MoonInfo: ', t.time() - timerstart)  # runtime clock

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
        attr_names = ['ramid','decmid','fraction','phase','rise','set']
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



class Gobservations(object):
    """
    A container class for gemini program information.
    
    Requires Gcatalog object as input (see gemini_classes.Gcatalog()).
    Automatically converts some variable types,
    performs merging of some columns, and performs computation
    and conversion of elevation constraints, conditions contraints, completed time,
    and total time.


    Attributes
    --------
    prog_ref            (string)                unique program identifier
    obs_id              (string)                unique observation identifier
    pi                  (string)                
    inst                (string)                instrument name
    target              (string)                target name
    ra                  (astropy degrees)       right ascension degrees
    dec                 (astropy degrees)       declination degrees
    band                (int)                   integer
    partner             (string)                gemini partner name
    obs_status          (string)                'ready' status of observation
    tot_time            (astropy hours)         total planned observation time
    obs_time            (astropy hours)         completed observation time
    obs_comp            (float)                 fraction of completed/total observation time
    charged_time        (string)                HH:MM:SS (required to compute obs_comp)
    obs_class           (string)                observation class
    iq                  (float)                 image quality constraint (percentile converted to decimal value) 
    cc                  (float)                 cloud condition constraint (percentile converted to decimal value) 
    bg                  (float)                 sky background constraint (percentile converted to decimal value) 
    wv                  (float)                 water vapor constraint (percentile converted to decimal value) 
    user_prior          (string)                user priority 
    group               (string)                observation group name
    elev_const          (dictionary)            elevation constraint {'type':string,'min':float,'max':float}
    ready               (boolean)               ready status
    disperser           (string)                disperser name
    fpu                 (string)                focal plane unit
    grcwlen             (string)                grating control wavelength
    crwlen              (string)                central wavelength
    filter              (string)                filter name
    mask                (string)                mask name 
    xbin                (string)                xbin number 
    ybin                (string)                ybin number 
    """


    def __init__(self, catinfo=None, i_obs=None):

        timer = False
        if timer:
            import time as t
            timerstart = t.time()  # runtime clock

        n_obs = len(i_obs)

        # copy required strings from catalog object 
        self.prog_ref = catinfo.prog_ref[i_obs]
        self.obs_id = catinfo.obs_id[i_obs]
        self.pi = catinfo.pi[i_obs]
        self.inst = catinfo.inst[i_obs]
        self.target = catinfo.target[i_obs]
        self.ra = np.array(list(map(float, catinfo.ra[i_obs]))) * u.deg
        self.dec = np.array(list(map(float, catinfo.dec[i_obs]))) * u.deg
        self.band = np.array(list(map(int, catinfo.band[i_obs])))
        self.partner = catinfo.partner[i_obs]
        self.obs_status = catinfo.obs_status[i_obs]
        # self.qa_status = catinfo.obs_qa[i_obs]
        # self.dataflow_step = catinfo.dataflow_step[i_obs]
        self.tot_time = catinfo.planned_exec_time[i_obs]
        self.obs_comp = np.zeros(n_obs)
        # self.planned_pi_time = catinfo.planned_pi_time[i_obs]
        self.charged_time = catinfo.charged_time[i_obs]
        self.obs_time = self.charged_time
        self.obs_class = catinfo.obs_class[i_obs]
        self.bg = catinfo.sky_bg[i_obs]
        self.wv = catinfo.wv[i_obs]
        self.cc = catinfo.cloud[i_obs]
        self.iq = catinfo.iq[i_obs]
        self.user_prior = catinfo.user_prio[i_obs]
        # self.ao = catinfo.ao[i_obs]
        self.group = catinfo.group[i_obs]
        # self.group_type = catinfo.gt[i_obs]
        self.elev_const = catinfo.elev_const[i_obs]
        # self.time_const = catinfo.time_const[i_obs]
        self.ready = np.array(list(map(bool, catinfo.ready[i_obs])))
        # self.color_filter = catinfo.color_filter[i_obs]
        # self.nd_filter = catinfo.neutral_density_filter[i_obs]
        # self.binning = catinfo.binning[i_obs]
        # self.windowing = catinfo.windowing[i_obs]
        # self.lens = catinfo.lens[i_obs]
        # self.cass_rotator = catinfo.cass_rotator[i_obs]
        # self.bh_ccdamps = catinfo.ccd_amplifiers[i_obs]
        # self.bh_ccdgain = catinfo.ccd_gain[i_obs]
        # self.bh_ccdspeed = catinfo.ccd_speed[i_obs]
        self.bh_xbin = catinfo.ccd_x_binning[i_obs]
        self.bh_ybin = catinfo.ccd_y_binning[i_obs]
        # self.bh_fibre = catinfo.entrance_fibre[i_obs]
        # self.bh_expmeter_filter = catinfo.exposure_meter_filter[i_obs]
        # self.bh_hartmann = catinfo.hartmann_flap[i_obs]
        # self.bh_issport = catinfo.iss_port[i_obs]
        # self.bh_pslitfilter = catinfo.post_slit_filter[i_obs]
        # self.bh_roi = catinfo.region_of_interest[i_obs]
        self.f2_disperser = catinfo.disperser[i_obs]
        self.f2_filter = catinfo.filter[i_obs]
        # self.f2_readmode = catinfo.read_mode[i_obs]
        # self.f2_lyot = catinfo.lyot_wheel[i_obs]
        self.f2_fpu = catinfo.focal_plane_unit[i_obs]
        self.grcwlen = catinfo.grating_ctrl_wvl[i_obs]
        self.xbin = catinfo.x_bin[i_obs]
        self.ybin = catinfo.y_bin[i_obs]
        # self.roi = catinfo.builtin_roi[i_obs]
        # self.nodshuffle = catinfo.nod_shuffle[i_obs]
        # self.dtax = catinfo.dta_x_offset[i_obs]
        # self.custom_mask = catinfo.custom_mask_mdf[i_obs]
        # self.preimage = catinfo.mos_pre_imaging[i_obs]
        # self.amp_count = catinfo.amp_count[i_obs]
        self.disperser = catinfo.disperser_2[i_obs]
        self.filter = catinfo.filter_2[i_obs]
        self.fpu = catinfo.fpu[i_obs]
        # self.detector = catinfo.detector_manufacturer[i_obs]
        # self.FIELD063 = catinfo.grating_ctrl_wvl_2[i_obs]
        # self.FIELD064 = catinfo.x_bin_2[i_obs]
        # self.FIELD065 = catinfo.y_bin_2[i_obs]
        # self.FIELD066 = catinfo.builtin_roi_2[i_obs]
        # self.FIELD067 = catinfo.nod_shuffle_2[i_obs]
        # self.FIELD068 = catinfo.dta_x_offset_2[i_obs]
        # self.FIELD069 = catinfo.custom_mask_mdf_2[i_obs]
        # self.FIELD070 = catinfo.mos_pre_imaging_2[i_obs]
        # self.FIELD071 = catinfo.amp_count_2[i_obs]
        # self.FIELD072 = catinfo.disperser_3[i_obs]
        # self.FIELD073 = catinfo.filter_3[i_obs]
        # self.FIELD074 = catinfo.fpu_2[i_obs]
        # self.FIELD075 = catinfo.detector_manufacturer_2[i_obs]
        # self.pixel_scale = catinfo.pixel_scale[i_obs]
        # self.FIELD077 = catinfo.disperser_4[i_obs]
        # self.FIELD078 = catinfo.focal_plane_unit_2[i_obs]
        # self.cross_dispersed = catinfo.cross_dispersed[i_obs]
        # self.FIELD080 = catinfo.read_mode_2[i_obs]
        self.crwlen = catinfo.central_wavelength[i_obs]
        # self.iss_port = catinfo.iss_port_2[i_obs]
        # self.FIELD083 = catinfo.well_depth[i_obs]
        # self.FIELD084 = catinfo.filter_4[i_obs]
        # self.readmode = catinfo.read_mode_3[i_obs]
        # self.astrometric = catinfo.astrometric_field[i_obs]
        # self.FIELD087 = catinfo.disperser_5[i_obs]
        # self.adc = catinfo.adc[i_obs]
        # self.observing_mode = catinfo.observing_mode[i_obs]
        # self.coadds = catinfo.coadds[i_obs]
        # self.exptime = catinfo.exposure_time[i_obs]
        # self.FIELD092 = catinfo.disperser_6[i_obs]
        self.mask = catinfo.mask[i_obs]
        # self.eng_mask = catinfo.engineering_mask[i_obs]
        # self.FIELD095 = catinfo.filter_[i_obs]
        # self.order = catinfo.disperser_order[i_obs]
        # self.nici_fpu = catinfo.focal_plane_mask[i_obs]
        # self.nici_pupil = catinfo.pupil_mask[i_obs]
        # self.nici_cassrot = catinfo.cass_rotator_2[i_obs]
        # self.nici_imgmode = catinfo.imaging_mode[i_obs]
        # self.nici_dichroic = catinfo.dichroic_wheel[i_obs]
        # self.nici_fw1 = catinfo.filter_red_channel[i_obs]
        # self.nici_fw2 = catinfo.filter_blue_channel[i_obs]
        # self.nici_welldepth = catinfo.well_depth_2[i_obs]
        # self.nici_dhs = catinfo.dhs_mode[i_obs]
        # self.imaging_mirror = catinfo.imaging_mirror[i_obs]
        # self.FIELD107 = catinfo.disperser_7[i_obs]
        # self.FIELD108 = catinfo.mask_2[i_obs]
        # self.FIELD109 = catinfo.filter_6[i_obs]
        # self.FIELD110 = catinfo.read_mode_4[i_obs]
        # self.camera = catinfo.camera[i_obs]
        # self.FIELD112 = catinfo.disperser_8[i_obs]
        # self.FIELD113 = catinfo.mask_3[i_obs]
        # self.FIELD114 = catinfo.filter_7[i_obs]
        # self.beam_splitter = catinfo.beam_splitter[i_obs]
        # self.FIELD116 = catinfo.read_mode_5[i_obs]
        # self.FIELD117 = catinfo.mask_4[i_obs]
        # self.FIELD118 = catinfo.filter_8[i_obs]
        # self.FIELD119 = catinfo.disperser_9[i_obs]
        # self.FIELD120 = catinfo.disperser_10[i_obs]
        # self.FIELD121 = catinfo.mask_5[i_obs]
        # self.FIELD122 = catinfo.filter_9[i_obs]


        # Merge data columns together
        ii = np.where(self.f2_fpu != 'null')[0][:]
        if len(ii)!=0:
            self.fpu[ii]=self.f2_fpu[ii]

        ii = np.where(self.crwlen != 'null')[0][:]
        if len(ii)!=0:
            self.grcwlen[ii]=self.crwlen[ii]

        ii = np.where(self.bh_xbin != 'null')[0][:]
        if len(ii)!=0:
            self.xbin[ii]=self.bh_xbin[ii]

        ii = np.where(self.bh_ybin != 'null')[0][:]
        if len(ii)!=0:
            self.ybin[ii]=self.bh_ybin[ii]

        ii = np.where(self.mask != 'null')[0][:]
        if len(ii)!=0:
            self.fpu[ii]=self.mask[ii]

        ii = np.where(self.group == '')[0][:]
        if len(ii)!=0:
            self.group[ii]=self.obs_id[ii]   

        ii = np.where(self.charged_time == '')[0][:]
        if len(ii)!=0:
            self.charged_time[ii]='00:00:00'

        # Conditions contraints
        for i in range(0,n_obs):
            if self.cc[i][0:1] == 'P': self.cc[i] == self.cc[i][8:10]
            if self.cc[i][0:1] == 'A': self.cc[i] =='Any'
            if self.iq[i][0:1] == 'P': self.iq[i] == self.iq[i][8:10]
            if self.iq[i][0:1] == 'A': self.iq[i] =='Any'
            if self.wv[i][0:1] == 'P': self.wv[i] == self.wv[i][8:10]
            if self.wv[i][0:1] == 'A': self.wv[i] == 'Any'

        # Convert condition contraints from strings to decimal values
        self.iq,self.cc,self.bg,self.wv = conditions(self.iq,self.cc,self.bg,self.wv)

        # Convert completed and total times.
        for i in range(0,n_obs): #cycle through selected observations
            #compute observed/total time, add additional time if necessary
            charged_time = hms_to_hr(self.charged_time[i])
            tot_time = hms_to_hr(self.tot_time[i])
            if (charged_time>0.):
                if 'Mirror' in self.disperser[i]:
                    tot_time = tot_time + 0.2
                else:
                    tot_time = tot_time + 0.3
            self.obs_comp[i] = charged_time / tot_time
            self.tot_time[i] = tot_time
            self.obs_time[i] = charged_time
        self.obs_time = np.array(self.obs_time, dtype=float) * u.h
        self.tot_time = np.array(self.tot_time, dtype=float) * u.h

        if timer: print('\n\tInitialize Gobservations: ', t.time() - timerstart)  # runtime clock


class Gcatfile(object):

    """
    Rough version of a container class for reading and storing catalog information.

    Reads file columns in to a dictionary w/ column names matching headers in OT file.
    
    Attribute naming convention: lowercase column headers w/ underscores (eg.'Obs. Status'=obs_status).

    Issue: repeated column names are not combined into a single attribute name.
    If an column name is repeated, a numerical value is appended to the corresponding attr. name (eg. 'disperser_2').
    """

    def __init__(self,otfile=None):

        cattext = []
        with open(otfile, 'r') as readcattext: #read file into memory. 
            #[print(line.split('\t')) for line in readcattext]
            [cattext.append(line.split('\t')) for line in readcattext] #Split lines where tabs ('\t') are found.
            readcattext.close()

        names = np.array(cattext[8])
        obsall = np.array(cattext[10:])

        #print('\notcat attribute names...',names)
        existing_names = []
        for i in range(0,len(names)):
            #remove special charaters, trim ends of string, replace whitespace with underscore
            string=names[i].replace('.','')
            string=re.sub(r'\W',' ',string)
            string=string.strip()
            string=re.sub(r' +','_',string)
            string=string.lower()
            if string=='class': #change attribute name (python doesn't allow attribute name 'class')
                string = 'obs_class'
            if np.isin(string,existing_names): #add number to end of repeated attribute name
                rename = True
                j = 0    
                while rename:
                    if j>=8: #if number 9 is reached, make next number 10
                        tempstring = string+'_1'+chr(50+j-10)
                    else:
                        tempstring = string+'_'+chr(50+j)
                    if np.isin(tempstring,existing_names): #if name taken, increment number and check again
                        j+=1
                    else: 
                        string = tempstring
                        rename = False

            setattr(self, string, obsall[:,i]) #set attribute name an include corresponding catalog column
            existing_names.append(string) #add name to library of used names
            #print(string)

        #print('\nFound '+str(len(obsall))+' observations in '+str(otfile))