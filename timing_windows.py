#!/usr/bin/env python3

# OT timing window manipulation
# Bryan Miller
# 2018-05-18

import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
import astroplan as ap
import re
import timeutils as tu
# from astroplan import download_IERS_A
# download_IERS_A()

# To make horizon calculation the same as skycalc
sunhoriz = -0.83*u.deg
EQUAT_RAD = 6378137.*u.m    # equatorial radius of earth, meters

# A date or duration to use for infinity
# This could be the deactivation date of the program, or three years at most (LPs)
# infinity = Time('2400-12-31 23:59:59.9', scale='utc')
infinity = 3. * 365. * 24. * u.h

# site
gs = ap.Observer.at_site('gemini_south')
print(gs.location.lon)
print(gs.location.lat)
print(gs.location.height)
gshoriz = np.sqrt(2.*gs.location.height/EQUAT_RAD)*(180./np.pi)*u.deg
# print(gshoriz)

# gn = ap.Observer.at_site('gemini_north', timezone="US/Hawaii")
# print(gn.location.height)
# gnhoriz = np.sqrt(2.*gn.location.height/EQUAT_RAD)*(180./np.pi)*u.deg

# string from OT ascii catalog dump
#
# otwin = '[{1522454400000 86400000 0 0}, {1523664000000 86400000 0 0}, {1523923200000 86400000 0 0}]'

# 18A-FT-107-11
# epoch 1
# otwin = '[{1493948945000 3600000 400 140740000}]'
# epoch 2,3
otwin = '[{1488592145000 3600000 400 140740000}]'
# New, all
otwin = '[{1526872703000 1800000 100 140740000}]'

target =  SkyCoord('10:14:51.895 -47:09:24.65',frame='icrs', unit = (u.h, u.deg))

# GS-2018A-Q-133-25
# otwin = '[{1524614400000 -1 0 0}]'

# split into a list
otstr = re.sub('[\[{}\]]','',otwin)
winlist = otstr.split(',')
# print(winlist)

for window in winlist:
    # print(window)
    values = window.strip(' ').split(' ')

    # The timestamps are in milliseconds
    # The start time is unix time (milliseconds from 1970-01-01 00:00:00) UTC
    # Time requires unix time in seconds
    t0 = float(values[0]) * u.ms
    begin = Time(t0.to_value('s'), format='unix', scale='utc')
    # print(begin.iso)

    # duration = -1 means forever
    duration = float(values[1])
    if duration == -1.0:
        duration = infinity
    else:
        # duration =  duration * u.ms
        duration = duration / 3600000. * u.h

    # repeat = -1 means infinite
    repeat = int(values[2])
    if repeat == -1:
        repeat = 1000

    # period between repeats
    # period = float(values[3]) * u.ms
    period = float(values[3]) / 3600000. * u.h

    if repeat == 0:
        start = begin
        end = start + duration

        print(start.iso, end.iso, duration.to_value('h'), repeat, tu.dec2sex(period.to_value('h')))
    else:
        for ii in range(repeat):
            start = begin + float(ii) * period
            end = start + duration

            evening_twilight = gs.twilight_evening_nautical(start, which='nearest')
            morning_twilight = gs.twilight_morning_nautical(start, which='next')
            # print(evening_twilight.iso, morning_twilight.iso)

            # Is window at night?
            if start > evening_twilight and end < morning_twilight:
                # targ_rise = gs.target_rise_time(start, target)
                # targ_set = gs.target_set_time(start, target, which='next')
                # Is target below airmass 2 during window?
                if gs.altaz(start, target).secz >= 1.0 and gs.altaz(start, target).secz < 2.0 and \
                    gs.altaz(end, target).secz >= 1.0 and gs.altaz(end, target).secz < 2.0:
                    # print(start.iso, end.iso, duration.to_value('h'), repeat, tu.dec2sex(period.to_value('h')),
                    #       gs.altaz(start, target).secz)
                    print('{:20} {:20} {:4.2f} {:4.2f}'.format(start.iso, end.iso, gs.altaz(start, target).secz,
                                                             gs.altaz(end, target).secz))
