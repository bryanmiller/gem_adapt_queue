import numpy as np
import astropy.units as u

def obsweight(cond, dec, AM, HA, AZ, band, user_prior, status, latitude,\
               acond, wind, otime, wra, elev, starttime):
     
     verbose = False

     """
     Calculate observation weights
     
     Definitions
     -----------
     cond - observation condition constraints, output from convcond
     dec  - decimal dec
     AM   - airmass
     HA   - decimal hour angle
     AZ   - azimuth
     band - ranking band
     user_prior - user priority
     status - completion status as fractional percent
     latitude - observator latitude
     time - hours remaining in observation
     acond - actual conditions (output from convcond)
     wind - wind speed (m/s) and direction
     otime - time used (observed)
     wra - RA weighting
     elev - elevation constraint
     starttime - beginning time of the plan
     """

     # if acond==None: acond=[0.,0.,0.,0.]
     if wind==None: wind=[0.*u.m/u.s,0.*u.deg]
     if wra==None: wra=1.0
     if starttime==None: starttime=17.0

     nt = len(HA) # number of time steps
     weight = np.ones(nt) 
     cmatch = np.ones(nt)
     wam = np.ones(nt)
     wwind = np.ones(nt)
     
     if verbose:
          print(cond)
          print(acond)

     # ======================== Conditions comparison ========================
     # Return 0 weight if requested conditions worse than actual.
     bad_iq = acond['iq']>cond['iq']
     bad_cc = acond['cc']>cond['cc']
     bad_bg = acond['bg']>cond['bg']
     bad_wv = acond['wv']>cond['wv']

     i_bad_cond = np.where(np.logical_or(np.logical_or(bad_iq,bad_cc),\
                         np.logical_or(bad_bg,bad_wv)))[0][:]

     if verbose:
          print('iq worse than required',bad_iq)
          print('cc worse than required',bad_cc)
          print('bg worse than required',bad_bg)
          print('wv worse than required',bad_wv)
          print('i_bad_cond',i_bad_cond)
     

     cmatch[i_bad_cond] = 0.

     #decrease weight if conditions are better than requested iq,cc.
     # *effectively drop one band if IQ or CC are better than needed
     # and not likely to lose target, need to make this wavelen. dep. 
     better_iq = acond['iq']<cond['iq']
     better_cc = acond['cc']<cond['cc']

     if verbose: 
          print('iq better than required',better_iq)
          print('cc better than required',better_cc)

     i_better_cond = np.where(np.logical_or(better_iq,better_cc))[0][:]
     if len(i_better_cond)!=0:
          cmatch[i_better_cond] = cmatch[i_better_cond] * 0.75

     if verbose: print('cmatch',cmatch)
     
     # Compute single weight representative of conditions 
     twcond = (1./cond['iq'])**3 + (1./cond['cc'])**3 + (1./cond['bg'])**3 + (1./cond['wv'])**3
     if verbose: print('twcond',twcond)

     # ======================== Airmass ========================
     i_bad_AM = np.where(AM>2.)[0][:]
     wam[i_bad_AM] = 0.



     # ======================== Elevation constraint ========================
     # Change constraints for testing
     # elev['type']='HourAngle'
     # elev['min']=-5
     # elev['max']=5
     
     if verbose:
          print('AM',AM)
          print('HA',HA)

     if elev['type']=='Airmass':
          i_bad_elev = np.where(np.logical_or(AM<elev['min'],AM>elev['max']))[0][:]
          wam[i_bad_elev] = 0.
     if elev['type']=='HourAngle':
          i_bad_elev = np.where(np.logical_or(HA<elev['min'],HA>elev['max']))[0][:]
          wam[i_bad_elev] = 0.
     else:
          None

     if verbose: print('wam',wam)


     # ======================== Wind ========================
     # Wind, do not point within 20deg of wind if over limit
     
     ii = np.where(np.logical_and(wind[0]>10.e3*u.m/u.s,abs(AZ - wind[1]) < 20.*u.deg))[0][:]
     wwind[ii] = 0.
     
     if verbose: print('wwind',wwind)


     # ======================== Visibility ========================
     if latitude<0:
          decdiff = latitude-dec
     else:
          decdiff = dec-latitude
     
     declim=[90.,-30.,-45.,-50,-90.]*u.deg
     wval=[1.0,1.3,1.6,2.0]
     wdec=0.
     for i in np.arange(4):
          if np.logical_and(decdiff<declim[i],decdiff>declim[i+1]):
               wdec = wval[i]
          else:
               None
     if verbose: print('wdec',wdec)

     # HA - if within -1hr of transit at  twilight it gets higher weight
     if np.logical_and(abs(decdiff)<40.*u.deg,starttime>12.):
          c = wdec * np.array([3.,0.1,-0.06]) #weighted to slightly positive HA
     else:
          c = wdec * np.array([3.,0.,-0.08]) #weighted to 0 HA if Xmin > 1.3
     wha = c[0] + c[1]*HA/(u.hourangle) + c[2]*HA**2/(u.hourangle**2)
     ii = np.where(wha<=0)[0][:]
     wha[ii] = 0.
          
     
     if np.amin(HA) >= -1.*u.hourangle:
          wha = wha*1.5
          if verbose: print('multiplied wha by 1.5')
     
     if verbose: 
          print('wha',wha)
          print('min HA',np.amin(HA))
          


     # ======================== Band ========================
     wband = (4. - np.int(band)) * 1000
     if verbose: print('wband',wband)


     # ======================== User Priority ========================
     if user_prior == 'Target of Opportunity':
          wprior = 500.
     elif user_prior == 'High':
          wprior = 2.
     elif user_prior == 'Medium':
          wprior = 1.
     elif user_prior == 'Low':
          wprior = 0.
     else:
          wprior = 0.
     
     if verbose: print('wprior',wprior)


     # ======================== Completion Status ========================
     wcplt = 1.
     if status >= 1.0:
          wcplt = 0.
          
     if verbose: print('wcplt',wcplt)

     wstatus = 1.
     if status > 0.:
          wstatus = wstatus * 1.5
     if otime > 0.:
          wstatus = 2.0
     
     if verbose: print('wstatus',wstatus)



     # ======================== Partner Balance ========================
     wbal = 0.

     if verbose: print('wbal',wbal)


     # ======================== Total weight ========================
     weight = (twcond + wstatus * wha + wband + wprior + wbal +wra) * cmatch * wam * wcplt * wwind
     
     if verbose: print('Total weight',weight)

     return weight