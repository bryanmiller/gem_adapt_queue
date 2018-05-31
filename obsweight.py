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
     if wind==None: wind=[0.,0.]
     if wra==None: wra=1.0
     if starttime==None: starttime=17.0

     nt = len(HA) # number of time steps
     weight = np.ones(nt) 
     cmatch = np.ones(nt)
     wam = np.ones(nt)
     wwind = np.ones(nt)
     
     # print(cond)
     # print(acond)

     # ======================== Conditions comparison ========================
     # Return 0 weight if requested conditions worse than actual.
     bad_iq = acond['iq']>cond['iq']
     bad_cc = acond['cc']>cond['cc']
     bad_bg = acond['bg']>cond['bg']
     bad_wv = acond['wv']>cond['wv']
     i_bad_cond = np.where(np.logical_or(np.logical_or(bad_iq,bad_cc),\
                         np.logical_or(bad_bg,bad_wv)))[0][:]
     cmatch[i_bad_cond] = 0.

     #decrease weight if conditions are better than requested iq,cc. 
     better_iq = acond['iq']<cond['iq']
     better_cc = acond['cc']>cond['cc']
     i_better_cond = np.where(np.logical_or(better_iq,better_cc))[0][:]
     cmatch[i_better_cond] = cmatch[i_better_cond] * 0.75

     # print(bad_iq),print(bad_cc),print(bad_bg),print(bad_wv),print(i_bad_cond),print(i_better_cond),print(cmatch)
     
     # Compute single weight representative of conditions 
     twcond = (1./cond['iq'])**3 + (1./cond['cc'])**3 + (1./cond['bg'])**3 + (1./cond['wv'])**3
     # print('twcond',twcond)

     # ======================== Airmass ========================
     i_bad_AM = np.where(AM>2.)[0][:]
     wam[i_bad_AM] = 0.
     # print(AM)
     # print(wam)


     # ======================== Elevation constraint ========================
     # Change constraints for testing
     # elev['type']='HourAngle'
     # elev['min']=-5
     # elev['max']=5
     
     # print(elev)
     # print('wam',wam)
     # print('AM',AM)
     # print('HA',HA)
     if elev['type']=='Airmass':
          i_bad_elev = np.where(np.logical_or(AM<elev['min'],AM>elev['max']))[0][:]
          wam[i_bad_elev] = 0.
     if elev['type']=='HourAngle':
          i_bad_elev = np.where(np.logical_or(HA<elev['min'],HA>elev['max']))[0][:]
          wam[i_bad_elev] = 0.
     else:
          None
     # print(wam)


     # ======================== Wind ========================
     # Wind, do not point within 20deg of wind if over limit
     wind = np.zeros(1,dtype={'names':('speed','dir'),'formats':('f8','f8')})
     wind['speed'] = 4.e3 #in m/h
     wind['dir']= 30. #in degrees

     
     ii = np.where(np.logical_and(wind['speed']>10.e3,abs(AZ - wind['dir']) < 20.))[0][:]
     wwind[ii] = 0.
     # print('wind',wind)
     # print('AZ',AZ)
     # print('wwind',wwind)


     # ======================== Visibility ========================
     if latitude<0:
          decdiff = latitude-dec
     else:
          decdiff = dec-latitude
     # print('lat',latitude)
     # print('dec',dec)
     # print('decdiff',decdiff)
     # print('HA',HA)
     
     declim=[90.,-30.,-45.,-50,-90.]
     wval=[1.0,1.3,1.6,2.0]
     wdec=0.
     for i in np.arange(4):
          if np.logical_and(decdiff<declim[i],decdiff>declim[i+1]):
               wdec = wval[i]
          else:
               None
     #print('wdec',wdec)

     # HA - if within -1hr of transit at  twilight it gets higher weight
     if np.logical_and(abs(decdiff)<40.,starttime>12.):
          c = wdec * np.array([3.,0.1,-0.06]) #weighted to slightly positive HA
     else:
          c = wdec * np.array([3.,0.,-0.08]) #weighted to 0 HA if Xmin > 1.3
     wha = c[0] + c[1]*HA + c[2]*HA**2
     ii = np.where(wha<=0)[0][:]
     wha[ii] = 0.
     # print('wha',wha)
     
     if np.amin(HA) >= -1.:
          wha = wha*1.5
     # print('min HA',np.amin(HA))
     # print('wha',wha)


     # ======================== Band ========================
     wband = (4. - band) * 1000
     # print('wband',wband)


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
     # print('wprior',wprior)


     # ======================== Completion Status ========================
     wcplt = 1.
     if status >= 1.0:
          wcplt = 0.
          
     # print('wcplt',wcplt)

     wstatus = 1.
     if status > 0.:
          wstatus = wstatus * 1.5
     if otime > 0.:
          wstatus = 2.0
     # print('wstatus',wstatus)



     # ======================== Partner Balance ========================
     wbal = 0.


     # ======================== Total weight ========================
     weight = (twcond + wstatus * wha + wband + wprior + wbal +wra) * cmatch * wam * wcplt * wwind
     # print('Total weight',weight)

     if verbose:
          print('cond',cond)
          print('acond',acond[0])
          print('twcond',twcond)
          print('wdec',wdec)
          print('wband',wband)
          print('wprior',wprior)
          print('wcplt',wcplt)
          print('wstatus',wstatus)
          print('wbal',wbal)
          print('cmatch',cmatch)
          print('wam',wam)
          print('wwind',wwind)
          print('wha',wha)
          print('Total weight: ',weight)
     return weight