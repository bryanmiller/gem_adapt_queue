import numpy as np
import re
import astropy.units as u

def obsweight(dec, AM, HA, AZ, band, user_prior, status, latitude,
              cond, acond, obs_time, elev, wind=[0.*u.m/u.s,0.*u.deg], wra=1., starttime=17.):

     """
     Calculate observation weight at all time intervals
     
     Parameters
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

     Return
     --------
     weight - array of floats with length nt
     """
     verbose = False

     nt = len(HA) # number of time steps
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
     if verbose: 
          better_iq = acond['iq']<cond['iq']
          better_cc = acond['cc']<cond['cc']
          print('iq better than required',better_iq)
          print('cc better than required',better_cc)

     i_better_iq = np.where(acond['iq']<cond['iq'])[0][:]
     if len(i_better_iq)!=0:
          cmatch = cmatch * 0.75

     i_better_cc = np.where(acond['cc']<cond['cc'])[0][:]
     if len(i_better_cc)!=0:
          cmatch = cmatch * 0.75

     if verbose: print('cmatch',cmatch)
     
     # Compute single weight representative of conditions 
     twcond = (1./cond['iq'])**3 + (1./cond['cc'])**3 + (1./cond['bg'])**3 + (1./cond['wv'])**3
     
     if verbose: 
          print('wiq,wcc,wbg,wwv',(1./cond['iq'])**3, (1./cond['cc'])**3, (1./cond['bg'])**3, (1./cond['wv'])**3)
          print('twcond',twcond)

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
          print('elev',elev)

     if elev['type']=='Airmass':
          i_bad_elev = np.where(np.logical_or(AM<elev['min'],AM>elev['max']))[0][:]
          wam[i_bad_elev] = 0.
     elif elev['type']=='HourAngle':
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
     if verbose:
         print('wdec',wdec)
         print('lat', latitude)
         print('decdiff', decdiff)
         print('starttime', starttime)
         print('HA/unit^2', HA/(u.hourangle**2))

     # HA - if within -1hr of transit at  twilight it gets higher weight
     if np.logical_and(abs(decdiff)<40.*u.deg,starttime>12.):
          c = wdec * np.array([3.,0.1,-0.06]) #weighted to slightly positive HA
     else:
          c = wdec * np.array([3.,0.,-0.08]) #weighted to 0 HA if Xmin > 1.3
     wha = c[0] + c[1]/u.hourangle*HA + c[2]/(u.hourangle**2)*HA**2
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
     if obs_time > 0.:
          wstatus = 2.0
     
     if verbose: print('wstatus',wstatus)



     # ======================== Partner Balance ========================
     wbal = 0.

     if verbose:
         print('wbal',wbal)
         print('wra', wra)


     # ======================== Total weight ========================
     weight = (twcond + wstatus * wha + wband + wprior + wbal + wra) * cmatch * wam * wcplt * wwind
     
     if verbose: print('Total weight',weight)

     return weight



def calc_weight(site,obs,timeinfo,targetinfo,acond):

     """
     Parameters
     ----------
     Site object :param site: astroplan.Observer

     OT catalog observation info :param obs: gemini_classes.Gobservations

     Observing window time info :param timeinfo: gemini_classes.TimeInfo

     Target time dependent parameters :param targetinfo: gemini_classes.TargetInfo

     Actual conditions :param acond: list of converted sky visibility conditions as
          decimal values (i.e. acond=[iq, cc, bg, wv])

     Return
     --------
     List of gemini_classes.TargetInfo objects with calculated weights
     """

     verbose = False

     #   ======================= target ra distribution =======================
     ras = np.array([target.ra/u.deg for target in targetinfo])*u.deg
     bin_edges = [0.,30.,60.,90.,120.,150.,180.,210.,240.,270.,300.,330.,360.]*u.deg

     if verbose:
        print('target ra distribution...')
        print('ras',ras)
        print('bins edges',bin_edges)

     bin_nums = np.digitize(ras,bins=bin_edges) -1  # get ra bin index for each target

     if verbose:
        print('histogram bin indices',bin_nums)

     # Sum total observing hours in bins and divide mean (wra weight)
     hhra = np.zeros(12)*u.h
     for i in np.arange(0,12):
        ii = np.where(bin_nums==i)[0][:]
        hhra[i] = hhra[i] + sum(obs.tot_time[ii]-obs.obs_time[ii])
     hhra = hhra/np.mean(hhra)

     if verbose: print('hhra (ra distribution weight)',hhra)

     nt = timeinfo.nt
     n_obs = len(obs.obs_id)

     for i in range(n_obs):

          #   ================== convert sky background mag to bg condition ===================
          # define sky background condition
          sbcond = np.zeros(nt,dtype=float)
          ii = np.where(targetinfo[i].vsb <= 19.61)[0][:]
          sbcond[ii] = 1.
          ii = np.where(np.logical_and(targetinfo[i].vsb>=19.61,targetinfo[i].vsb<=20.78))[0][:]
          sbcond[ii] = 0.8
          ii = np.where(np.logical_and(targetinfo[i].vsb>=20.78,targetinfo[i].vsb<=21.37))[0][:]
          sbcond[ii] = 0.5
          ii = np.where(targetinfo[i].vsb>=21.37)[0][:]
          sbcond[ii] = 0.2

          # set actual and required conditions
          cond = {'iq': obs.iq[i], 'cc': obs.cc[i], 'bg': obs.bg[i], 'wv': obs.wv[i]}
          actualcond = {'iq': acond[0], 'cc': acond[1], 'bg': sbcond, 'wv': acond[3]}

          if verbose:
              print('cond', cond)
              print('acond', actualcond)

          #=================== Compute observation weights ========================
          ttime = np.round((obs.tot_time[i]-obs.obs_time[i])*10)/10 # remaining time in observation

          if verbose:
               ii = np.where(obs.prog_ref == obs.prog_ref[i])[0][:]
               ptime = np.round(sum(obs.tot_time[ii] - obs.obs_time[ii]) * 10) / 10
               print('Prog remaining time (ptime):',ptime)
               print('Obs. remaining time (ttime):',ttime)

          if ttime > 0.0:

               ii = 0  # reset index value
               for j in np.arange(12):  # Get corresponding hour angle histogram bin
                    if bin_edges[j] <= obs.ra[i] < bin_edges[j + 1]:  # get ra historgram bin
                         ii = j

               # Convert elevation constraints dictionaries
               # of the form {'type':string,'min':float,'max':float}
               if (obs.elev_const[i].find('None') != -1) or (obs.elev_const[i].find('null') != -1) or (
                       obs.elev_const[i].find('*NaN') != -1):
                    emin = 0.
                    emax = 0.
                    etype = 'None'
                    # print('Read none, null, or *NaN',self.elev_const[i])
               elif obs.elev_const[i].find('Hour') != -1:
                    nums = re.findall(r'\d+.\d+', obs.elev_const[i])
                    emin = nums[0]
                    emax = nums[1]
                    etype = 'Hour Angle'
                    # print('Read Hour',self.elev_const[i])
               elif obs.elev_const[i].find('Airmass') != -1:
                    nums = re.findall(r'\d+.\d+', obs.elev_const[i])
                    # print('Read Airmass',AirmassConstraint(min=float(nums[0]),max=float(nums[1])))
                    emin = nums[0]
                    emax = nums[1]
                    etype = 'Airmass'
               else:
                    print('Could not read elevation constraint from catalog = ', obs.elev_const[i])
                    None
               elev = {'type': etype, 'min': float(emin), 'max': float(emax)}

               if verbose:
                    print('Obs. weights: ',obs.obs_id[i])
                    print('dec',obs.dec[i])
                    print('latitude',site.location.lat)
                    print('band',obs.band[i])
                    print('user_prior',obs.user_prior[i])
                    print('status',obs.comp_time[i])
                    print('elevation constraint ',elev)
                    print('condition constraints',cond)
                    print('actual conditions',acond)

               targetinfo[i].weight = obsweight(dec=targetinfo[i].dec, AM=targetinfo[i].AM,
                                         HA=targetinfo[i].HA, AZ=targetinfo[i].AZ, band=obs.band[i],
                                         user_prior=obs.user_prior[i], status=obs.comp_time[i],
                                         latitude=site.location.lat, cond=cond, acond=actualcond,
                                         obs_time=obs.obs_time[i], wra=hhra[ii], elev=elev)
          else:
               targetinfo[i].weight = np.zeros(nt)


     return targetinfo