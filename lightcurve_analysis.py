# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:09:24 2019

@author: Amena Faruqi
"""

import lightkurve as lk
import pandas as pd
from PyAstronomy import pyasl
import matplotlib.pyplot as plt
import numpy as np
from bisect import bisect
import scipy.signal as sig
from astropy.convolution import convolve, Box1DKernel
from astropy.time import Time

tsi = pd.read_csv('tsi_data_6h.txt') # TSI SORCE data (6-hourly)

montet = pd.read_csv('montet_stars.txt')  #Original data from Montet et al. (2018)

#Downloaded from: https://iopscience.iop.org/0004-637X/851/2/116/suppdata/apjaa9e00t1_mrt.txt

#-------------------------------- FILTER DATA ----------------------------------------
montet = montet[(montet['Notes1'] ==  0)] #Filter out short-period binaries

#Temperature limit stars for analysis (sun-like stars only):
T_upper = 6000
T_lower = 5000
#Convert chosen temperature limits to B-V using method in Ramirez & Melendez (2005)
bv_upper = pyasl.Ramirez2005().teffToColor_nop("B-V",T_upper,0.0) 
bv_lower = pyasl.Ramirez2005().teffToColor_nop("B-V",T_lower,0.0)

montet_bv = montet[(montet['B-V'] < bv_lower)] #Filter stars by temperature
montet_bv = montet[(montet['B-V'] > bv_upper)]

montet = montet_bv[(montet_bv['Scatter'] < 10)] #Remove high scatter stars (> 10)
montet = montet[(montet['Prot'] > 1)] #Remove stars with rotations periods < 1 day

#----------------------------------------------------------------------------------------


fac = montet[(montet['Var'] == 'Faculae')].reset_index() #Montet stars identified as facula-dominated
spots = montet[(montet['Var'] == 'Spots')].reset_index() #Montet stars identified as spot-dominated


#-------------------------------- Kepler Data Functions ----------------------------------
def remove_anomalies(lc):
    """
    Uses PyAstronomy package to remove anomalous individual data points and 
    replace them with NaN.
    Inputs: Kepler lightcurve object (lightkurve)
    Outputs: Kepler lightcurve object (lightkurve) with anomalies removed
    
    """
    fluxes = lc.flux
    anomaly_pos = pyasl.pointDistGESD(fluxes, 10, alpha=0.05)[1]
    for i in anomaly_pos:
        fluxes[i] = np.nan
    lc.flux = fluxes
    return lc


def makeLCQ(ID,q):
    """
    Generates a Kepler lightcurve file for a given target ID and quarter.
    1 <= q <= 17
    """
    tpf = lk.search_targetpixelfile(ID, quarter = q).download()
    lightcurve = tpf.to_lightcurve()
    lightcurve = remove_anomalies(lightcurve)
    return lightcurve

def joinLC(ID):
    """
    Generates a normalized lightcurve for the entire duration of the Kepler 
    mission (missing quarters are removed).
    """
    lcfiles = lk.search_lightcurvefile(ID).download_all()
    stitched_lc = lcfiles[0].PDCSAP_FLUX.normalize().flatten(window_length=401)
    
    for i in range(1,len(lcfiles)):
        lc = lcfiles[i].PDCSAP_FLUX.normalize().flatten(window_length=401)
        stitched_lc = stitched_lc.append(lc)
        
    stitched_lc = stitched_lc.remove_outliers()
    stitched_lc = remove_anomalies(stitched_lc).remove_nans()
    
    return stitched_lc


def check_stars(i1,i2,df):
    """
    Function used to view plots of the first quarter of multiple Kepler
    stars at once, to obtain an overview of data. 
    df = dataframe being checked
    i1,i2 = indexes of beginning and end of the subset of data being checked
    """
    for index,row in df[i1:i2].iterrows():
        star_id = row['KID']
        prot = round(row['Prot'],3)
        lc = makeLCQ(star_id,1)
        lc.flatten(window_length=401).plot()
        lc.plot()
        plt.title('Rot. period = %g days, Q = %g' %(prot, 1))
            
def check_quarters(ID,q1,q2):
    """
    Function used to view plots of multiple quarters of a single Kepler star 
    at once. Displays quarters range from q1 to q2 (not inclusive).  
    """
    loc = fac.loc[fac['KID'] == ID]
    prot = loc['Prot']
    for q in range(q1,q2):
        try:
            lc = makeLCQ(ID,q)
            lc.flatten(window_length=401).bin(binsize=5).plot()
            lc.plot()
            plt.title('Rot. period = %g, Q = %g' %(prot,q))
        except:
            pass
      
def rot_periods(IDs):
    """
    Function to calculate the rotation period of a Kepler star using the 
    periodogram function, for comparison to those found using the methods of 
    McQuillan et al. (2014). 
    """
    montet_rot_periods = []
    lk_rot_periods = []
    for ID in IDs:
        montet_period = fac.loc[fac['KID'] == ID]['Prot']
        montet_rot_periods.append(float(montet_period))
        
        lc = joinLC(ID)
        pg = lc.to_periodogram(oversample_factor=1)
        period = pg.period_at_max_power.value
        lk_rot_periods.append(period)
    
    return montet_rot_periods,lk_rot_periods

#---------------------------------------------------------------------------------

#------------------- Single-Double Ratios (Basri et al. 2018) ------------------------

def SDR(x,y,prot,yerr=0,sep='peaks',retSD=True):
    """
    Calculates the SDR of lightcurve data from any source. 
    x = time data (days)
    y = (normalized) flux 
    yerr = errors associated with fluxes (default value = 0)
    sep = 'dips' or 'peaks'
    """
    
    dt = x - np.roll(x, 1)
    dt = dt[1:]
    ddt = np.median(dt)
    if np.isnan(ddt) == True:
        ddt = 0.1
    
    width = int(np.round(prot/(ddt*8)))  
    if width == 0:
        width = 1
    y = convolve(y,Box1DKernel(width),'wrap')
    pwidth = 0.02  # peak width must be greater than Kepler cadence (30 mins)

    if sep=='peaks':
        peaks = sig.find_peaks(y,width=pwidth,prominence=yerr)[0]
    elif sep=='dips':
        peaks = sig.find_peaks(-y,width=pwidth,prominence=yerr)[0]
        
    xpeaks = [x[i] for i in peaks]
    ypeaks = [y[i] for i in peaks]

    psep = xpeaks - np.roll(xpeaks, 1)
    psep = psep[1:]
    
    psep = np.round(psep, 13)
    psep, counts = np.unique(psep, return_counts=True)

    if retSD==False:        
        return xpeaks,ypeaks,psep,y
    
    elif retSD==True:
        tdb = 0
        tsn = 0
        if len(psep) > 0:
            for sep in psep:
                if sep > 0.8*prot:
                    tsn += 1
                else:
                    tdb += 1
        
        #print(tsn,tdb)
        if tdb > 0 and tsn > 0:
            sdr = np.log10(tsn/tdb)
            if sdr > 2.0:
                sdr = 2.0
            elif sdr < -2.0:
                sdr = -2.0
        
        elif tdb == 0 and tsn != 0:
            sdr = 2.0
        elif tsn == 0 and tdb != 0:
            sdr = -2.0
        else:
            sdr = 0.0 
        
        return sdr 
        
    else:
        raise Exception("retSD must be True or False")
    
      
def binned_SDR(x,y,prot,yerr=0,makeplots=True,sep='peaks'):
    """
    Calculates multiple SDRs for a given star by binning lightcurve data.
    
    """
    y_smooth = SDR(x,y,prot,yerr,sep='peaks',retSD=False)[3]
    xrange = np.max(x) - np.min(x)   
    binsize = prot*1.5
    binshift = prot
    maxshift = (xrange-binsize)/binshift
    xlims = []
    x_sdr = []
    sdrs = []
    for n in range(int(maxshift)+1):  
        lower_lim = np.min(x) + n*binshift
        upper_lim = lower_lim + binsize
        lower_ind = bisect(x,lower_lim)
        upper_ind = bisect(x,upper_lim)
        ybin = y[lower_ind:upper_ind]
        xbin = x[lower_ind:upper_ind]
        xlims.append(x[lower_ind])
        xlims.append(x[upper_ind])
        
        if type(yerr) == np.ndarray:
            yerrbin = yerr[lower_ind:upper_ind]
        else:
            yerrbin = 0

        xmid = np.median(xbin)
        x_sdr.append(xmid)
        sdr = SDR(xbin,ybin,prot,yerrbin,sep,retSD=True)
        
            
        sdrs.append(sdr)
        
    #-------------------------- Generate Plots -------------------------------
    if makeplots == True:
        f, (ax1, ax2) = plt.subplots(2,1,sharex=True)
        ax1.plot(x,y,color='black')
        ax1.plot(x,y_smooth,color='b')
        ax2.set_xlabel('Time (JDN)')
        ax2.plot(x_sdr,sdrs,color = 'black')
        for v in range(0,len(xlims),2):
            color = np.random.rand(3,)
            ax2.scatter(x_sdr[int(v/2)],sdrs[int(v/2)],c=color)
            ax2.axvline(xlims[v],ls='--',c = color)
            ax2.axvline(xlims[v+1],ls='--',c = color)
            
        ax1.set_ylabel('Normalized Flux')
        ax2.set_ylabel('SDR')   
        ax2.set_ylim(-2.0,2.0)
        fig = plt.gcf()
        fig.suptitle('Rotation Period: %g days, Separation = %s'%(prot,sep))
        
    #-------------------------------------------------------------------------
    return x_sdr,sdrs


#-------------------------- Kepler SDR Functions -------------------------------
    
def kepler_SDR(lc,th=3,makeplots=True):
    """
    Calculates the single-double ratio (SDR) of Kepler stars, from Basri et 
    al. (2018) i.e. the ratio of the time spent by the star in 'single mode' 
    to the time spent in 'double mode'. 
    
    th = threshold for prominence of peaks/dips detected.
    
    """
    kid = lc.targetid
    prot = float(montet[montet['KID'] == kid]['Prot'])
    binwidth=10
    lc = lc.flatten(window_length=401).remove_nans().normalize().bin(binsize=binwidth)
    
    x = lc.time    
    y = lc.flux
    y_std = lc.flux_err*th
    
    xpeaks,ypeaks,psep,y_smooth = SDR(x,y,prot,y_std,sep='peaks',retSD=False)
    xdips,ydips,dsep,y_smooth = SDR(x,y,prot,y_std,sep='dips',retSD=False)
    
    #------------------------- Plot Outputs --------------------------------
    if makeplots == True:
        lc.plot()
        plt.scatter(xpeaks,ypeaks,color='r')
        plt.scatter(xdips,ydips,color='b')
        plt.title('Original LC Data')
        
        plt.figure()
        plt.plot(x,y_smooth)
        plt.scatter(xpeaks,ypeaks,color='r')
        plt.scatter(xdips,ydips,color='b')
        plt.xlabel('Time(JDN)')
        plt.ylabel('Normalized Flux')
        plt.title('Smoothed LC Data')
        plt.show()
    
    #-----------------------------------------------------------------------
    
    peaks_sdr = SDR(x,y,prot,y_std,sep='peaks',retSD=True)
    dips_sdr = SDR(x,y,prot,y_std,sep='dips',retSD=True)
    
    return peaks_sdr,dips_sdr

def kepler_SDR_binned(lc,th=3,makeplots=True):
    """
    Calculates the single-double ratio (SDR) of Kepler stars, within 
    overlapping bins of lightcurve data. 
    
    th = threshold for prominence of peaks/dips detected.
    
    """
    kid = lc.targetid
    prot = float(montet[montet['KID'] == kid]['Prot'])
    binwidth=10
    lc = lc.flatten(window_length=401).remove_nans().normalize().bin(binsize=binwidth)
    
    x = lc.time    
    y = lc.flux
    y_std = lc.flux_err*th
    
    xp_sdr,psdrs = binned_SDR(x,y,prot,y_std,makeplots,sep='peaks')
    xd_sdr,dsdrs = binned_SDR(x,y,prot,y_std,makeplots,sep='dips')
    
    psdrcount = psdrs.count(-2.0)     
    sdr_pfrac = psdrcount/len(psdrs)   # double peak fraction
    
    return sdr_pfrac

     

#----------------------------- TSI SDR Functions -----------------------------
    
def TSI_SDR(start_date,end_date,makeplots=True):
    """
    Calculates the single-double ratio (SDR) of TSI data, from the input 
    start date to end date. 
    
    Dates should be given as a string in the format 'YYYY-MM-DD'.
    
    """
    data = tsi[(tsi['tsi_1au'] > 0)] #remove null irradiance values 
    
    prot = 24   #rotation period of the sun
    
    #binwidth=10
    x = np.array(data['avg_time_JDN'])  
    y = np.array(data['tsi_1au'])
    y_std = np.array(data['std_tsi_1au'])/np.median(y) 
    y = y/np.median(y)                     #normalise y values 
    
    start_date_jd = Time(start_date,format='iso').jd 
    end_date_jd = Time(end_date,format='iso').jd     
    
    start_ind = bisect(x.tolist(),start_date_jd)
    end_ind = bisect(x.tolist(),end_date_jd)
    
    y = y[start_ind:end_ind]
    x = x[start_ind:end_ind]
    y_std = y_std[start_ind:end_ind]
    

    xpeaks,ypeaks,psep,y_smooth = SDR(x,y,prot,y_std,sep='peaks',retSD=False)
    xdips,ydips,dsep,y_smooth = SDR(x,y,prot,y_std,sep='dips',retSD=False)
    
    #---------------------------- Plot Outputs -------------------------------
    if makeplots == True:
        plt.figure()
        plt.plot(x,y,color='black')
        #plt.scatter(xpeaks,ypeaks,color='r')
        #plt.scatter(xdips,ydips,color='b')
        plt.title('Original TSI Data from %s to %s' %(start_date,end_date))
        plt.xlabel('Average Date JDN')
        plt.ylabel('Normalized Flux')
    
        
        plt.figure()
        plt.plot(x,y_smooth,color='black')
        plt.scatter(xpeaks,ypeaks,color='r', label = 'Peaks')
        plt.scatter(xdips,ydips,color='b',label = 'Dips')
        plt.title('Smoothed TSI Data from %s to %s' %(start_date,end_date))
        plt.xlabel('Average Date JDN')
        plt.ylabel('Normalized Flux')
        plt.legend()
        plt.show()
      
    #------------------------------------------------------------------------
    
    peaks_sdr = SDR(x,y,prot,y_std,sep='peaks',retSD=True)
    dips_sdr = SDR(x,y,prot,y_std,sep='dips',retSD=True)
    
    return peaks_sdr,dips_sdr

def TSI_SDR_binned(start_date,end_date,makeplots=True):
    """
    Calculates the single-double ratio (SDR) of TSI data within bins slightly
    larger than the solar rotation period, from the input 
    start date to end date. 

    """
    data = tsi[(tsi['tsi_1au'] > 0)] #remove null irradiance values 
    
    prot = 24   #rotation period of the sun
    
    #binwidth=10
    x = np.array(data['avg_time_JDN'])  
    y = np.array(data['tsi_1au'])
    y_std = np.array(data['std_tsi_1au'])/np.median(y)
    y = y/np.median(y)   #normalise y values 
    
    start_date_jd = Time(start_date,format='iso').jd 
    end_date_jd = Time(end_date,format='iso').jd     
    
    start_ind = bisect(x.tolist(),start_date_jd)
    end_ind = bisect(x.tolist(),end_date_jd)
    
    y = y[start_ind:end_ind]
    x = x[start_ind:end_ind]
    y_std = y_std[start_ind:end_ind]
    

    xp_sdr,psdrs = binned_SDR(x,y,prot,y_std,makeplots,sep='peaks')
    xd_sdr,dsdrs = binned_SDR(x,y,prot,y_std,makeplots,sep='dips')

    psdrcount = psdrs.count(-2.0)
    sdr_pfrac = psdrcount/len(psdrs)  # double peak fraction
    
    return sdr_pfrac

#------------------------------------------------------------------------------
        
    
def spot_fac_subsets(showhist=True):
    """
    Generate databases of spot-dominated and facula-dominated stars that have 
    similar temperatures and rotation periods to help identify distinguishable
    features due to spots/faculae. 
    
    """
    spots_subset = pd.DataFrame(columns=fac.columns)
    fac_subset = fac
    fac_approx = fac.round(0)
    spots_approx = spots.round(0)
    for index,row in fac_approx.iterrows():
        spot_matches = spots_approx[(spots_approx['Prot'] == row['Prot'])]   # identify spots with similar rotation periods
        spot_matches = spot_matches[(spot_matches['B-V'] == row['B-V'])]     # identify spots with similar B-V and prot
        if len(spot_matches) > 0:
            i = spot_matches.index[0] 
            spots_approx = spots_approx.drop(i)
            spots_subset = pd.concat([spots_subset,spots[i:i+1]])
        else:
            fac_subset = fac_subset.drop(index)
    
    fac_subset = fac_subset.drop(fac_subset.columns[0:3],axis=1)
    spots_subset = spots_subset.drop(spots_subset.columns[0:3],axis=1)
    fac_prots = fac_subset['Prot'].tolist()
    spot_prots = spots_subset['Prot'].tolist()
    fac_bvs = fac_subset['B-V'].tolist()
    spot_bvs = spots_subset['B-V'].tolist()
    
    fac_prot_avg = np.mean(fac_prots)
    fac_prot_std = np.std(fac_prots)
    fac_bv_avg = np.mean(fac_bvs)
    fac_bv_std = np.std(fac_bvs)
    
    spot_prot_avg = np.mean(spot_prots)
    spot_prot_std = np.std(spot_prots)
    spot_bv_avg = np.mean(spot_bvs)
    spot_bv_std = np.std(spot_bvs)
    
    #------------------------------ Histograms ---------------------------------
    """
    Generate histograms of rotation periods/B-V values for the produced 
    databases, to verify that the datasets are comparable. 
    """
    if showhist == True:
        plt.figure()
        plt.hist(fac_prots,alpha=0.5,label = 'Faculae, mean = %g, sd = %g' %(fac_prot_avg,fac_prot_std))
        plt.hist(spot_prots,alpha=0.5,label = 'Spots, mean = %g, sd = %g' %(spot_prot_avg,spot_prot_std))
        plt.xlabel('Rotation Period (Days)')
        plt.ylabel('Frequency')
        plt.title('Rotation periods of Faculae and Spots')
        plt.legend()
        plt.show()
        
        plt.figure()
        plt.hist(fac_bvs,alpha=0.5,label = 'Faculae, mean = %g,sd = %g' %(fac_bv_avg,fac_bv_std))
        plt.hist(spot_bvs,alpha=0.5,label = 'Spots, mean = %g,sd = %g' %(spot_bv_avg,spot_bv_std))
        plt.xlabel('B-V')
        plt.ylabel('Frequency')
        plt.title('B-V of Faculae and Spots')
        plt.legend()
        plt.show()

    
    #----------------------------------------------------------------------------
    
    return spots_subset.reset_index(),fac_subset.reset_index()



#---------------------------- Variability Measures ----------------------------
    
def spread(x,y,prot,bin_frac,var='percentiles'):
    """
    Determines measures of spread across bins for a lightcurve, 'var' can be:
        - rms = root mean square
        - p95_to_p5 = difference in 95th and 5th percentile
        - std = standard deviation 
        
    Time values (x-axis) are also binned, the median value of each time 
    bin is used for plotting.
    
    """
    
    xrange = np.max(x) - np.min(x)   
    binsize = prot                    # bin width 
    binshift = prot/bin_frac          # amount by which to shift the left most edge of bin along
    maxshift = (xrange-binsize)/binshift
    var_list = []
    t_list = []

    for n in range(int(np.round(maxshift)+1)):  
        lower_lim = np.min(x) + n*binshift
        upper_lim = lower_lim + binsize
        lower_ind = bisect(x,lower_lim)
        upper_ind = bisect(x,upper_lim)
        bin_values = y[lower_ind:upper_ind]
        t_avg = np.median(x[lower_ind:upper_ind])
        
        if len(bin_values) > 0:
        
            t_list.append(t_avg)
            
            if var.lower() == 'percentiles':
                p95_to_p5 = (np.percentile(bin_values,95) - np.percentile(bin_values,5)) 
                var_list.append(p95_to_p5)
                
            elif var.lower() == 'std':
                std = np.std(bin_values) 
                var_list.append(std)
                
            elif var.lower() == 'rms':
                 rms = ((np.mean(bin_values**2))**0.5)/np.mean(bin_values)
                 var_list.append(rms)  
        else:
            pass
    
    return t_list,var_list
    

def Rvar(db,datasource='kepler',showhist=True,binned=False):
    """
    Calculates Rvar values (Basri et al. 2013) for a database of Kepler stars.
    datasource =  'kepler', 'tsi' or 'sim' for Kepler data, TSI data and 
    simulation data, respectively. 
    db = database containing data needed to calculate Rvar (must match datasource)
    
    """
    xlist = []
    ylist = []
    protlist = []
    rvars = []
    
    if datasource.lower() ==  'kepler': 
        startype = db['Var'].tolist()[0][:-1] + '-dominated Kepler Stars'
        for kid in db['KID'].tolist():
            print(kid)
            prot = float(db[db['KID'] == kid]['Prot'])
            protlist.append(prot)
            lc = joinLC(kid)
            lc = lc.normalize().flatten(window_length=401).remove_nans()
            y = lc.flux   
            x = lc.time
            ylist.append(y)
            xlist.append(x)
            
    elif datasource.lower() == 'tsi':
        db = db[(db['tsi_1au'] > 0)]
        startype = 'the Sun'
        prot = 24
        protlist.append(prot)
        y = db['tsi_1au']
        y = (y/np.median(y)).tolist()
        x = db['avg_time_JDN'].tolist()
        ylist.append(y)
        xlist.append(x)
        
    elif datasource.lower() == 'sim':
        startype = ' a Simulated Star'
        prot = 1
        protlist.append(prot)
        y = db['y']
        y = (y/np.median(y)).tolist()
        x = db['x'].tolist()
        ylist.append(y)
        xlist.append(x)
        
    else:
        raise Exception('Not a valid source of data')
        
    for stari in range(len(xlist)):
        x = xlist[stari]
        y = ylist[stari]  
        prot = protlist[stari]
        xbins,all_rvars = spread(x,y,prot,3)
        
        if binned == True:
            f, (ax1, ax2) = plt.subplots(2,1,sharex=True)
            ax1.plot(x,y,color='black')
            ax2.set_xlabel('Time (JDN)')
            ax2.plot(xbins,all_rvars,color = 'black')
            ax2.scatter(xbins,all_rvars,color = 'r')
            ax1.set_ylabel('Normalized Flux')
            ax2.set_ylabel('$R_{var}$')   
            fig = plt.gcf()
            fig.suptitle('Rotation Period: %g days'%(prot))
        
        
        med_rvar = np.median(all_rvars)
        rvars.append(med_rvar)

    # ---------------------------- Histogram -------------------------------
    if showhist == True:
        plt.figure()
        rvar_avg = np.mean(rvars)
        rvar_std = np.std(rvars)
        plt.hist(rvars,bins=10,label = 'mean = %g, sd = %g' %(rvar_avg,rvar_std))
        plt.xlabel('$R_{var}$')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title('$R_{var}$ for %s' %startype)
        
    # ----------------------------------------------------------------------
    
    return rvars

    

def MDV(db,binsize,datasource = 'kepler',showhist=True):
    """
    Calculates MDV values (Basri et al. 2013) for a database of stars.
    datasource =  'kepler', 'tsi' or 'sim' for Kepler data, TSI data and 
    simulation data, respectively. 
    db = database containing data needed to calculate MDV (must match datasource)
    
    """
    xlist = []
    ylist = []
    mdvs = []
    
    if datasource.lower() == 'kepler':
        startype = db['Var'].tolist()[0][:-1] + '-dominated Kepler Stars'
        for kid in db['KID'].tolist():
            print(kid)
            lc = joinLC(kid)
            y = lc.flux
            x = lc.time
            ylist.append(y)
            xlist.append(x)
            
    elif datasource.lower() == 'tsi':
        db = db[(db['tsi_1au'] > 0)]
        startype = 'the Sun'
        y = db['tsi_1au']
        y = (y/np.median(y)).tolist()
        x = db['avg_time_JDN'].tolist()
        ylist.append(y)
        xlist.append(x)
        
    elif datasource.lower() == 'sim':
        startype = ' a Simulated Star'
        y = db['y']
        y = (y/np.median(y)).tolist()
        x = db['x'].tolist()
        ylist.append(y)
        xlist.append(x)
        
    else:
        raise Exception('Not a valid source of data')
    
    
    for stari in range(len(xlist)):
        bin_avgs = []
        x = xlist[stari]
        y = ylist[stari]
        xrange = np.max(x) - np.min(x)   
        binshift = binsize 
        maxshift = (xrange-binsize)/binshift
        for n in range(int(np.round(maxshift)+1)):  
            lower_lim = np.min(x) + n*binshift
            upper_lim = lower_lim + binsize
            lower_ind = bisect(x,lower_lim)
            upper_ind = bisect(x,upper_lim)
            ybin = y[lower_ind:upper_ind]
            if len(ybin) > 0:
                avg_ybin = np.mean(ybin)
                bin_avgs.append(avg_ybin)
        
        bin_avgs = np.array(bin_avgs)
        bindiff = bin_avgs - np.roll(bin_avgs,1)
        bindiff = bindiff[1:]
        mdv = np.median(bindiff)
        
        mdvs.append(mdv)            

    # ---------------------------- Histogram -------------------------------
    if showhist == True:
        plt.figure()
        mdv_avg = np.mean(mdvs)
        mdv_std = np.std(mdvs)
        plt.hist(mdvs,bins='auto',label = 'mean = %g, sd = %g' %(mdv_avg,mdv_std))
        plt.xlabel('MDV')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title('MDV for %s' %startype)
        plt.show()
    # ----------------------------------------------------------------------
        
    return mdvs
        

