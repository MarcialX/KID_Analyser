#!/usr/bin/env python
# -*- coding: utf-8 -*-
#************************************************************
#*                    Data Reduction                        *
#*          Salvador Ventura - Marcial Becerril             *
#************************************************************

import numpy as np
from matplotlib import pyplot as plt
from numpy import fft
import math
import glob
import os
from astropy.io import fits
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from scipy import signal
import sys

class dataRed():

    # Getting the arrays  from the fits file in the path given
    def get_arrays_fits(self, path):
        data_fits = fits.getdata(path, ext=0)
        return data_fits

    def get_header(self, path):
        hdul = fits.open(path)
        hdr = hdul[1].header

        I0 = hdr['IF0']
        Q0 = hdr['QF0']
    
        dqdf = hdr['DQDF']
        didf = hdr['DIDF']
        
        Fs = hdr['SAMPLERA']
        
        actual_temp = hdr['SAMPLETE']
        kid_number = hdr['TONE']
        input_att = hdr['INPUTATT']

        f0 = hdr['SYNTHFRE']

        return I0,Q0,Fs,actual_temp,kid_number,input_att,didf,dqdf,f0

    def get_vna_sweep(self,path):
        data_fits = self.get_arrays_fits(path)

        freqs = data_fits.field(0)
        I = data_fits.field(1)
        Q = data_fits.field(2)   
        return freqs,I,Q

    def smooth_IQ(self,I,Q):    
        sI = savgol_filter(I,51,10)
        sQ = savgol_filter(Q,51,10)
       # fs21 = np.sqrt((fI**2)+(fQ**2))
        return sI,sQ

    def get_vna_sweep_parameters(self,freqs,I,Q):
        
        mag = np.sqrt((I**2)+(Q**2))
        f0_ind = np.argmin(mag)

        f0 = freqs[f0_ind]
        I0 = I[f0_ind]
        Q0 = Q[f0_ind]

        try:
            didf = (I[f0_ind + 1]-I[f0_ind])/(freqs[f0_ind + 1]-freqs[f0_ind])
            dqdf = (Q[f0_ind + 1]-Q[f0_ind])/(freqs[f0_ind + 1]-freqs[f0_ind])
            check = True
        except:
            didf = 0
            dqdf = 0
            check = False

        return mag,f0,I0,Q0,didf,dqdf,check
        
    def df(self, I0,Q0,didf,dqdf,I,Q):
        vel = (didf**2)+(dqdf**2)
        df = [  ( ( I[i]-I0 ) * didf) + ( ( Q[i] - Q0) * dqdf ) / vel for i in range(len(I))  ]
        return df, vel

    def get_IQ_data_homo(self, path):
        data_fits = self.get_arrays_fits(path)
        
        I = [data_fits.field(2*i) for i in range(len(data_fits[0])/2)]     
        Q = [data_fits.field(2*i+1) for i in range(len(data_fits[0])/2)]  

        return I,Q    

    def psd(self, df, Fs):
        psd = [signal.periodogram(df[i], Fs)[1]  for i in range(len(df))]
        freqs = signal.periodogram(df[0],Fs)[0]

        psd_mean = 10*np.log10(np.average(psd,axis=0))

        return freqs, psd, psd_mean

    def get_homodyne_psd(self, path, didf, dqdf, check):

        I0,Q0,Fs,actual_temp,kid_number,input_att,didf_p,dqdf_p,f0 = self.get_header(path)

        if check == False:
            didf = didf_p
            dqdf = dqdf_p

        I,Q = self.get_IQ_data_homo(path)
        df,vel = self.df(I0,Q0,didf,dqdf,I,Q)

        freqs, psd, psd_mean = self.psd(df,Fs)

        step = 1/Fs
        df_avg = np.average(df,axis=0)
        I_avg = np.average(I,axis=0)
        Q_avg = np.average(Q,axis=0)
        time = np.arange(0,step*len(I_avg),step)

        return freqs, psd, psd_mean, time, I_avg, Q_avg, df_avg, kid_number, actual_temp, input_att

    def get_all_data(self, path):

        try:
            sweep_path, ONTemp, OFFTemp = self.get_each_path(path) 
            
            path_off_up, path_off_low = self.split_array(OFFTemp)
            path_on_up, path_on_low = self.split_array(ONTemp)

            freq, sweep_i, sweep_q = self.get_vna_sweep(os.path.join(path, sweep_path))

            mag,f0,I0,Q0,didf,dqdf,check = self.get_vna_sweep_parameters(freq,sweep_i,sweep_q)

            f0_fits = self.get_header(os.path.join(path, path_on_up))[8]

            psd = self.get_homodyne_psd(os.path.join(path, path_on_up),didf,dqdf,check)
            psd_low = self.get_homodyne_psd(os.path.join(path, path_on_low),didf,dqdf,check)
            psd_OFF = self.get_homodyne_psd(os.path.join(path, path_off_up),didf,dqdf,check)
            psd_low_OFF = self.get_homodyne_psd(os.path.join(path, path_off_low),didf,dqdf,check)
            
            return freq, sweep_i, sweep_q, psd, psd_low, psd_OFF, psd_low_OFF, f0, f0_fits                
        except:
            return -1

    # Choose the newest version of the fits files
    def split_array(self, array):
        low = []
        up = []
        a1 = []
        a2 = []

        ref = array[0][1]
        
        for m in array:
            if m[1] == ref:
                a1.append(m)
            else:
                a2.append(m)

        if a1[0][1] >= a2[0][1]:
            up = a1
            low = a2
        else:
            up = a2
            low = a1

        aux = 0
        for n in low:
            if n[2] >= aux:
                aux = n[2]
                low_path = n[0]

        aux = 0
        for m in up:
            if m[2] >= aux:
                aux = m[2]
                up_path = m[0]

        return up_path, low_path

    # Get the data of the directory
    def get_each_path(self, diry):

        files = os.listdir(diry)
        sweep_path = ""
        ONTemp = []
        OFFTemp = []

        for i in files:
            k = i.lower()
            if k == "sweep.fits":
                sweep_path = i
            else:
                freq = ''
                mode = ''
                n = ''
                cnt = 0
                for j in k:
                    if j == '_':
                        cnt += 1
                    elif j == '.':
                        cnt = 0
                    elif cnt == 1:
                        freq = freq + j
                    elif cnt == 3:
                        mode = mode + j
                    elif cnt == 6:
                        n = n + j
                if n == '':
                    n = 0
                if mode == "on":
                    ONTemp.append([i,int(freq),int(n)])
                elif mode == "off":
                    OFFTemp.append([i,int(freq),int(n)])

        return sweep_path, ONTemp, OFFTemp
