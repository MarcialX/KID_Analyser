import os

import numpy as np
import matplotlib.pyplot as plt

from dataRed import dataRed
from detector_peaks import detect_peaks
from scipy import signal
from scipy.signal import savgol_filter

def delBaseLine(freq,mag,thresh,p):
	baseline = []

	freqb = []

	for i in range(len(freq)-p):
		slocal = np.std(mag[i:p+i-1])
		mlocal = np.mean(mag[i:p+i-1])

		if not mag[p+i] < thresh*slocal + mlocal:
			baseline.append(mag[p+i])
			freqb.append(freq[p+i])	

	return baseline, freqb

path = "/home/muscat/Documents/VNA_meas/meas_121018/INAOE_INAOE_C4/20181012_Dark_Data_Prelim"

dataRedtn = dataRed()

sweep = []

vnaPath = path + "/VNA_Sweeps/"
files = os.listdir(vnaPath)

for file in files:

	# Delete the continuous
	freq, mag = dataRedtn.get_full_vna(str(vnaPath + file))
	base = savgol_filter(mag, 50001, 3)	
	mag_wBL = mag - base

	# --------- PLOT DEBUG ----------
	plt.plot(freq,mag,'r*-')
	# -------------------------------

	mag_wBL = savgol_filter(mag_wBL,51,3)
	mag_wBL = -1*mag_wBL

	# Parameters MPH and MPD 
	mph = np.max(mag_wBL)/8
	mpd = 0.5*(freq[-1] - freq[0])/len(freq)

	# Index of peaks detectors
	ind = detect_peaks(signal.detrend(mag_wBL),mph=mph,mpd=mpd)

	sweep.append(ind)
	print len(ind)

	for i in ind:
	    plt.axvline(freq[i],color='r',linewidth=0.75)  
	    #plt.plot(freq[i],mag[i],'bo')
	    #plt.annotate(r"$"+str(freq[i]/1e6)+"MHz$",xy=(freq[i],mag[i]))

#shift = np.abs(sweep[1] - sweep[0])

#print len(shift)

#plt.plot(sweep[1],shift,'co-')
#plt.xlabel('F0 without hornblock [Hz]')
#plt.ylabel('Shift frequency [Hz]')

plt.show()