#!/usr/bin/env python
# -*- coding: utf-8 -*-
#************************************************************
#*                    KID-ANALYSER                          *
#*                FIT CURVE. LORENTZIAN                     *
#*                   13/agosto/2018                         *
#************************************************************

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

class get_Q_factor():
	def __init__(self):
		print "Q factor loaded"

	def lorentz(self, x, I, gamma, x0):
		return I * gamma**2 / ((x - x0)**2 + gamma**2)

	def fit(self, x, y):
		popt, pcov = curve_fit(self.lorentz, x, y, bounds=([np.min(y), 0, np.min(x)], [np.max(y), np.max(x) - np.min(x), np.max(x)]))

		fit_y = self.lorentz(x, *popt)
		f_3db = x[np.argmin(np.abs(fit_y - (np.max(fit_y) - 3)))]
		
		# Frecuencia central
		f0 = popt[2]

		# Ancho de banda
		bw = np.abs(f0 - f_3db)*2

		# Factor Q
		Q = f0/bw

		return popt[0], popt[1], f0, bw, Q