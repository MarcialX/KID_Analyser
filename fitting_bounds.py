# -*- coding: utf-8 -*-
"""
Resonator/KID fitting routines.

Contains rountines for fitting to resonance curves, noise data and pulse decays

Taken from Pete Barry code
"""

import numpy as np


from numpy import argmin, array, diag, sqrt, log10
from scipy.optimize import curve_fit
from matplotlib.pyplot import subplot

from numpy import gradient

class fitting_resonators():

    def s21skewed(self,freq, F0, Qr, Qcre, Qcim, A):
        """
        Expression for a skewed Lorentzian lineshape used for fitting resonance data.

        Parameters
        ----------
        freq : array_like
            Array containing frequency values for the sweep
        F0   : number
            Resonant frequency in Hz
        Qr   : number
            Total resonator quality factor
        Qcre : number
            Coupling quality factor of the resonator
        Qcim : number
            Imaginary part of the coupling quality factor. Accounts for degree of
            asymmetry found in resonance curves. See [1] for more details.
        A    : number
            Baseline level.

        Returns
        -------
        s21skewed : array
            Returns array of len(freq) containing the absolute value of the transmission
            response S21.

        References
        -------
        [1] M. S. Khalil et al. J. Appl. Phys., vol. 111, no. 5, p. 054510, 2012.
        """
        x = (freq-F0)/F0
        return abs(A*(1 - Qr/(Qcre + 1j*Qcim)/(1 + 2j*Qr*x)))

    def cardan(self,a,b,c,d):
        """
        Analytically solve equations of form ax^3 + bx^2 + cx + d = 0 based on the Cardan formulae [1].
        This specific python implementation is shamelessly borrowed from [2] and adapted to make it
        vector compatible.

        Parameters
        ----------
        a,b,c,d : number, array_like
            Coefficients of the third order polynomial.

        Returns
        -------
        roots : tuple
            Returns tuple containing the three, possibly complex, roots.

        References
        -------
        [1] https://fr.wikipedia.org/wiki/M%C3%A9thode_de_Cardan#Formules_de_Cardan)
        [2] https://goo.gl/C4VrbX
        """

        J  = np.exp(2j*np.pi/3)
        Jc = 1/J
        u  = np.empty(2, np.complex128)
        
        z0=b/3/a
        a2,b2 = a*a,b*b
        p = -b2/3/a2 + c/a; q = (b/27*(2*b2/a2 - 9*c/a) + d)/a
        D = -4*p*p*p - 27*q*q
        r = np.sqrt(-D/27+0j); u=((-q-r)/2)**(1./3); v=((-q+r)/2)**(1./3); w=u*v
        w0 = abs(w + p/3); w1 = abs(w*J + p/3); w2 = abs(w*Jc + p/3)
        #original logic code (not vectorisable)
        if type(b) in [float, np.float32, np.float64]:
            if w0<w1:
                if w2<w0 : v*=Jc
            elif w2<w1 : v*=Jc
            else: v*=J
        else:
            #Attempt at code to vectorise for performance increase
            i0 = (w0 < w1); i1 = i0 & (w2 < w0); i2 = (w2 < w1) & ~i0
            v[ i1 | i2 ] *= Jc
            v[~(i0 | i2)] *= J # equivalent as else

        return u + v - z0, u*J + v*Jc - z0, u*Jc + v*J - z0


    def s21skewed_nonlinear(self, freq, F0, Qr, Qcre, Qcim, A, a):
        """
        Expression for a skewed Lorentzian lineshape [1] that includes a non-linearity term [2]
        used for fitting resonance data.

        Parameters
        ----------
        freq : array_like
            Array containing frequency values for the sweep
        F0   : number
            Resonant frequency in Hz
        Qr   : number
            Total resonator quality factor
        Qcre : number
            Coupling quality factor of the resonator
        Qcim : number
            Imaginary part of the coupling quality factor. Accounts for degree of
            asymmetry found in resonance curves. See [1] for more details.
        A    : number
            Baseline level.
        a    : number
            The degree of non-linearity. a is currently subject to the bound 0 < a < 4*sqrt(3)/9,
            i.e. this function does work for fully bifurcated resonances.

        Returns
        -------
        s21skewed_nonlinear : array
            Returns array of len(freq) containing the absolute value of the transmission
            response S21, including the non-linear behaviour.

        References
        -------
        [1] M. S. Khalil et al. J. Appl. Phys., vol. 111, no. 5, p. 054510, 2012.
        [2] L. J. Swenson et al. J. Appl. Phys., vol. 113, no. 10, p. 104501, 2013.
        """
        y0 = Qr*(freq-F0)/F0
        # find roots of 4y^3+ -4y0 * y^2 + y -(y0 + a) = 0. Used many attempts at trying to speed this up.
        #roots = np.array([np.roots( np.array([4.0, -4.0*yi, 1.0, -(yi + a)]) ) for yi in y0])
        #roots = np.array([cardan( * [4.0, -4.0*yi, 1.0, -(yi + a)] ) for yi in y0])
        roots = np.array(self.cardan( 4.0, -4.0*y0, 1.0, -(y0 + a) )).T

        # Extract real roots only
        y = roots[np.where(abs(roots.imag) < 1e-5)].real
        #print roots.shape
        #print roots
        #print 
        #print
        #print y.shape
        #print y
        return abs(A*(1 - Qr/(Qcre + 1j*Qcim)/(1 + 2j*y)))


    def fitmags21(self, freq, s21, approxQr = 1.e5,  nonlinear=False, normalise=False, db=False, make_plot=False,ax=None,bounds=None):
        """Function to fit a skewed Lorentzian to a resonator sweep. Uses scipy.optimize.curve_fit
        to perform least squares minimization.

        Parameters
        ----------
        freq : array_like
            Array containing frequency values for the sweep
        s21 : array_like
            Array containing s21 data in the form I + 1j*Q
        approxQr : number, optional
            Approximate value for the expected value of Qr to be used as the initial
            guess for the fit. Experience suggests that convergence of the fit is a
            soft function of this parameter. Default value = 1.e5
        nonlinear : bool, optional
            If True, use the non-linear skewed model for S21. Takes ~10x longer due to the
            additional root finding step. See documentation for s21skewed_nonlinear for more details.

        Returns
        -------
        fitoutput : tuple
            A tuple containing three arrays;
                - popt; parameters found from least-squares fit. [F0, Qr, Qc, Qi, a]
                - perr; an estimate of the errors associated with the fit parameters
                - fit; the fit function evaluated using the fitting parameters

        """

        # guess = [f0, Qr, Qc_re, Qc_im, offset, a] # note that starting values matter, Qr should be less than Qc.

        abss21=abs(s21)
        di=gradient(s21.real)
        dq=gradient(s21.imag)
        #weights = sqrt(1./(sqrt(di**2+dq**2)))
        #weights = lowpass_cosine.lowpass_cosine(sqrt(1./(sqrt(di**2+dq**2))),1.0,0.1,0.1)
        #weights = sqrt(abss21)
        #weights = 1./sqrt(abss21)
        weights = np.ones_like(freq)
        
        if nonlinear==True:
            init = np.array([freq[np.argmin(abss21)], approxQr, approxQr*2, 1.0, abss21.max(), 0.01])
            #param_bounds = np.array([[0.,0.,0.,0.,0.,0.],[np.inf,np.inf,np.inf,np.inf,np.inf,np.around(4*np.sqrt(3)/9, 3)]])
            if bounds is not None:
    			param_bounds = bounds
            else:
    			param_bounds = np.array([np.array([1e-9]*2+[-np.inf]*2+[1e-9]*2), np.array([np.inf]*5 + [np.around(4*np.sqrt(3)/9, 6)] )]) # set the parameter bounds
    			#param_bounds = np.array([np.array([0.]*6), np.array([np.inf]*6)] ) # set the parameter bounds
            
            for p in range(6):
    			if init[p] < param_bounds[0][p]:
    				init[p] = param_bounds[0][p]
    			elif init[p] > param_bounds[1][p]:
    				init[p] = param_bounds[1][p]
    				
            s21fitp, s21cov = curve_fit(self.s21skewed_nonlinear, freq, abss21, p0=init, bounds = param_bounds,sigma=weights)
            #s21fitp, s21cov = curve_fit(s21skewed_nonlinear, freq, abs(s21), p0=init, bounds = param_bounds,sigma=weights)

            errs = np.sqrt(diag(s21cov))
            fits21 = self.s21skewed_nonlinear(freq, *s21fitp)
            a = s21fitp[-1]

        else:
            init = np.array([freq[argmin(abss21)], approxQr/2., approxQr, 1., abss21.max()])
            param_bounds = np.array([np.array([1e-9]*5), np.array([np.inf]*5)]) # set the parameter bounds)

            s21fitp, s21cov = curve_fit(self.s21skewed, freq, abss21, p0=init, bounds = param_bounds,sigma=weights)
            #s21fitp, s21cov = curve_fit(s21skewed, freq, abs(s21), p0=init, bounds = param_bounds,sigma=weights)
            errs = np.sqrt(np.diag(s21cov))
            fits21 = self.s21skewed(freq, *s21fitp)
            a = 0.  # need to add error for a here

        f0, Qr = s21fitp[0:2]
        Qe = s21fitp[2] + 1j*s21fitp[3]
        A = s21fitp[4]

        Qc = abs(abs(Qe)**2/Qe.real)
        Qi = 1./(1/Qr - 1/Qc)

        errf0, errQr, errQere, errQeim = errs[0:4] # error in f0, Qr, Qe_re, Qe_im
        errQc = sqrt(errQere**2 + (2 * errQeim * Qe.imag/Qe.real)**2 + errQere*Qe.imag**2/Qe.real**2)
        errQi = sqrt(Qi**4*(errQr**2/Qr**4 + errQc**2/Qc**4))
        errA  = errs[4]
        erra  = errs[-1]

        if make_plot:
            if ax is None: axs21 = subplot(1,1,1)
            else: axs21 = ax
            if normalise: fits21 = fits21/s21fitp[-1]; s21 = abs(s21/s21fitp[-1])
            elif db: fits21 = 20*log10(fits21); s21 = 20*log10(abss21)
            axs21.plot((freq-f0)/1.e3, s21, label='Data'); axs21.plot((freq-f0)/1.e3, fits21, 'r--', lw=2, label='Fit'); #axs21.axvline(f0-f0, ls = '--', color = 'k', lw=1)
            axs21.set_xlabel('Frequency offset (kHz)'); axs21.set_ylabel(u'|S21|\u00B2 (dB)')
            axs21.legend(loc='lower right'); axs21.grid()
            axs21.text(0.1, 0.15, 'Qr = %.1f\nQc = %.1f\nQi = %.1f\n'%(Qr, Qc, Qi), transform=axs21.transAxes)
            #print 'F0 = %.1f ± %d Hz\tQr = %d ± %03d\tQc = %d ± %03d\tQi = %d ± %d'%(f0, errf0, abs(Qr), errQr, abs(Qc), errQc, abs(Qi), errQi)

        return array([f0, Qr, Qc, Qi, A, a]), array([errf0, errQr, errQc, errQi, errA,erra]), fits21