# -*- coding: utf-8 -*-
"""
Spectral noise fitting and NEP calculation.

Contains routines to get quasiparticle lifetime tqp, density of quasiparticle nqp
and NEP g-r.

"""

import numpy as np
from scipy.optimize import curve_fit

class fit_spectral_noise():

    """
    Get the model which match the PSD noise.
    A hundred percent based in PhD Sam Rowe code
    """
    def combined_model(self,freqs,gr_noise,tau_qp,amp_noise,tls_a,tls_b):
        # Ruido Generación-Recombinación
        gr = gr_noise/(1.+(2*np.pi*freqs*tau_qp)**2) / (1.+(2*np.pi*freqs*kid_qr/np.pi/kid_f0)**2)
        # Ruido TLS
        tls = tls_a*freqs**tls_b / (1.+(2*np.pi*freqs*kid_qr/np.pi/kid_f0)**2)
        # Ruido del amplificador
        amp = amp_noise

        # Ruido Total
        return gr + tls + amp

    def fit_kid_psd(self,psd_freqs, psd_df, kid_f0, kid_qr,
                    gr_guess=None, tauqp_guess=None, amp_guess=None, tlsa_guess=None,  tlsb_guess=-1.5,
                    gr_min=0,      tauqp_min=0,      amp_min=0,      tlsa_min=-np.inf, tlsb_min=-1.501,
                    gr_max=np.inf, tauqp_max=np.inf, amp_max=np.inf, tlsa_max=np.inf,  tlsb_max=-1.499,
                    sigma = None):

        if gr_guess is None:
            gr_guess = 0.01
        if tauqp_guess is None:
            tauqp_guess = 1./(psd_freqs.max()-psd_freqs.min())
        if amp_guess is None:
            amp_guess = psd_df[-1]
        if tlsa_guess is None:
            tlsa_guess = psd_df[-1]
        if tlsb_guess is None:
            tlsb_guess = 1.5

        guess = np.array([gr_guess, tauqp_guess, amp_guess, tlsa_guess,  tlsb_guess ])
        bounds = np.array([ [gr_min, tauqp_min, amp_min, tlsa_min,  tlsb_min ],
                            [gr_max, tauqp_max, amp_max, tlsa_max,  tlsb_max ]])

        if sigma is None:
            sigma = (1 / abs(gradient(psd_freqs)))

        pval, pcov = curve_fit(self.combined_model, psd_freqs, psd_df, guess, bounds=bounds, sigma=sigma)
        (gr_noise,tau_qp,amp_noise,tls_a,tls_b) = pval

        return (gr_noise,tau_qp,amp_noise,tls_a,tls_b,f0,qr)

    def spectral_noise(self, freq, Nqp, tqp):
        """
        Expression for the calculus of Spectral Noise

        Parameters
        ----------
        Nqp : float number
            Number of quasiparticles
        tqp : float number
            Lifetime of quasiparticles
        freq : array
            Frequency of the spectral plot
        Returns
        -------
        Sn : array
            Spectral noise

        References
        -------
        [1] P. J. de Visser et al. J. Low Temperature Physics 2012
        """
        w = 2*np.pi*freq
        return (4*Nqp*tqp)/(1 + (w*tqp)**2)

    def fitPSD(self, freq, psd, bounds=None):
        """Function to fit a skewed Lorentzian to a resonator sweep. Uses scipy.optimize.curve_fit
        to perform least squares minimization.

        Parameters
        ----------
        freq : array_like
            Array containing frequency values of spectral noise
        psd : array_like
            Array containing spectral noise values

        Returns
        -------
        fitoutput : tuple
            A tuple containing three arrays;
                - popt; parameters found from least-squares fit. [Nqp, tqp]
                - perr; an estimate of the errors associated with the fit parameters
                - fit; the fit function evaluated using the fitting parameters
        """

        weights = np.ones_like(freq)

        init = np.array([0, 0])

        if bounds is not None:
			param_bounds = bounds
        else:
            param_bounds = np.array([np.array([0,0]), np.array([np.inf, 10])])

        PSDfitp, PSDcov = curve_fit(self.spectral_noise, freq, psd, p0=init, bounds = param_bounds,sigma=weights)

        errs = np.sqrt(np.diag(PSDcov))
        fitPSD = self.spectral_noise(freq, *PSDfitp)

        Nqp = PSDfitp[0]
        tqp = PSDfitp[1]

        errNqp = errs[0]
        errtqp = errs[1]

        return np.array([Nqp, tqp]), np.array([errNqp, errtqp]), fitPSD
