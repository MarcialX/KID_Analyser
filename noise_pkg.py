# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import curve_fit

def fit_kid_psd(psd_freqs, psd_df, kid_f0, kid_qr,
                gr_guess=None, tauqp_guess=None, amp_guess=None, tlsa_guess=None,  tlsb_guess=-1.5,
                gr_min=0,      tauqp_min=0,      amp_min=0,      tlsa_min=-np.inf, tlsb_min=-1.501,
                gr_max=np.inf, tauqp_max=np.inf, amp_max=np.inf, tlsa_max=np.inf,  tlsb_max=-1.499,
                sigma = None):

    def combined_model(freqs,gr_noise,tau_qp,amp_noise,tls_a,tls_b):
            gr = gr_noise/(1.+(2*np.pi*freqs*tau_qp)**2) / (1.+(2*np.pi*freqs*kid_qr/np.pi/kid_f0)**2)
            tls = tls_a*freqs**tls_b / (1.+(2*np.pi*freqs*kid_qr/np.pi/kid_f0)**2)
            amp = amp_noise
            return gr + tls + amp

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

    pval, pcov = curve_fit(combined_model, psd_freqs, psd_df, guess, bounds=bounds, sigma=sigma)

    (gr_noise,tau_qp,amp_noise,tls_a,tls_b) = pval

    return (gr_noise,tau_qp,amp_noise,tls_a,tls_b,f0,qr)
