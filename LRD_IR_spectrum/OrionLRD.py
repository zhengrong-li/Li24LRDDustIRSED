from typing import override

import numpy as np
from scipy import integrate, optimize
import astropy.units as u

from .model_base import Planck_B_nu

from .utils import LogLogInterpolator, quad_vec_log, trapz_log
from .opacity import OpacityData
from .incident_SED import IncidentSED
from .model_base import LRD_IR_ModelBase


class SemiOrionLRDModel(LRD_IR_ModelBase):

    T_floor = 10  # Kelvin, set T_floor to avoid too large R_out
    T_accuracy = 1.0

    def __init__(
        self,
        n_0: float,
        gamma: float,
        L_UV: float,
        T_sub: float,
        NH_target: float | None,
        opacity: OpacityData,
    ):
        super().__init__(n_0, gamma, L_UV, T_sub, NH_target, opacity=opacity)
        self.sigma_H_UV_ext = self.opacity.sigma_H_ext.max().cgs.value
        self.sigma_H_UV_abs = self.opacity.sigma_H_abs.max().cgs.value

    def tau_UV_profile(self, r):
        """Optical depth profile"""
        return self.sigma_H_UV_ext * self.NH_profile(r)


    def UV_Flux(self, r):
        """
        r can be scalar or numpy array.
        """
        tau = self.tau_UV_profile(r)
        return self.L_UV / (4 * np.pi * (r * u.pc.to(u.cm))**2) * np.exp( -tau ) * self.sigma_H_UV_abs


    def IR_Flux(self, T, method:str = 'trapz_log'):
        """
        T can be scalar or numpy array.
        """
        nu_cgs = self.opacity.nu_cgs

        def integrand(nu, T):
            return self.opacity.interp_abs(nu * u.Hz).cgs.value * Planck_B_nu(nu, T)

        if method == 'quad':
            flux = integrate.quad_vec(lambda nu: integrand(nu, T), nu_cgs.min(), nu_cgs.max())[0]
        elif method == 'quad_log':
            flux = quad_vec_log(lambda nu: integrand(nu, T), nu_cgs.min(), nu_cgs.max())[0]
        elif method == 'trapz':
            flux = integrate.trapezoid(integrand(nu_cgs[:, None], T, ), nu_cgs, axis=0)
        elif method == 'trapz_log':
            flux = trapz_log(integrand(nu_cgs[:, None], T, ), nu_cgs[:, None], axis=0)
        else:
            raise ValueError(f"method {method} is invalid! ")

        flux *= 4*np.pi

        if flux.size == 1:
            flux = flux.item() # convert to scalar if possible
        return flux

    def _T_dust_eqn(self, r, T, **kwargs):
        return self.UV_Flux(r) - self.IR_Flux(T, **kwargs)


    def _calc_r_in(self):
        IR_Flux = self.IR_Flux(self.T_sub)
        return np.sqrt(self.L_UV * self.sigma_H_UV_abs / (4 * np.pi * IR_Flux)) * u.cm.to(u.pc)
    

    def T_dust_profile_brentq(self, r):
        if isinstance(r, (int, float)):
            return optimize.brentq(lambda T: self._T_dust_eqn(r, T), 0, self.T_sub, xtol=self.T_accuracy)
        else:
            return np.array([self.T_dust_profile_brentq(r) for r in r])

    def T_dust_profile(self, r):
        T_array = np.linspace(1, self.T_sub, int((self.T_sub - 1) / self.T_accuracy) + 1)
        T_array_low = np.geomspace(1e-10, 1, 10, endpoint=False)   # to avoid log(0)
        T_array = np.concatenate([T_array_low, T_array])
        interp = LogLogInterpolator(self.IR_Flux(T_array), T_array)
        ret = interp(self.UV_Flux(r))
        return np.maximum(ret, self.T_floor)
    
    
    
class OrionLRDModel(SemiOrionLRDModel):
    
    def __init__(
        self,
        n_0: float,
        gamma: float,
        L_UV: float,
        T_sub: float,
        NH_target: float | None,
        opacity: OpacityData,
        incident_SED: IncidentSED
    ):
        self.incident_SED = incident_SED
        LRD_IR_ModelBase.__init__(self, n_0=n_0, gamma=gamma, L_UV=L_UV, T_sub=T_sub, NH_target=NH_target, opacity=opacity)
        
    def tau_nu_profile(self, nu, r):
        """Optical depth profile"""
        return self.opacity.interp_ext(nu).cgs.value * self.NH_profile(r) 
    
    @override
    def UV_Flux(self, r):
        """
        r can be scalar or numpy array.
        """
        
        r = np.array(r)
        incident_SED = self.incident_SED
        nu_array = incident_SED.nu
        sigma_H = self.opacity.interp_abs(nu_array).cgs.value
        
        r_arr = r[:, None] if np.size(r) > 1 else r
        tau = self.tau_nu_profile(nu_array, r_arr)
            
        return 1/(4*np.pi * (r * u.pc.to(u.cm))**2) * trapz_log(incident_SED.L_nu * sigma_H * np.exp(-tau), nu_array.cgs.value)
    
    @override
    def _calc_r_in(self):
        IR_Flux = self.IR_Flux(self.T_sub)
        
        incident_SED = self.incident_SED
        nu_array = incident_SED.nu
        sigma_H = self.opacity.interp_abs(nu_array).cgs.value

        return np.sqrt(trapz_log(incident_SED.L_nu * sigma_H, nu_array.cgs.value) / (4 * np.pi * IR_Flux)) * u.cm.to(u.pc)