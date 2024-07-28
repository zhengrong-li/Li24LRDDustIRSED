from abc import ABC, abstractmethod

import numpy as np
from scipy import integrate, special
import astropy.units as u
import astropy.constants as const


from .utils import LogLogInterpolator, trapz_log, quad_vec_log
from .opacity import OpacityData

class R_out_Error(Exception):
    pass


def Planck_B_nu(nu, T):
    """Planck B_nu function. Pure number (without unit), all values in cgs units.
    """
    h = const.h.cgs.value
    c = const.c.cgs.value
    k_B = const.k_B.cgs.value
    x = h * nu / (k_B * T)
    return (2 * h / c**2) * nu**3 / (np.exp(x) - 1)


class LRD_IR_ModelBase(ABC):
    paras_name = ['n_0', 'gamma', 'L_UV', 'T_sub']
    def __init__(
        self,
        n_0: float,
        gamma: float,
        L_UV: float,
        T_sub: float,
        NH_target: float | None,
        opacity: OpacityData,
    ):
        """
        Initialize the model with these parameters. The units are in cgs unless specified otherwise.
        r_in is in [pc]
        
        These input paras should NOT be changed after initialization
        """
        self.n_0 = n_0
        self.gamma = gamma
        self.L_UV = L_UV
        self.T_sub = T_sub
        self.NH_target = NH_target
        self.opacity = opacity
        self.r_in = self._calc_r_in()   # in [pc]
    

    @property
    def paras(self):
        """get parameters in the model"""
        return {self.__dict__[name] for name in self.paras_name}

    def __repr__(self):
        return f"{self.__class__.__name__}(n_0={self.n_0}, gamma={self.gamma}, L_UV={self.L_UV}, T_sub={self.T_sub}, r_in={self.r_in})"

    def _repr_latex_(self):
        return rf"""{self.__class__.__name__}($n_0=${self.n_0:.2e}, $\gamma=${self.gamma}, 
    $L_{{\rm UV}}=${self.L_UV:.2e}, $T_{{\rm sub}}=${self.T_sub}, 
    $r_{{\rm in}}=${self.r_in:.2e}, $r_{{\rm out}}=${self.r_out:.2e}, $T_{{\rm out}}=${self.T_out:.1f})
    """

    def __format__(self, format_spec: str) -> str:
        """usage: ` f"{model:latex}" `
        """
        if format_spec == 'latex':
            return self._repr_latex_()
        else:
            return super().__format__(format_spec)

    @abstractmethod
    def _calc_r_in(self):  # in [pc]
        pass

    def n_profile(self, r):
        r_ratio = r / self.r_in
        return self.n_0 * r_ratio**(-self.gamma)

    def NH_profile(self, r):
        """Column density profile in cgs"""
        gamma = self.gamma  # for brevity
        r_ratio = r / self.r_in
        r_in_cgs = self.r_in * (u.pc.to(u.cm))
        if gamma == 1:  # special case: gamma == 1
            factor = np.log(r_ratio)
        else:
            factor = (r_ratio**(1 - gamma) - 1) / (1 - gamma)
        return self.n_0 * r_in_cgs * factor

    @property
    def r_out(self):
        if not hasattr(self, '_r_out'):
            self._r_out = self._calc_r_out()
        return self._r_out
    
    @r_out.setter
    def r_out(self, value):
        self._r_out = value
    
    @r_out.deleter
    def r_out(self):
        del self._r_out

    def _calc_r_out(self):
        """calc the r_out that could give the specified NH"""
        NH_target = self.NH_target
        if NH_target is None:
            raise ValueError("NH_target is not specified yet!")
        gamma = self.gamma
        r_in_cgs = self.r_in * (u.pc.to(u.cm))
        factor = NH_target / (self.n_0 * r_in_cgs)  #* if NH_target is None, this will raise an error
        if gamma > 1 and factor > 1/(gamma-1):
            #! when gamma > 1, the maximum of NH is n_0*r_in_cgs / (gamma-1). If NH_target > this value, r_out cannot be found and will give negative or complex value.
            raise R_out_Error(f"The {NH_target = } is larger than possible in this NH_profile with {gamma = }, cannot find r_out")

        if gamma == 1:
            r_out_ratio = np.exp(factor)
        else:
            r_out_ratio = (factor * (1-gamma) + 1) ** (1/(1-gamma))
        return self.r_in * r_out_ratio


    def T_dust_power_law(self, p, r):
        return self.T_sub * (r / self.r_in) ** (-p)

    @abstractmethod
    def T_dust_profile(self, r):
        """Temperature profile T(r)
        """
        pass

    @property
    def T_in(self):
        return self.T_dust_profile(self.r_in)

    @property
    def T_out(self):
        return self.T_dust_profile(self.r_out)

    method_L_nu = 'trapz_log'  # default method for calc_L_nu
    def calc_L_nu(self, nu_array: u.Quantity | None = None, r_sample_num:int = 1000):
        """calc L_nu at given frequency array, using the given opacity data

        Parameters
        ----------
        nu_array : u.Quantity | None, optional
            If not specified (default = None), will use opacity.nu.

        Returns
        -------
        L_nu : u.Quantity
            L_nu corresponding to nu_array。in cgs unit.
        """
        if nu_array is None:  # if nu_array is not given, use the nu array in opacity
            sigma_H = self.opacity.sigma_H_abs.cgs.value  # then, sigma_H directly from data, don't need interpolation
            nu_array = self.opacity.nu
        else:
            sigma_H = self.opacity.interp_abs(nu_array).cgs.value
            nu_array = nu_array.to(u.Hz)
        nu_cgs = nu_array.cgs.value

        def func(nu, r):
            return Planck_B_nu(nu, self.T_dust_profile(r)) * self.n_profile(r) * 4*np.pi * r**2

        if self.method_L_nu == 'quad':
            L_nu = integrate.quad_vec(lambda r: func(nu_cgs, r), self.r_in, self.r_out)[0]
        elif self.method_L_nu == 'quad_log':
            L_nu = quad_vec_log(lambda r: func(nu_cgs, r), self.r_in, self.r_out)[0]
        elif self.method_L_nu == 'trapz':
            r_array = np.linspace(self.r_in, self.r_out, r_sample_num)   # this method should not be used
            integrand_array = func(nu_cgs[:, None], r_array)
            L_nu = integrate.trapezoid(integrand_array, r_array)
        elif self.method_L_nu == 'trapz_log':
            r_array = np.geomspace(self.r_in, self.r_out, r_sample_num)  # sample in log scale
            integrand_array = func(nu_cgs[:, None], r_array)
            L_nu = trapz_log(integrand_array, r_array)
        else:
            raise ValueError(f"method {self.method_L_nu} is invalid! ")

        L_nu *= 4*np.pi * sigma_H

        return L_nu

    def calc_L_nu_photon(self, nu_array: u.Quantity | None = None, r_sample_num:int = 1000):
        """calc L_nu at given frequency array
        This version is for considering the feedback: all the dust inside r_photosphere is accumalted at the thin shell at r_photosphere.
        We assume their temperature is np.e**(1/5.6) times higher than the dust temperature at tau_UV = 1 (eq1)
        Parameters
        ----------
        nu_array : u.Quantity | None, optional
            If not specified (default = None), will use opacity.nu.

        Returns
        -------
        L_nu : u.Quantity
        """
        y_out_Orion = self.r_out/self.r_in
        if self.gamma == 1:
            y_photosphere = np.exp(np.log(y_out_Orion)*(1/4.8176)) # magic number: 4.8176 from the Orion data
        else:
            y_photosphere = ((1/4.8176)*(y_out_Orion**(1-self.gamma)-1)+1)**(1/(1-self.gamma))
        r_photosphere = self.r_in*y_photosphere
        if nu_array is None:  # if nu_array is not given, use the nu array in opacity
            sigma_H = self.opacity.sigma_H_abs.cgs.value  # then, sigma_H directly from data, don't need interpolation
            nu_array = self.opacity.nu
        else:
            sigma_H = self.opacity.interp_abs(nu_array).cgs.value
            nu_array = nu_array.to(u.Hz)
        nu_cgs = nu_array.cgs.value

        def func(nu, r):
            return Planck_B_nu(nu, self.T_dust_profile(r)) * self.n_profile(r) * 4*np.pi * r**2

        def func_photosphere(nu, r, r_photosphere):
            return Planck_B_nu(nu, (np.e**(1/5.6)) * self.T_dust_profile(r_photosphere)) * self.n_profile(r) * 4*np.pi * r**2
        if self.method_L_nu == 'quad':
            L_nu = integrate.quad_vec(lambda r: func(nu_cgs, r), self.r_in, self.r_out)[0]
        elif self.method_L_nu == 'quad_log':
            L_nu = quad_vec_log(lambda r: func(nu_cgs, r), self.r_in, self.r_out)[0]
        elif self.method_L_nu == 'trapz':
            r_array = np.linspace(r_photosphere, self.r_out, r_sample_num)
            integrand_array = func(nu_cgs[:, None], r_array)
            L_nu = integrate.trapezoid(integrand_array, r_array)
        elif self.method_L_nu == 'trapz_log':
            r_array = np.geomspace(r_photosphere, self.r_out, r_sample_num)  # sample in log scale
            integrand_array = func(nu_cgs[:, None], r_array)
            L_nu = trapz_log(integrand_array, r_array)

            r_inner_array = np.geomspace(self.r_in, r_photosphere, r_sample_num)
            integrand_inner_array = func_photosphere(nu_cgs[:, None], r_inner_array,r_photosphere=r_photosphere)
            L_nu_inner = trapz_log(integrand_inner_array, r_inner_array)
            L_nu += L_nu_inner
        else:
            raise ValueError(f"method {self.method_L_nu} is invalid! ")

        L_nu *= 4*np.pi * sigma_H
        L = -1*trapz_log(L_nu, nu_cgs)
        L_nu = L_nu/L*self.L_UV
        return L_nu
