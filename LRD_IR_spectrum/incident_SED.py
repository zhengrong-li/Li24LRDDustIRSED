import numpy as np
import astropy.units as u
import astropy.constants as const

from .utils import LogLogInterpolator

class IncidentSED:
    def __init__(self, filename: str):
        #* the incident SED data need to be in the format of wavelength (Angstrom) and L_lambda (erg/s/Angstrom)
        data = np.loadtxt(filename, comments='#', delimiter=' ', usecols=(0, 1))
        #* nu increasing ,wavelength decreasing
        self.wavelength = data[::-1, 0] * u.AA
        L_lambda = data[::-1, 1]

        self.nu = self.wavelength.to(u.Hz, u.spectral())  # Hz
        L_nu = L_lambda * self.wavelength.cgs.value**2 / const.c.cgs.value

        L_bol = 1e46 # erg/s
        NormalizedFactorAt1450 = L_bol/4.4/2.0675e+15 # erg/s/Hz

        # L_nu is pure value without unit
        self.L_nu = L_nu / LogLogInterpolator(self.wavelength, L_nu)(1450 * u.AA) * NormalizedFactorAt1450