import numpy as np
import astropy.units as u

from .utils import LogLogInterpolator


class OpacityData:
    """Wrapper for opacity data from CLOUDY output file.
    
    API:
    `nu, energy, wavelength` : the x-axis of the opacity data (with units)
    `nu_cgs` : nu in Hz, for convenience
    `sigma_H_ext, sigma_H_abs` : the opacity data
    `interp_ext, interp_abs` : interpolators for the opacity data
    
    Usage:
    ```python
    Orion_opacity = OpacityData('Orion.opc')
    ```
    """
    def __init__(self, filename: str):
        self.filename = filename
        self.data = np.loadtxt(filename, comments='#', delimiter='\t', usecols=(0, 1, 2))  # read from file
        self.data = np.unique(self.data, axis=0)  # remove duplicate values, and sort by the first column
        
        self.sigma_H_ext = self.data[:, 1] * u.cm**2
        self.sigma_H_abs = self.data[:, 2] * u.cm**2
        #* The x-axis: energy / nu / wavelength, have units
        self.energy = self.data[:, 0] * u.Ry  # in [Rydberg]
        self.nu = self.energy.to(u.Hz, equivalencies=u.spectral())
        self.wavelength = self.nu.to(u.um, equivalencies=u.spectral())
        
        self.interp_ext = LogLogInterpolator(self.nu, self.sigma_H_ext)
        self.interp_abs = LogLogInterpolator(self.nu, self.sigma_H_abs)
    
    @property
    def nu_cgs(self):
        return self.nu.cgs.value
    
