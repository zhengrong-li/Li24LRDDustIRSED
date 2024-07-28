import numpy as np
import astropy.units as u
from scipy import interpolate, integrate


try:
    import astropy.units as u
    INSTALLED_astropy = True
except ImportError:
    INSTALLED_astropy = False
try:
    import unyt
    INSTALLED_unyt = True
except ImportError:
    INSTALLED_unyt = False


class ScaledInterpolator:
    @staticmethod
    def _identity(x):
        return x
    
    @staticmethod
    def get_value_unit(x):
        if INSTALLED_astropy and isinstance(x, u.Quantity):  # if astropy is installed, then enable the Quantity support
            return x.value, x.unit
        elif INSTALLED_unyt and isinstance(x, unyt.unyt_array):  # if unyt is installed, then enable the unyt_array support
            return x.value, x.units
        return x, 1
    
    def __init__(self, x, y, scale_x=(_identity, _identity), scale_y=(_identity, _identity)):
        self.x = x
        self.y = y
        self.f_x, self.inv_f_x = scale_x
        self.f_y, self.inv_f_y = scale_y
        
        x_value, self.x_unit = self.get_value_unit(x)
        y_value, self.y_unit = self.get_value_unit(y)
        
        self._interp = interpolate.interp1d(self.f_x(x_value), self.f_y(y_value), kind='linear')
        
    def __call__(self, x):
        unit = self.get_value_unit(x)[1] 
        if self.x_unit != 1 and unit != 1:   # if input x has a unit
            x = x.to_value(self.x_unit)   # convert to the same internal unit. Note the .to_value() method is the same in astropy and unyt
        elif self.x_unit == 1 and unit == 1:
            pass
        else:  # if x is unitless, but the interpolator has a unit
            raise ValueError(f"The input x should have the same unit as {self.x_unit = } in the interpolator")
        
        return self.inv_f_y(self._interp(self.f_x(x))) * self.y_unit
    
    @property
    def inverse(self):
        """Return a new interpolator with the x and y swapped. The scale_x and scale_y are also swapped."""
        return __class__(self.y, self.x, scale_x=(self.f_y, self.inv_f_y), scale_y=(self.f_x, self.inv_f_x))


class LogLogInterpolator(ScaledInterpolator):
    def __init__(self, x, y):
        super().__init__(x, y, scale_x=(np.log, np.exp), scale_y=(np.log, np.exp))
        
class LogLinearInterpolator(ScaledInterpolator):
    def __init__(self, x, y):
        _identity = self._identity
        super().__init__(x, y, scale_x=(np.log, np.exp), scale_y=(_identity, _identity))
        
        
        
# integrate in log log space 

def trapz_log(y, x, *args, **kwargs):
    return integrate.trapezoid(y * x, np.log(x), *args, **kwargs)

def quad_vec_log(f, a, b, *args, **kwargs):
    def integrand(log_x):
        x = np.exp(log_x)
        return f(x) * x
    return integrate.quad_vec(integrand, np.log(a), np.log(b), *args, **kwargs)
        
        
        