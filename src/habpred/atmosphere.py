"""Atmosphere model used to scale the parachute descent rate with altitude.

This is the simple piecewise model from NASA Glenn's "Earth Atmosphere Model"
page (troposphere / lower stratosphere / upper stratosphere), the same one
Tawhiri uses, so descent times match SondeHub's predictor.
"""

import math

from numba import njit

# Tawhiri scales the sea-level descent rate by 1.1045 / sqrt(rho). 1.1045**2
# is the sea-level density that convention implies; we keep it for parity.
RHO0_TAWHIRI = 1.1045**2


@njit(cache=True)
def temperature_pressure(alt: float) -> tuple[float, float]:
    """Return (temperature degC, pressure kPa) at geometric altitude ``alt`` m."""
    if alt > 25000.0:
        temp = -131.21 + 0.00299 * alt
        pressure = 2.488 * ((temp + 273.1) / 216.6) ** -11.388
    elif alt > 11000.0:
        temp = -56.46
        pressure = 22.65 * math.exp(1.73 - 0.000157 * alt)
    else:
        temp = 15.04 - 0.00649 * alt
        pressure = 101.29 * ((temp + 273.1) / 288.08) ** 5.256
    return temp, pressure


@njit(cache=True)
def density(alt: float) -> float:
    """Air density in kg/m^3 at altitude ``alt`` m."""
    temp, pressure = temperature_pressure(alt)
    return pressure / (0.2869 * (temp + 273.1))


@njit(cache=True)
def descent_rate(alt: float, sea_level_rate: float) -> float:
    """Parachute descent speed (positive, m/s) at ``alt`` for a given sea-level rate.

    Terminal velocity scales with 1/sqrt(rho) when drag area and mass are fixed.
    """
    return sea_level_rate * math.sqrt(RHO0_TAWHIRI / density(alt))
