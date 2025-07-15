import numpy as np
from numba import njit, prange
from src.constants.units import MSUN, YR_S, LOG_ETA, LOG_C2

@njit(parallel=True)
def compute_log_luminosity(mdot):
    n = len(mdot)
    log_lum = np.full(n, -np.inf, dtype=np.float64)
    for i in prange(n):
        if mdot[i] > 0:
            mdot_cgs = mdot[i] * MSUN / YR_S
            log_lum[i] = LOG_ETA + np.log10(mdot_cgs) + LOG_C2
    return log_lum
