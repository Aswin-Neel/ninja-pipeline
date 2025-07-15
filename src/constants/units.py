import numpy as np
from astropy.constants import c, G

MSUN = 1.989e33
YR_S = 3.154e7
C_CGS = c.cgs.value
G_CGS = G.cgs.value

ETA = 0.1
LOG_C2 = 2 * np.log10(C_CGS)
LOG_ETA = np.log10(ETA)
LUMINOSITY_THRESHOLD = 1e40
LOG_LUMINOSITY_THRESHOLD = np.log10(LUMINOSITY_THRESHOLD)
