import os
import numpy as np
from astropy.table import Table
from src.io.load_fits import get_redshift_from_filename, get_snapshot_number, load_fits_table
from src.process.accretion import compute_log_luminosity
from src.constants.units import LOG_LUMINOSITY_THRESHOLD

# Sentinel value indicating no merger
SWALLOW_SENTINEL = 18446744073709551615

def find_matching_file(directory, z):
    for f in os.listdir(directory):
        if f.endswith(".fits") and f"_z_{z}" in f and "BH_in_snapshot" in f:
            return os.path.join(directory, f)
    return None

def get_merger_flags(prev_table):
    ids = np.array(prev_table['ID'])
    swallows = np.array(prev_table['SwallowID'])
    return {id_: swallows[i] != SWALLOW_SENTINEL for i, id_ in enumerate(ids)}

def compute_qso_merger_fraction(bh_dir, z, log_lcut=LOG_LUMINOSITY_THRESHOLD):
    z = float(z)
    zs = sorted([
        get_redshift_from_filename(f) for f in os.listdir(bh_dir)
        if f.endswith('.fits') and 'BH_in_snapshot' in f
    ])
    if z not in zs:
        raise ValueError(f"No BH file found for z = {z}")
    
    i = zs.index(z)
    if i == 0:
        raise ValueError("No previous redshift file available.")

    z_prev = zs[i - 1]

    f_curr = find_matching_file(bh_dir, z)
    f_prev = find_matching_file(bh_dir, z_prev)
    
    curr_table = load_fits_table(f_curr)
    prev_table = load_fits_table(f_prev)
    
    bh_ids = np.array(curr_table['ID'])
    mdot = np.array(curr_table['BlackholeAccretionRate'])
    
    logL = compute_log_luminosity(mdot)
    is_qso = logL > log_lcut
    
    merger_flags = get_merger_flags(prev_table)

    merger_qsos = 0
    total_qsos = 0

    for i, isqso in enumerate(is_qso):
        if isqso:
            total_qsos += 1
            bh_id = bh_ids[i]
            if merger_flags.get(bh_id, False):
                merger_qsos += 1

    frac = (merger_qsos / total_qsos) if total_qsos > 0 else 0.0
    return total_qsos, merger_qsos, frac
