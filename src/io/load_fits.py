import os
import glob
import re
from astropy.io import fits
from astropy.table import Table

def load_fits_table(filepath):
    with fits.open(filepath) as hdul:
        return Table(hdul[1].data)

def get_redshift_from_filename(filepath):
    base = os.path.basename(filepath)
    match = re.search(r'_z_([0-9.]+)\.fits$', base)
    if match:
        return float(match.group(1))
    return None

def get_snapshot_number(filename):
    match = re.search(r'_in_snapshot_(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None
