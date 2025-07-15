import os
import glob
from astropy.io import fits
from astropy.table import Table

def load_fits_table(filepath):
    with fits.open(filepath) as hdul:
        return Table(hdul[1].data)

def get_redshift_from_filename(filepath):
    base = os.path.basename(filepath)
    snap_num = int(base.split('_')[-1].split('.')[0])
    return snap_num
