import numpy as np
from astropy.table import Table

def compute_virial_radius(mass_hmsun):
    """
    Compute the virial radius in Kpc for a halo of mass M (in h^-1 M_sun).
    Uses the approximation: R_vir ≈ (GM / 100 H^2)^{1/3}
    """
    G = 4.302e-6  # kpc * (km/s)^2 / M_sun
    H0 = 70       # km/s/Mpc
    h = 0.7
    H = H0 * h    # Simplified approximation
    M = mass_hmsun / h
    R = (G * M / (100 * (H**2)))**(1/3)  # in kpc
    return R

def assign_bh_to_fof(bh_table: Table, fof_table: Table, max_distance_kpc=300):
    """
    Assign each BH to the nearest FOF group within the virial radius.
    Returns a dictionary of BH ID -> assigned FOF Group ID (or None).
    """
    bh_pos = np.vstack([bh_table['PositionX'], bh_table['PositionY'], bh_table['PositionZ']]).T
    fof_pos = np.vstack([fof_table['COMPosX'], fof_table['COMPosY'], fof_table['COMPosZ']]).T
    fof_ids = fof_table['GroupID']
    fof_mass = fof_table['M_FOF']

    virial_radii = compute_virial_radius(fof_mass)  # in Kpc

    assignments = {}
    for i, bh in enumerate(bh_table):
        bh_id = bh['ID']
        bh_xyz = bh_pos[i]

        dists = np.linalg.norm(fof_pos - bh_xyz, axis=1)  # Euclidean
        within_radius = dists < virial_radii

        if np.any(within_radius):
            closest_idx = np.argmin(dists[within_radius])
            valid_ids = fof_ids[within_radius]
            assignments[bh_id] = valid_ids[closest_idx]
        else:
            assignments[bh_id] = None

    return assignments
