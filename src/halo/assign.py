import numpy as np
from astropy.table import Table
from scipy.spatial import cKDTree

def compute_virial_radius(mass_hmsun):
    """
    Compute the virial radius in kpc for a halo of mass M (in M_sun).
    Uses the standard scaling: R_vir ≈ 206 * (M / 1e12)^{1/3} kpc
    """
    return 206 * (np.array(mass_hmsun) / 1e12) ** (1/3)

def assign_bh_to_fof(bh_table: Table, fof_table: Table, max_distance_kpc=300):
    """
    Assign each BH to the nearest FOF group and return distances.
    Returns a dictionary of BH ID -> (assigned FOF Group ID, distance in kpc, distance in virial radii).
    """

    bh_pos = np.vstack([bh_table['PositionX'], bh_table['PositionY'], bh_table['PositionZ']]).T
    fof_pos = np.vstack([fof_table['COMPosX'], fof_table['COMPosY'], fof_table['COMPosZ']]).T
    fof_ids = np.array(fof_table['GroupID'])
    fof_mass = np.array(fof_table['M_FOF'])

    # Compute virial radii for all FOF groups
    virial_radii = compute_virial_radius(fof_mass)
    
    tree = cKDTree(fof_pos)

    # Batch query: get nearest FOF group for all BHs at once
    dists, idxs = tree.query(bh_pos, k=1)

    assignments = {}
    bh_ids = np.array(bh_table['ID'])
    # Assign each BH to its nearest FOF group and store both distances
    for i, bh_id in enumerate(bh_ids):
        fof_idx = idxs[i]
        distance_kpc = dists[i]
        distance_virial = distance_kpc / virial_radii[fof_idx]
        assignments[bh_id] = (fof_ids[fof_idx], distance_kpc, distance_virial)

    return assignments
