from astropy.table import Table
import numpy as np

def get_bh_by_id(table: Table, bh_id: int):
    """Retrieve a single black hole entry by its ID."""
    mask = table['ID'] == bh_id
    return table[mask]

def match_bh_across_snapshots(prev_table: Table, curr_table: Table, bh_id: int):
    """
    Given a BH ID, return its entries from previous and current snapshots if it exists in both.
    """
    prev_entry = get_bh_by_id(prev_table, bh_id)
    curr_entry = get_bh_by_id(curr_table, bh_id)
    return prev_entry, curr_entry

def compute_mass_ratio(prev_table: Table, curr_table: Table, bh_id: int):
    """
    Compute the mass ratio between a BH's current and previous snapshot mass.
    """
    prev, curr = match_bh_across_snapshots(prev_table, curr_table, bh_id)
    if len(prev) == 0 or len(curr) == 0:
        return None
    return curr['Mass'][0] / prev['Mass'][0]

def construct_merger_tree(bh_snapshots: dict, bh_id: int):
    """
    Traverse backward in time to construct the merger history of a given BH ID
    across snapshots using SwallowID links.
    """
    tree = []
    current_id = bh_id
    sorted_snaps = sorted(bh_snapshots.keys(), reverse=True)

    for snap in sorted_snaps:
        table = bh_snapshots[snap]
        entry = get_bh_by_id(table, current_id)
        if len(entry) == 0:
            continue
        tree.append((snap, entry))
        if entry['Progenitors'][0] == 0:
            break
        current_id = entry['SwallowID'][0]

    return tree[::-1]  # Return in increasing snapshot order
