"""This module compares two snapshots to ensure they are the same.
The main purpose is for development, to make sure that optimizations do not change
the code output."""

import bigfile
import numpy as np

def compare_fields(newsnap, oldsnap, ptype=1, field="Position"):
    """Compare two fields in a snapshot (by default DM positions):
       newsnap and oldsnap are compared.
       the 'field' array is compared for particle type 'ptype'.
       Returns the absolute value of the differences."""
    pp_old = bigfile.BigFile(oldsnap)
    box = pp_old["Header"].attrs["BoxSize"]
    otime = pp_old["Header"].attrs["Time"]
    pp_new = bigfile.BigFile(newsnap)
    ntime = pp_new["Header"].attrs["Time"]
    nbox = pp_new["Header"].attrs["BoxSize"]
    assert np.abs(otime - ntime) < 1e-8
    assert np.abs(box - nbox) < 1e-8
    sptype = str(ptype)
    id_new = pp_new[sptype+"/ID"][:]
    id_old = pp_old[sptype+"/ID"][:]
    pos_new = pp_new[sptype+"/"+field][:]
    pos_old = pp_old[sptype+"/"+field][:]
    p_sort_new = pos_new[np.argsort(id_new)]
    p_sort_old = pos_old[np.argsort(id_old)]
    diff = p_sort_new - p_sort_old
    #Positions wrap, so the differences
    if field == "Position":
        ii = np.where(diff > box/2)
        diff[ii] = diff[ii] - box
        ii = np.where(diff < -box/2)
        diff[ii] = diff[ii] + box
    return np.abs(diff)
