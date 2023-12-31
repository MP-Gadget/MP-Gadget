"""Checks output of DM simulation with halo catalogue and power"""
import os
import os.path
import numpy as np
from numpy.testing import assert_allclose
import scipy.interpolate
import bigfile

def check_hmf(pig):
    """Check we have the mass functions."""
    bff = bigfile.BigFile(pig)
    lbox = float(bff["Header"].attrs["BoxSize"])
    hh = bff["Header"].attrs["HubbleParam"]
    fofmasses = bff['FOFGroups/Mass'][:]*10**10/hh
    assert np.size(fofmasses > 0)

# Mass functions
check_hmf('output/PIG_002')
