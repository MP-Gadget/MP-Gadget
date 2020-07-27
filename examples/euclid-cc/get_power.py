"""Generate a power spectrum from the snapshot"""
from nbodykit.lab import BigFileCatalog,FFTPower
import numpy as np

def compute_power(output, Nmesh=4096):
    """Compute the compensated power spectrum from a catalogue."""
    catcdm = BigFileCatalog(output, dataset='1/', header='Header')
    assert 1 - catcdm.attrs['Time'] < 0.01
    catcdm.to_mesh(Nmesh=Nmesh, resampler='cic', compensated=True, interlaced=True)
    pkcdm = FFTPower(catcdm, mode='1d', Nmesh=Nmesh)
    return pkcdm.power

pkk = compute_power("output/PART_007")
np.savetxt(pkk['k'], pkk['power'].real)
