from nbodykit.lab import *
from nbodykit.cosmology import WMAP9, LinearPower
import os
import numpy
from numpy.testing import assert_allclose

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

def compute_power(output, Nmesh=128):
    """Compute the compensated power spectrum from a catalogue."""
    catdm = BigFileCatalog(output, dataset='1/', header='Header')
    catgas = BigFileCatalog(output, dataset='0/', header='Header')
    meshdm = catdm.to_mesh(Nmesh=Nmesh, window='cic', compensated=True, interlaced=True)
    meshgas = catgas.to_mesh(Nmesh=Nmesh, window='cic', compensated=True, interlaced=True)
    pksimgas = FFTPower(catgas, mode='1d', Nmesh=128)
    pksimdm = FFTPower(catdm, mode='1d', Nmesh=128)
    z = 1. / catdm.attrs['Time'] - 1
    return pksimdm.power, pksimgas.power, z

def test_power(output, ref, ref_z, final=8):
    """Check the initial power against linear theory and a linearly grown IC power"""
    print('testing', output)
    pksim, pkgas, z = compute_power(output)

    #Check types have the same power
    fig = Figure(figsize=(5, 5), dpi=200)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    pklin = LinearPower(WMAP9, redshift=z)

    D = WMAP9.scale_independent_growth_factor(z) / WMAP9.scale_independent_growth_factor(ref_z)

    ax.plot(pksim['k'][1:], pksim['power'][1:].real / pklin(pksim['k'][1:]), label=output + " DM")
    ax.plot(pkgas['k'][1:], pkgas['power'][1:].real / pklin(pkgas['k'][1:]), ls="-.", label=output + " gas")
    ax.plot(ref['k'][1:], ref['power'][1:].real * D ** 2 / pklin(ref['k'][1:]), ls="--", label='ref, linearly grown')
    ax.set_xscale('log')
    ax.set_ylim(0.8, 1.2)
    ax.legend()
    fig.savefig(os.path.basename(output) + '.png')

    # asserting the linear growth is 1% accurate
    assert_allclose(pksim['power'][2:final],
                    ref['power'][2:final] * D ** 2,
                        rtol=0.012, atol=0.0)
    # Assert we agree with linear theory
    assert_allclose(pksim['power'][2:final],
                pklin(ref['k'])[2:final],
                rtol=0.025*D**2, atol=0.0)
    #Assert gas and DM are similar
    assert_allclose(pkgas['power'][1:final], pksim['power'][1:final], rtol=0.01*D**2, atol=0.)

# asserting the initial power spectrum is 1% accurate
print("testing IC power")
ref, refgas, ref_z = compute_power('output/IC')
pklin = LinearPower(WMAP9, redshift=ref_z)
#This checks that the power spectrum loading and rescaling code is working.
genpk = numpy.loadtxt("output/inputspec_IC.txt")
assert_allclose(pklin(genpk[:,0]), genpk[:,1], rtol=5e-4, atol=0.0)
#This checks that the output is working
test_power('output/IC', ref, ref_z)
test_power('output/PART_000', ref, ref_z)
test_power('output/PART_001', ref, ref_z)
test_power('output/PART_002', ref, ref_z)
