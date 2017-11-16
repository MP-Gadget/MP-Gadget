from nbodykit.lab import *
from nbodykit.cosmology import WMAP9, LinearPower
import os
import numpy
from numpy.testing import assert_allclose

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

def compute_power(output, Nmesh=128):
    """Compute the compensated power spectrum from a catalogue."""
    catcdm = BigFileCatalog(output, dataset='1/', header='Header')

    try:
        catb = MultipleSpeciesCatalog(["gas", "star"], 
                BigFileCatalog(output, dataset='0/', header='Header'),
                BigFileCatalog(output, dataset='4/', header='Header'))
    except:
        catb = BigFileCatalog(output, dataset='0/', header='Header')

    meshcdm = catcdm.to_mesh(Nmesh=Nmesh, window='cic', compensated=True, interlaced=True)
    meshb = catb.to_mesh(Nmesh=Nmesh, window='cic', compensated=True, interlaced=True)
    pkb = FFTPower(catb, mode='1d', Nmesh=128)
    pkcdm = FFTPower(catcdm, mode='1d', Nmesh=128)
    z = 1. / catcdm.attrs['Time'] - 1
    return pkcdm.power, pkb.power, z

def test_power(output, refcdm, refb, ref_z, final=8):
    """Check the initial power against linear theory and a linearly grown IC power"""
    print('testing', output)
    pkcdm, pkb, z = compute_power(output)

    #Check types have the same power
    fig = Figure(figsize=(5, 5), dpi=200)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(121)

    pklin = LinearPower(WMAP9, redshift=z)

    D = WMAP9.scale_independent_growth_factor(z) / WMAP9.scale_independent_growth_factor(ref_z)

    ax.plot(pkcdm['k'][1:], pkcdm['power'][1:].real / pklin(pkcdm['k'][1:]), label=output + " DM")
    ax.plot(refcdm['k'][1:], refcdm['power'][1:].real * D ** 2 / pklin(refcdm['k'][1:]), ls="--", label='refcdm, linearly grown')

    ax.set_xscale('log')
    ax.set_ylim(0.8, 1.2)
    ax.legend()

    ax = fig.add_subplot(122)
    ax.plot(pkb['k'][1:], pkb['power'][1:].real / pklin(pkb['k'][1:]), ls="-.", label=output + " gas")
    ax.plot(refb['k'][1:], refb['power'][1:].real * D ** 2 / pklin(refb['k'][1:]), ls="--", label='refcdm gas, linearly grown')

    ax.set_xscale('log')
    ax.set_ylim(0.8, 1.2)
    ax.legend()
    fig.savefig(os.path.basename(output) + '.png')

    # asserting the linear growth is 1% accurate
    assert_allclose(pkcdm['power'][2:final],
                    refcdm['power'][2:final] * D ** 2,
                        rtol=0.012, atol=0.0)
    # Assert we agree with linear theory
    assert_allclose(pkcdm['power'][2:final],
                pklin(refcdm['k'])[2:final],
                rtol=0.025*D**2, atol=0.0)

    # asserting the linear growth is 1% accurate
    assert_allclose(pkb['power'][2:final],
                    refb['power'][2:final] * D ** 2,
                        rtol=0.012, atol=0.0)
    # Assert we agree with linear theory
    assert_allclose(pkb['power'][2:final],
                pklin(refb['k'])[2:final],
                rtol=0.025*D**2, atol=0.0)

# asserting the initial power spectrum is 1% accurate
print("testing IC power")
refcdm, refb, ref_z = compute_power('output/IC')
pklin = LinearPower(WMAP9, redshift=ref_z)
#This checks that the power spectrum loading and rescaling code is working.
genpk = numpy.loadtxt("output/inputspec_IC.txt")
assert_allclose(pklin(genpk[:,0]), genpk[:,1], rtol=5e-4, atol=0.0)
#This checks that the output is working
test_power('output/IC', refcdm, refb, ref_z)
test_power('output/PART_000', refcdm, refb, ref_z)
test_power('output/PART_001', refcdm, refb, ref_z)
test_power('output/PART_002', refcdm, refb, ref_z)
