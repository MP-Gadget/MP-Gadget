from nbodykit.lab import *
from nbodykit.cosmology import WMAP9, LinearPower
import os
import math
import numpy
import scipy.interpolate
from numpy.testing import assert_allclose

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def compute_power(output):

    cat = MultipleSpeciesCatalog(['gas', 'dm'],
            BigFileCatalog(output, dataset='0/', header='Header'),
            BigFileCatalog(output, dataset='1/', header='Header'))

    cat.attrs['BoxSize'] = cat.attrs['dm.BoxSize'] * [1, 1, 1]

    pksim = FFTPower(cat.to_mesh(interlaced=True, window='tsc', compensated=True, Nmesh=128), mode='1d')
#    pksimgas = FFTPower(cat['gas'], mode='1d', Nmesh=128)
#    pksimdm = FFTPower(cat['dm'], mode='1d', Nmesh=128)
    z = 1. / cat.attrs['dm.Time'] - 1
    return pksim, z

ref, ref_z = compute_power('output/IC')

def test_power(output, ref, ref_z):
    print('testing', output)
    pksim, z = compute_power(output)

    fig = Figure(figsize=(5, 5), dpi=200)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    pklin = LinearPower(WMAP9, redshift=z)

    D = WMAP9.scale_independent_growth_factor(z) / WMAP9.scale_independent_growth_factor(ref_z)

    ax.plot(pksim.power['k'], pksim.power['power'] / pklin(pksim.power['k']), label=output)
    ax.plot(ref.power['k'], ref.power['power'] * D ** 2 / pklin(ref.power['k']), label='ref, linearly grown')
    ax.set_xscale('log')
    ax.set_ylim(0, 2)
    ax.legend()
    fig.savefig(os.path.basename(output) + '.png')

    # asserting the linear growth is 1% accurate
    assert_allclose(abs(pksim.power['power'])[2:8],
                    abs(ref.power['power'])[2:8] * D ** 2,
                        rtol=0.01, atol=0.0)

def invwindow(kk):
    """Sunc function, inverse window of the CIC."""
    #2 pi nmesh / box
    grid = 2* math.pi * 128 / 512.
    return 1./numpy.sinc(kk/grid)**6

# asserting the initial power spectrum is 1% accurate
print("testing IC power")
pklin = LinearPower(WMAP9, redshift=ref_z)
#This checks that the power spectrum loading and rescaling code is working.
genpk = numpy.loadtxt("output/inputspec_IC.txt")
assert_allclose(pklin(genpk[:,0]), genpk[:,1], rtol=5e-4, atol=0.0)
#This checks that the output is working
print(ref_z, ref.power['power'][2:8])
print(pklin(ref.power['k'][2:8]))
#Deconvolve by the CIC window. This is not exact.
assert_allclose(abs(ref.power['power'])[2:8]*invwindow(ref.power['k'][2:8]),
                abs(pklin(ref.power['k']))[2:8],
                rtol=0.025, atol=0.0)

test_power('output/IC', ref, ref_z)

test_power('output/PART_000', ref, ref_z)
test_power('output/PART_001', ref, ref_z)
test_power('output/PART_002', ref, ref_z)
