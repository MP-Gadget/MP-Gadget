from nbodykit.lab import *
from nbodykit.cosmology import Planck15, LinearPower

from numpy.testing import assert_allclose

iccat = MultipleSpeciesCatalog(['gas', 'dm'],
        BigFileCatalog('output/IC', dataset='0/', header='Header'),
        BigFileCatalog('output/IC', dataset='1/', header='Header'))

iccat.attrs['BoxSize'] = iccat.attrs['dm.BoxSize'] * [1, 1, 1]

"""
p000 = MultipleSpeciesCatalog(['gas', 'dm'],
        BigFileCatalog('output/PART_000', '0/', header='Header'),
        BigFileCatalog('output/PART_000', '1/', header='Header'))

p001 = MultipleSpeciesCatalog(['gas', 'dm'],
        BigFileCatalog('output/PART_001', '0', header='Header'),
        BigFileCatalog('output/PART_001', '1', header='Header'))
"""

print(iccat.attrs)

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

fig = Figure(figsize=(5, 5), dpi=200)
canvas = FigureCanvasAgg(fig)
ax = fig.add_subplot(111)

pklin = LinearPower(Planck15, redshift=1 / iccat.attrs['dm.Time'] - 1)

icp = FFTPower(iccat['dm'], mode='1d', Nmesh=128)
ax.plot(icp.power['k'], icp.power['power'], label='data')
ax.plot(icp.power['k'], pklin(icp.power['k']), label='model')
ax.loglog()

fig.savefig('ic.png')
print(abs(icp.power['power'])[:10], pklin(icp.power['k'])[:10])
assert_allclose(abs(icp.power['power'])[:10], pklin(icp.power['k'])[:10], rtol=0.1, atol=0.1)
