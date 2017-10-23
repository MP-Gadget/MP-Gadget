from nbodykit.lab import *
from nbodykit.cosmology import WMAP9, LinearPower
import numpy
pklin0 = LinearPower(WMAP9, redshift=0.0)
pklin9 = LinearPower(WMAP9, redshift=9.0)
k = numpy.logspace(-3, 2, 10000, endpoint=True)

numpy.savetxt('wmap9-z0.txt', list(zip(k, pklin0(k))))
numpy.savetxt('wmap9-z9.txt', list(zip(k, pklin9(k))))

