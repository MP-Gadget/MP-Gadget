from nbodykit.lab import *
from nbodykit.cosmology import Planck15, LinearPower
import numpy
pklin0 = LinearPower(Planck15, redshift=0.0)
pklin9 = LinearPower(Planck15, redshift=9.0)
k = numpy.logspace(-4, 1, 1000, endpoint=True)

numpy.savetxt('planck15-z0.txt', list(zip(k, pklin0(k))))
numpy.savetxt('planck15-z9.txt', list(zip(k, pklin9(k))))

