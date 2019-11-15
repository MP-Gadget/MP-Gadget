"""
    Produce a patchy reionization modulation field.
    Different parts of the universe reionizes at different times.
    Denser region is ionized earlier because there is more source.
    Thus there is a correlation between density field and the reionization
    time (time the UV background penetrates the IGM).

    This script implements a correlation model in Battaglia et al. 2013
    http://adsabs.harvard.edu/abs/2013ApJ...776...81B

    The model is used in BlueTides Simulation.

    This version of code takes the non-linear particle output of FastPM.

    It is supposed to write a UV Modulation in a format known by MP-Gadget.


    Authors:

        Yu Feng <rainwoodman@gmail.com>

"""

import argparse
import bigfile
from pmesh.pm import ParticleMesh, RealField, ComplexField
import logging
from mpi4py import MPI
import numpy

from fastpm.core import Solver
from nbodykit.cosmology import Cosmology, LinearPower
from astropy.cosmology import FlatLambdaCDM
from astropy import units as U, constants as C

ap = argparse.ArgumentParser("preion-make-zreion.py")
ap.add_argument("output", help='name of bigfile to store the mesh')
ap.add_argument("--dataset", default='Zreion_table', help='name of the dataset that stores the reionization redshift')
ap.add_argument("--resolution", type=float, default=1.0, help='resolution in Mpc/h')
ap.add_argument("--filtersize", type=float, default=1.0, help='resolution in Mpc/h')
ap.add_argument("--chunksize", type=int, default=1024*1024*16, help='number of particle to read at once')
ap.add_argument("--boxsize",type=float,default=400.,help='box size in Mpc/h')
ap.add_argument("--redshift",type=float,default=8.,help='median redshift of reionisation')
logger = logging
logging.basicConfig(level=logging.INFO)

def tophat(R, k):
    rk = R * k
    mask = rk == 0
    rk[mask] = 1
    ans = 3.0/(rk*rk*rk)*(numpy.sin(rk)-(rk)*numpy.cos(rk))
    ans[mask] = 1
    return ans

def Bk(k):
    # patchy reionization model
    # FIXME: need a citation, any meta parameters?

    b0 = 1.0 / 1.686;
    k0 = 0.185;
    #k0 = 0.000185
    al = 0.564;
    ans =  b0/pow(1 + (k/k0),al);
    return ans;

def get_lpt(pm,z):
    a = 1/(1+z)
    Planck18 = FlatLambdaCDM(H0=67.36,Om0=0.3153,Tcmb0=2.7255*U.Unit('K'),Neff=3.046
                ,m_nu=numpy.array([0,0,0.06])*U.Unit('eV'),Ob0=0.02237/0.6736/0.6736)
    Planck18 = Cosmology.from_astropy(Planck18)
    Plin = LinearPower(Planck18, redshift=0, transfer='EisensteinHu')
    solver = Solver(pm, Planck18, B=1)
    Q = pm.generate_uniform_particle_grid()

    wn = solver.whitenoise(422317)
    dlin = solver.linear(wn, lambda k: Plin(k))

    state = solver.lpt(dlin, Q, a=a, order=2)

    return state

def main():
    ns = ap.parse_args()
    comm = MPI.COMM_WORLD

    BoxSize = ns.boxsize
    Redshift = ns.redshift

    Nmesh = int(BoxSize / ns.resolution)
    # round it to 8.
    Nmesh -= Nmesh % 8

    if comm.rank == 0:
        logger.info("output = %s", ns.output)
        logger.info("BoxSize = %g", BoxSize)
        logger.info("Redshift = %g", Redshift)
        logger.info("Nmesh = %g", Nmesh)

    pm = ParticleMesh([Nmesh, Nmesh, Nmesh], BoxSize, comm=comm)

    state = get_lpt(pm,Redshift)
    real = state.to_mesh()

    logger.info("field painted")

    mean = real.cmean()

    if comm.rank == 0:
        logger.info("mean 2lpt density = %s", mean)

    real[...] /= mean
    real[...] -= 1

    complex = real.r2c()
    logger.info("field transformed")

    for k, i, slab in zip(complex.slabs.x, complex.slabs.i, complex.slabs):
        k2 = sum(kd ** 2 for kd in k)
        # tophat
        f = tophat(ns.filtersize, k2 ** 0.5)
        slab[...] *= f
        # zreion
        slab[...] *= Bk(k2 ** 0.5)
        slab[...] *= (1 + Redshift)

    real = complex.c2r()
    real[...] += Redshift
    logger.info("filters applied %d",real.size)

    mean = real.cmean()
    if comm.rank == 0:
        logger.info("zreion.mean = %s", mean)

    buffer = numpy.empty(real.size, real.dtype)
    real.sort(out=buffer)
    if comm.rank == 0:
        logger.info("sorted for output")

    with bigfile.BigFileMPI(comm, ns.output, create=True) as ff:
        with ff.create_from_array(ns.dataset, buffer) as bb:
            bb.attrs['BoxSize'] = BoxSize
            bb.attrs['Redshift'] = Redshift
            bb.attrs['TopHatFilterSize'] = ns.filtersize
            bb.attrs['Nmesh'] = Nmesh
        #
        # hack: compatible with current MPGadget. This is not really needed
        # we'll remove the bins later, since BoxSize and Nmesh are known.
        with ff.create("XYZ_bins", dtype='f8', size=Nmesh) as bb:
            if comm.rank == 0:
                bins = numpy.linspace(0, BoxSize, Nmesh, dtype='f8')
                bb.write(0, bins)

    if comm.rank == 0:
        logger.info("done. written at %s", ns.output)

if __name__ == '__main__':
    main()
