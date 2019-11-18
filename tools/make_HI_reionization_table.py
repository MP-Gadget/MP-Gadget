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

import os.path
import argparse
import logging
import bigfile
from mpi4py import MPI
import numpy

import configobj
import validate

from pmesh.pm import ParticleMesh
from fastpm.core import Solver
from nbodykit.cosmology import Cosmology, LinearPower

GenICconfigspec = """
FileWithInputSpectrum = string(default='')
FileWithTransferFunction = string(default='')
Ngrid = integer(min=0)
BoxSize = float(min=0)
Omega0 = float(0,1)
OmegaLambda = float(0,1)
OmegaBaryon = float(0,1,default=0.0486)
HubbleParam = float(0,2)
Redshift = float(0,1100)
Sigma8 = float(default=-1)
Seed = integer(min=0)
InputPowerRedshift = float(default=-1)
DifferentTransferFunctions = integer(0,1, default=1)
UnitLength_in_cm  = float(default=3.085678e21)
Omega_fld = float(0,1,default=0)
w0_fld = float(default=-1)
wa_fld = float(default=0)
MNue = float(min=0, default=0)
MNum = float(min=0, default=0)
MNut = float(min=0, default=0)
MWDM_Therm = float(min=0, default=0)
PrimordialIndex = float(default=0.971)
PrimordialAmp = float(default=2.215e-9)
PrimordialRunning = float(default=0)
CMBTemperature = float(default=2.7255)""".split('\n')

def tophat(R, k):
    """Top hat filter for the reionization patches"""
    rk = R * k
    mask = rk == 0
    rk[mask] = 1
    ans = 3.0/(rk*rk*rk)*(numpy.sin(rk)-(rk)*numpy.cos(rk))
    ans[mask] = 1
    return ans

def Bofk(k):
    """Patchy reionization model. This associates the overdensity
    with a reionization probability, from Battaglia 2013."""

    #Critical density.
    b0 = 1.0 / 1.686
    k0 = 0.185
    #k0 = 0.000185
    al = 0.564
    ans =  b0/pow(1 + (k/k0),al)
    return ans

def get_lpt(pm,z, cosmology, seed):
    """Evolve the linear power using a 2LPT solver,
       so we get a good model of the density structure at the reionization redshift."""
    a = 1/(1+z)
    Plin = LinearPower(cosmology, redshift=0, transfer='EisensteinHu')
    solver = Solver(pm, cosmology, B=1)
    Q = pm.generate_uniform_particle_grid()

    wn = solver.whitenoise(seed)
    dlin = solver.linear(wn, Plin)

    state = solver.lpt(dlin, Q, a=a, order=2)

    return state

def generate_zreion_file(paramfile, output, redshift, resolution):
    """Do the work and output the file.
    This reads parameters from the MP-GenIC paramfile, generates a table of patchy reionization redshifts, and saves it.
    This table can be read by MP-Gadget. The core of the method is a correlation between large-scale overdensity and
    redshift of reionization calibrated from a radiative transfer simulation. To realise this, the code needs to know the large
    scale overdensity, which it does using FastPM.

    Arguments:
    - paramfile: genic parameter file to read from
    - output: file to save the reionization table to
    - redshift: redshift for the midpoint of reionization
    - Nmesh: sie of the particle grid to use
    """
    config = configobj.ConfigObj(infile=paramfile, configspec=GenICconfigspec, file_error=True)
    #Input sanitisation
    vtor = validate.Validator()
    config.validate(vtor)
    comm = MPI.COMM_WORLD

    logger = logging
    cm_per_mpc = 3.085678e24
    BoxSize = config["BoxSize"] * config["UnitLength_in_cm"] / cm_per_mpc
    Redshift = redshift

    Nmesh = int(BoxSize / resolution)
    # round it to 8.
    Nmesh -= Nmesh % 8

    # Top-hat filter the density field on one grid scale
    filtersize = resolution


    if comm.rank == 0:
        logger.info("output = %s", output)
        logger.info("BoxSize = %g", BoxSize)
        logger.info("Redshift = %g", Redshift)
        logger.info("Nmesh = %g", Nmesh)

    pm = ParticleMesh([Nmesh, Nmesh, Nmesh], BoxSize, comm=comm)

    mnu = numpy.array([config["MNue"], config["MNum"], config["MNut"]])
    omegacdm = config["Omega0"] - config["OmegaBaryon"] - numpy.sum(mnu)/93.14/config["HubbleParam"]**2
    cosmo = Cosmology(h=config["HubbleParam"],Omega0_cdm=omegacdm,T0_cmb=config["CMBTemperature"])
    state = get_lpt(pm,Redshift, cosmo, config["Seed"])
    real = state.to_mesh()

    logger.info("field painted")

    mean = real.cmean()

    if comm.rank == 0:
        logger.info("mean 2lpt density = %s", mean)

    real[...] /= mean
    real[...] -= 1

    cmplx = real.r2c()
    logger.info("field transformed")

    for k, _, slab in zip(cmplx.slabs.x, cmplx.slabs.i, cmplx.slabs):
        k2 = sum(kd ** 2 for kd in k)
        # tophat
        f = tophat(filtersize, k2 ** 0.5)
        slab[...] *= f
        # zreion
        slab[...] *= Bofk(k2 ** 0.5)
        slab[...] *= (1 + Redshift)

    real = cmplx.c2r()
    real[...] += Redshift
    logger.info("filters applied %d",real.size)

    mean = real.cmean()
    if comm.rank == 0:
        logger.info("zreion.mean = %s", mean)

    buffer = numpy.empty(real.size, real.dtype)
    real.ravel(out=buffer)
    if comm.rank == 0:
        logger.info("sorted for output")
    if os.path.exists(output):
        raise IOError("Refusing to write to existing file: ",output)

    with bigfile.BigFileMPI(comm, output, create=True) as ff:
        with ff.create_from_array("Zreion_Table", buffer) as bb:
            bb.attrs['BoxSize'] = BoxSize
            bb.attrs['Redshift'] = Redshift
            bb.attrs['TopHatFilterSize'] = filtersize
            bb.attrs['Nmesh'] = Nmesh
        #
        # hack: compatible with current MPGadget. This is not really needed
        # we'll remove the bins later, since BoxSize and Nmesh are known.
        with ff.create("XYZ_bins", dtype='f8', size=Nmesh) as bb:
            if comm.rank == 0:
                bins = numpy.linspace(0, BoxSize, Nmesh, dtype='f8')
                bb.write(0, bins)

    if comm.rank == 0:
        logger.info("done. written at %s", output)

if __name__ == '__main__':
    ap = argparse.ArgumentParser("preion-make-zreion.py")
    ap.add_argument("--output", help='name of bigfile to store the mesh', required=True)
    ap.add_argument("--genic", help="Name of genic parameter file to read for cosmology and box size", required=True)
    ap.add_argument("--resolution", type=float, default=1.0, help='Resolution of the reionization field in Mpc/h. 1 Mpc is the value from Battaglia 2013')
    ap.add_argument("--redshift",type=float,default=7.5,help='median redshift of reionisation')
    ns = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    generate_zreion_file(output=ns.output, paramfile = ns.genic, resolution=ns.resolution, redshift = ns.redshift)
