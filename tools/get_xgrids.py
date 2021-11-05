"""
	James' hacked up grid maker
	created from parts of the reionisation model

    Example usage: python get_xgrids.py  output/PART_005 grids/example_grid_z10 --pos=0/Position --weight=0/Mass

    For a 1Mpc resolution grid of gas particle overdensity at the sixth snapshot and save it to grids/example_grid_z10
"""

import argparse
import bigfile
from pmesh.pm import ParticleMesh, RealField, ComplexField
import logging
from mpi4py import MPI
import numpy
logger = logging
logging.basicConfig(level=logging.INFO)

def main(pfile,output,pos,weight,dataset,resolution=1.0,normtype='mean',chunksize=1024*1024*32):
    """ Takes a MP-Gadget output file (bigfile, formatted /PART_XYZ/ptype/prop/...) and builds grids of the desired property 

        Positional Arguments:
            pfile: path to particle file
            output: path to output bigfile
            pos: name of position dataset (usually 'X/Position')
            weight: name of weighting dataset

        Keyword Arguments:
            resolution: desired output resolution (Mpc h-1)
            normtype: normalization, one of:
                'mean' : produces overdensity grid (value / mean + 1)
                'volume' : regular volume weighted density grid (value Mpc-3)
                'particle' : per particle average, approximate mass weighting
                'mass' : WIP mass weighting
    """
    comm = MPI.COMM_WORLD

    ff = bigfile.BigFileMPI(comm, pfile)
    BoxSize = ff['Header'].attrs['BoxSize'] / 1000
    Redshift = 1/ff['Header'].attrs['Time'] - 1
	
    if normtype not in ['volume','mean','particle']:
        logger.info("normtype %s not supported",normtype)
        return

    Nmesh = int(BoxSize / resolution)
    # round it to 8.
    Nmesh -= Nmesh % 8

    if comm.rank == 0:
        logger.info("source = %s", pfile)
        logger.info("output = %s", output)
        logger.info("BoxSize = %g", BoxSize)
        logger.info("Redshift = %g", Redshift)
        logger.info("Nmesh = %g", Nmesh)
        logger.info("normtype = %s", normtype)

    pm = ParticleMesh([Nmesh, Nmesh, Nmesh], BoxSize, comm=comm)

    real = RealField(pm)
    real[...] = 0

    #particle field
    if normtype in ['mass','particle']:
    	norm = RealField(pm)
    	norm[...] = 0

    with ff[pos] as ds, ff[weight] as dx:
        logger.info(ds.size)
        for i in range(0, ds.size, chunksize):
            sl = slice(i, i + chunksize)
            posx = ds[sl] / 1000. #convert to Mpc
            nh = dx[sl]
            layout = pm.decompose(posx)
            lpos = layout.exchange(posx)
            real.paint(lpos, mass=nh, hold=True)
            if normtype == 'mass':
                 #TODO: finish mass weighting, same as particle for now which may cause issues for
                 norm.paint(lpos, mass=1.0, hold=True)
            elif normtype == 'particle':
                 norm.paint(lpos, mass=1.0, hold=True)

    #normalization
    if normtype == 'mean':
        normd = real.cmean()
    elif normtype == 'volume':
        normd = Boxsize/Nmesh/Nmesh/Nmesh
    elif normtype in ['particle','mass']:
        normd = norm[...]

    real[...] /= normd

    if comm.rank == 0:
        logger.info("mean %s per cell = %s", weight, real.cmean())

    buffer = numpy.empty(real.size, real.dtype)
    real.sort(out=buffer)
    if comm.rank == 0:
        logger.info("sorted for output")

    with bigfile.BigFileMPI(comm, output, create=True) as ff:
        with ff.create_from_array(dataset, buffer) as bb:
            bb.attrs['BoxSize'] = BoxSize
            bb.attrs['Redshift'] = Redshift
            bb.attrs['Nmesh'] = Nmesh

    if comm.rank == 0:
        logger.info("done. written at %s", output)

def run_multiple(datadir, datasets, normtypes, redshifts, outdir, resolution=1.,chunksize=1024*1024*32):
    """ Runs 'main' at multiple redshifts and datasets to produce all desired grids

        Positional Arguments:
            datadir: path to directory with all the PART files and snapshot list
            datasets: list of all desired datasets e.g: '4/Mass' for stellar mass
            normtypes: list of desired normalisations for each dataset (see 'main' docstring)
            redshifts: list of desired redshifts
            outdir: directory to save all the grids

        Keyword Arguments:
            resolution: desired output resolution (Mpc h-1)
            chunksize: number of particles to read at a time, helps memory for large files
    """
    #set up the list of snapshots, and redshifts
    snapshot_list = np.loadtxt(f'{ns.bigfile}/Snapshots.txt', dtype=float, ndmin=2)
    time_list = snapshot_list[:,1]
    redshift_list = 1/time_list - 1

    for z in redshifts:
        idx = np.argmin(np.fabs(redshift_list - z))
        snap = snapshot_list[idx,0]
        partfile = f'{datadir}/PART_{snap:03d}'
        for n,d in zip(normtypes,datasets):
            data = f'{partfile}/{d}'
            ptype = d.split('/')[1]
            pos = f'{ptype}/Position'

            dname = d.replace('/','-')
            zname = f'{z:.2f}'.replace('.','-')
            outname = f'{outdir}/{dname}_grid_z{zname}'

            main(partfile,outname,pos,d,resolution,n,chunksize)

if __name__ == '__main__':
    ap = argparse.ArgumentParser("get_xgrids.py")
    ap.add_argument("pfile", help='Particle data')
    ap.add_argument("output", help='Name of bigfile to store the mesh')
    ap.add_argument("--pos", help='Name of the position dataset in the bigfile')
    ap.add_argument("--weight", help='Name of the weighting dataset in the bigfile')
    ap.add_argument("--dataset", default='grid', help='name of the dataset to write to')
    ap.add_argument("--resolution", type=float, default=1.0, help='resolution in Mpc/h')
    ap.add_argument("--chunksize", type=int, default=1024*1024*32, help='number of particle to read at once')
    ns = ap.parse_args()
    main(ns.pfile,ns.output,ns.pos,ns.weight,ns.dataset,ns.resolution,ns.chunksize)
