import argparse
import bigfile
import logging
from mpi4py import MPI
import numpy as np
logger = logging
logging.basicConfig(level=logging.INFO)
from nbodykit.lab import BigFileCatalog
from os.path import exists

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

def main(pfile,output,outname,pos='Position',weight='Mass',dataset='0/',resolution=1.0,value='Value',norm='global'):
    """ Takes a MP-Gadget output file (bigfile, formatted /PART_XYZ/ptype/prop/...) and builds grids of the desired property 

        Positional Arguments:
            pfile: path to particle file
            output: path to output bigfile
            outname: dataset name to save

        Keyword Arguments:
            resolution: desired output resolution (Mpc h-1)
            pos: name of position column (usually 'Position')
            weight: name of weighting column (usually 'Mass')
            value: name of the vlaue column (desired property)
            dataset: name of particle dataset

            norm: 'global' => normalise the output grids by the global mean (weight*value / global mean)
                  'local' => divide by mass grid (average value in cell)
                  'none' => do not normalise (grid is weight*value in cell)

        Defaults produce a 1+delta field of gas
    """
    comm = MPI.COMM_WORLD
    
    cat = BigFileCatalog(pfile,dataset=dataset,header='Header')
    BoxSize = cat.attrs['BoxSize'] / 1000
    Redshift = 1/cat.attrs['Time'] - 1
	
    Nmesh = int(BoxSize / resolution)
    # round it to 8.
    Nmesh -= Nmesh % 8

    if comm.rank == 0:
        logger.info("source = %s", pfile)
        logger.info("output = %s", output)
        logger.info("Weight = %s",weight)
        logger.info("Value = %s",value)
        logger.info("BoxSize = %g", BoxSize)
        logger.info("Redshift = %g", Redshift)
        logger.info("Nmesh = %g", Nmesh)

    #particle field

    mesh = cat.to_mesh(Nmesh=Nmesh,weight=weight,value=value,position=pos,compensated=True)    
    #mesh = cat.to_mesh(Nmesh=Nmesh,value=value,position=pos,compensated=True)    

    #TODO: nbodykit has a save function which does almost everything below, but doesn't have the option for a non-normalised field
    # we sometimes need: mass-weighted & globally normalized (density)
    # mass-weighted & locally normalised (neutral fraction, local J21)
    # particle-weighted & not normalised (SFR)
    # mass-weighted & not normalised (Stellar Mass)
    #mesh.save(f'{outname}.bigfile')
    #'''

    field = mesh.to_real_field(normalize=(norm=='global'))

    if norm == 'local':
        mesh_mass = cat.to_mesh(Nmesh=Nmesh,weight=weight,position=pos,compensated=True)
        #mesh_mass = cat.to_mesh(Nmesh=Nmesh,weight=weight,position=pos)
        field_mass = mesh_mass.to_real_field(normalize=False)
        field[...] /= field_mass[...]
    
    meshmean = field.cmean()
    meshshape = field.cshape
    meshsize = field.csize
    if comm.rank == 0:
        logger.info("mean %s per cell = %s, grid size %s (%s)", value, meshmean,meshshape,meshsize)


    data = np.empty(shape=field.size, dtype=field.dtype)
    field.ravel(out=data)

    with bigfile.BigFileMPI(comm, output, create=True) as ff:
        with ff.create_from_array(outname, data) as bb:
            bb.attrs['BoxSize'] = BoxSize
            bb.attrs['Redshift'] = Redshift
            bb.attrs['Nmesh'] = Nmesh
    #'''
    if comm.rank == 0:
        logger.info("done. written at %s / %s", output, outname)

def run_multiple(datadir, datasets, values, weightings, normtypes, redshifts, outdir, resolution=1.):
    """ Runs 'main' at multiple redshifts and datasets to produce all desired grids

        Positional Arguments:
            datadir: path to directory with all the PART files and snapshot list
            datasets: list of all desired datasets e.g: '4/' for star particles
            normtypes: list of desired normalisations for each dataset (see 'main' docstring)
            weightings: list of desired weight columns for each dataset (see 'main' docstring)
            values: list of desired value columns for each dataset (see 'main' docstring)
            redshifts: list of desired redshifts
            outdir: directory to save all the grids

        Keyword Arguments:
            resolution: desired output resolution (Mpc h-1)
    """
    #set up the list of snapshots, and redshifts
    snapshot_list = np.loadtxt(f'{datadir}/Snapshots.txt', dtype=float, ndmin=2)
    time_list = snapshot_list[:,1]
    redshift_list = 1/time_list - 1

    for z in redshifts:
        idx = np.argmin(np.fabs(redshift_list - z))
        snap = int(snapshot_list[idx,0])
        partfile = f'{datadir}/PART_{snap:03d}'
        for n,d,v,w in zip(normtypes,datasets,values,weightings):
            dname = f'{d.strip("/")}_{v}'
            outname = f'{dname}_grid_{snap:03d}'

            if exists(f'{outdir}/{outname}'):
                logger.info('file %s already exists',outname)
                continue
            print(f'starting {outname}')
            main(partfile,outdir, outname,dataset=d,value=v,weight=w,norm=n,resolution=resolution)

if __name__ == '__main__':
    ap = argparse.ArgumentParser("get_xgrids.py")
    ap.add_argument("pfile", help='Particle data')
    ap.add_argument("--output", help='Name of bigfile to store the mesh')
    ap.add_argument("--outname", default='grid', help='name of the dataset to write to')
    ap.add_argument("--pos", help='Name of the position column')
    ap.add_argument("--weight", help='Name of the weight column')
    ap.add_argument("--value", help='Name of the value column')
    ap.add_argument("--dataset", help='Name of the particle dataset')
    ap.add_argument("--norm", help='normalisation type "global" "local" or "none"')
    ap.add_argument("--resolution", type=float, default=1.0, help='resolution in Mpc/h')
    ns = ap.parse_args()
    main(ns.pfile,ns.output,ns.outname,pos=ns.pos,weight=ns.weight,dataset=ns.dataset,resolution=ns.resolution,value=ns.value,norm=ns.norm)
