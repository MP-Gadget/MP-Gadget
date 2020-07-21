"""Script to convert from Gadget-1 format to MP-Gadget bigfile. Uses nbodykit. Modified from fastpm:
https://github.com/rainwoodman/fastpm/blob/master/python/convert-from-gadget-1.py

Tested to convert ICs.

Be warned that this is Gadget-1, so there is no metadata in the snapshot and
columns may be mis-interpreted.

If you need to convert a multi-file archive, pass the filename base, ie, excluding ".0"
"""
# This uses nbodykit

from argparse import ArgumentParser
import glob
import os.path
import numpy
import dask
from nbodykit.source.catalog import Gadget1Catalog

ap = ArgumentParser()
ap.add_argument('source', help='Gadget filename base, EXCLUDING the ".0".')
ap.add_argument('dest', help='Bigfile snapshot; dir will be created on the fly.')
ap.add_argument('--time-ic', type=float, default=None, help='Time of IC of this simulation, default is the time of snapshot')
ap.add_argument('--unit-system', choices=['Mpc', 'Kpc'], default='Kpc')
ap.add_argument('--subsample', type=int, help='keep every n particles')

def getg1cat(root):
    """glob multiple files to get all the Gadget type 1 snapshot files in order"""
    if os.path.exists(root):
        cat = Gadget1Catalog(root)
    else:
        #Ensure files are sorted
        gg =  sorted(glob.glob(root+".?"))
        gg += sorted(glob.glob(root+".??"))
        gg += sorted(glob.glob(root+".???"))
        gg += sorted(glob.glob(root+".????"))
        assert len(set(gg)) == len(gg)
        cat = Gadget1Catalog(gg)
    print("Loaded %d files" % len(gg))
    return cat

def main(ns):
    """Load the Gadget 1 Catalog and do the conversion, taking care of velocity units."""
    cat = getg1cat(ns.source)

    #Fix up the header
    attrs = cat.attrs.copy()
    cat.attrs.clear()

    cat.attrs['MassTable'] = attrs['Massarr']
    cat.attrs['TotNumPart'] = numpy.int64(attrs['Nall']) + (numpy.int64(attrs['NallHW']) << 32)
    cat.attrs['TotNumPartInit'] = numpy.int64(attrs['Nall']) + (numpy.int64(attrs['NallHW']) << 32)
    cat.attrs['BoxSize'] = attrs['BoxSize']
    cat.attrs['Time'] = attrs['Time']

    if ns.time_ic is None:
        ns.time_ic = attrs['Time']
    cat.attrs['TimeIC'] = ns.time_ic

    cat.attrs['UnitVelocity_in_cm_per_s'] = 1e5
    if ns.unit_system == 'Mpc':
        cat.attrs['UnitLength_in_cm'] = 3.085678e24
    if ns.unit_system == 'Kpc':
        cat.attrs['UnitLength_in_cm'] = 3.085678e21

    cat.attrs['UnitMass_in_g'] = 1.989e43

    # The velocity convention is weird without this
    cat.attrs['UsePeculiarVelocity'] = True

    a = attrs['Time']
    #Convert from Gadget-format velocities.
    cat['Velocity'] = cat['GadgetVelocity'] * a ** 0.5

    # The IDs may wrap around after 2^32 and become non-unique.
    #This should hopefully be small
    ii = dask.array.argwhere(cat["ID"] == cat["ID"][0]).compute()
    #Add 2**32 for each wrap around
    if ii.size > 1:
        for ind in ii[1:]:
            cat["ID"] += 2**32*(dask.array.arange(cat["ID"].size) >= ind)
    if ns.subsample is not None:
        cat = cat[::ns.subsample]

    ptypes = [str(pt) for pt in range(6)  if cat.attrs['TotNumPart'][pt] > 0]

    #Use all existing columns, remove
    columns = cat.columns
    columns.remove("GadgetVelocity")
    columns.remove("Weight")
    columns.remove("Value")
    columns.remove("Selection")

    #If we have all needed mass entries from the mass table, don't need a mass array.
    #This usually means ICs.
    gotmass = sum([cat.attrs['MassTable'][pt] > 0 for pt in range(6) if cat.attrs['TotNumPart'][pt] > 0 ])
    if gotmass:
        columns.remove("Mass")

    print("Keeping cols: ",columns," for types: ",ptypes)
    for pt in ptypes:
        cat.save(ns.dest, columns=columns, dataset=pt, header='Header')

if __name__ == '__main__':
    args = ap.parse_args()

    main(args)
