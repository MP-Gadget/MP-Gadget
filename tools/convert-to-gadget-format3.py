"""
Scripts to convert MP-Gadget formatted bigfile to other Gadget formats.
This is for incorporation in existing pipelines, and aims to make it easier for simulators
to start using MP-Gadget.

Formats supported:
    - hdf5.

TODO: Support Gadget-1,2.
TODO: Support Black hole data arrays better.
TODO: Support conversion of HDF5/Gadget-2 to bigfile.

Known name changes are accounted for: for example MP-Gadget has "Position" and Gadget-3 has "Coordinates".

No unit conversion is done! Be careful with older Gadget's which do not store units in the snapshot header.
"""

from __future__ import print_function
import argparse
import os
import os.path
import re
import bigfile
import h5py
import numpy as np


class NameMaps(object):
    """Maps names between HDF5 and Bigfile."""
    def __init__(self):
        #Map of (only changed) block names between HDF5 and bigfile snapshots.
        self.hdf_to_bigfile_map = { "Coordinates" : "Position",
                                    "Velocities": "Velocity", "Masses": "Mass",
                                    "NeutralHydrogenAbundance": "NeutralHydrogenFraction",
                                    "GFM_Metallicity": "Metallicity",
                                    "ParticleIDs" : "ID",
                                  }
        #This requires python 2.7
        self.bigfile_to_hdf_map = {v : k for (k,v) in self.hdf_to_bigfile_map.items()}
        #Leave the metallicity array unchanged.
        del self.bigfile_to_hdf_map["Metallicity"]

    def get_bigfile_name(self, hdfname):
        """Get the bigfile name from an HDF5 name."""
        try:
            return self.hdf_to_bigfile_map[hdfname]
        except KeyError:
            #Unrecognised names are assumed unchanged
            return hdfname

    def get_hdf5_name(self, bigfilename):
        """Get the HDF5 name from a bigfile name."""
        try:
            return self.bigfile_to_hdf_map[bigfilename]
        except KeyError:
            #Unrecognised names are assumed unchanged
            return bigfilename

#Global name registry
names = NameMaps()

def write_hdf_header(bf, hdf5, nfiles, npart_file):
    """Generate an HDF5 header from a bigfile header."""
    head = hdf5.create_group("Header")
    hattr = head.attrs
    battr = bf["Header"].attrs
    #As a relic from Gadget-1, the total particle numbers
    #are written as two separate 32 bit integers.
    hattr["NumPart_Total"] = np.uint32(battr["TotNumPart"] % 2**32)
    hattr["NumPart_Total_HighWord"] = np.uint32(battr["TotNumPart"] // 2**32)
    hattr["NumPart_ThisFile"] = np.int32(npart_file)
    hattr["NumFilesPerSnapshot"] = np.int32(nfiles)
    #Assume star formation implies the rest.
    flag_sfr = np.int32(battr["TotNumPart"][4] > 0)
    hattr["Flag_Sfr"] = flag_sfr
    hattr["Flag_Cooling"] = flag_sfr
    hattr["Flag_StellarAge"] = flag_sfr
    hattr["Flag_Metals"] = flag_sfr
    hattr["Flag_Feedback"] = 0
    hattr["Flag_DoublePrecision"] = 1
    hattr["Flag_IC_Info"] = 0
    hattr["Redshift"] = 1./battr["Time"] - 1
    #Pass other keys through unchanged. We whitelist expected keys to avoid confusing Gadget.
    hdfats = ["MassTable", "Time", "BoxSize", "Omega0", "OmegaLambda", "HubbleParam", "OmegaBaryon", "UnitLength_in_cm", "UnitMass_in_g", "UnitVelocity_in_cm_per_s"]
    for attr in hdfats:
        hattr[attr] = battr[attr]

def write_hdf_file(bf, hdf5name, fnum, nfiles):
    """Write the data arrays to an HDF5 file."""
    #Compute the particles to write
    npart = bf["Header"].attrs["TotNumPart"]
    startpart = fnum * (npart//nfiles)
    endpart = startpart + (npart//nfiles)
    if fnum == nfiles - 1:
        endpart = npart

    #Open the file
    hfile = hdf5name + "."+str(fnum)+".hdf5"
    if os.path.exists(hfile):
        raise IOError("Not over-writing existing file ",hfile)

    with h5py.File(hfile,'w') as hdf5:
        #Write the header
        write_hdf_header(bf, hdf5, nfiles, endpart-startpart)
        #Write the data
        for ptype in range(6):
            hdf5.create_group("PartType" + str(ptype))

        for block in bf.list_blocks():
            if block == "Header":
                continue
            ptype, bname = os.path.split(block)
            #Deal with groups that don't contain particles: just write them to the first file.
            if ptype == '':
                if fnum == 0:
                    hgr = hdf5.create_group(block)
                    hgr = bf[block]
                continue
            ptype = int(ptype)
            hname = names.get_hdf5_name(bname)
            hdf5["PartType"+str(ptype)][hname] =  bf[block][startpart[ptype]:endpart[ptype]]

def compute_nfiles(npart):
    """Work out how many files we need to split the snapshot into.
       We want less than 2^31 bytes per data array, and we want a power of two,
       so that we probably divide the particles evenly."""
    nfiles = 1
    #Largest possible data array: a 3-vector in double precision.
    maxarray = np.max(npart) * 3 * 8
    while maxarray // nfiles >= 2**31:
        nfiles *=2
    return nfiles

def write_all_hdf_files(hdf5name, bfname):
    """Work out which particle set goes to which HDF5 file and write it."""
    bf = bigfile.BigFile(bfname, 'r')
    nfiles = compute_nfiles(bf["Header"].attrs["TotNumPart"])
    if not os.path.exists(hdf5name):
        os.mkdir(hdf5name)
    mm = re.search("PART_([0-9]*)", bfname)
    nsnap = '000'
    if len(mm.groups()) > 0:
        nsnap = mm.groups()[0]
    hdf5name = os.path.join(hdf5name, "snap_"+nsnap)
    print("Writing %d hdf snapshot files to %s" % (nfiles, hdf5name))
    for nn in range(nfiles):
        write_hdf_file(bf, hdf5name, nn, nfiles)
        print("Wrote file %d" % nn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Input bigfile snapshot to convert.')
    parser.add_argument('--output', type=str, help='Output directory for converted HDF5 files. Will create $output/snap_$n.hdf5.')
    parser.add_argument('--oformat', type=str, default="hdf5", help='Output format. Only currently supported option is hdf5.',required=False)
    parser.add_argument('--iformat', type=str, default="bigfile", help='Input format. Only currently supported option is bigfile.',required=False)

    args = parser.parse_args()
    assert args.oformat == "hdf5"
    assert args.iformat == "bigfile"
    write_all_hdf_files(args.output, args.input)
