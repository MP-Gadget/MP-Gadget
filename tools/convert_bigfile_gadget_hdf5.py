"""
Script to convert between MP-Gadget formatted bigfile snapshots and Gadget-3 formatted HDF5 snapshots.
Both directions are supported.

This is to make it easier for simulators to start using MP-Gadget.

Known name changes are accounted for: for example MP-Gadget has "Position" and Gadget-3 has "Coordinates".

No unit conversion is done! Be careful with older Gadgets which do not store units in the snapshot header.

TODO: Support Black hole data arrays better.
"""

import argparse
import os
import os.path
import re
import glob
import bigfile
import h5py
import numpy as np
try:
    import hdf5plugin
except ImportError:
    pass


class NameMaps:
    """Maps names between HDF5 and Bigfile."""
    def __init__(self):
        #Map of (only changed) block names between HDF5 and bigfile snapshots.
        self.hdf_to_bigfile_map = { "Coordinates" : "Position",
                                    "Velocities": "Velocity", "Masses": "Mass",
                                    "NeutralHydrogenAbundance": "NeutralHydrogenFraction",
                                    "GFM_Metallicity": "Metallicity",
                                    "ParticleIDs" : "ID",
                                  }
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
    hattr["Flag_DoublePrecision"] = (bf["1/Position"].dtype == np.float64)
    hattr["Flag_IC_Info"] = 0
    hattr["Flag_Entropy_ICs"] = 0
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
    # Variables for the velocity factors
    pecvel = bf["Header"].attrs["UsePeculiarVelocity"]
    atime = bf["Header"].attrs["Time"]

    #detect ICs by the lack of a DM mass field
    ics = False
    try:
        bf["1/Mass"]
    except bigfile.pyxbigfile.Error:
        ics = True

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
            assert bname != ''
            #Deal with groups that don't contain particles: just write them to the first file.
            if ptype == '':
                if fnum == 0:
                    hgr = hdf5.create_group(block)
                    hgr = bf[block]
                continue
            ptype = int(ptype)
            hname = names.get_hdf5_name(bname)
            bfdata = bf[block][startpart[ptype]:endpart[ptype]]
            #For velocity, Gadget-2/3 uses a^{1/2} dx / dt
            #(where x is comoving distance).
            if bname == "Velocity":
                if pecvel:
                    #pecvel is a dx/dt: (physical peculiar velocity)
                    bfdata /= np.sqrt(atime)
                else:
                    #MP-Gadget old-school snapshots, used in BlueTides.
                    # vel is a^2 dx/dt in snapshots and dx/dt /sqrt(a)
                    #in the ICs.
                    if not ics:
                        bfdata /= atime**(3/2.)
            hdf5["PartType"+str(ptype)][hname] = bfdata
        #Gadget-3 requires an InternalEnergy block for ICs, even if it is zero.
        if "InternalEnergy" not in hdf5["PartType0"].keys():
            hdf5["PartType0"]["InternalEnergy"] = np.zeros(endpart[0]-startpart[0])

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
    try:
        mm = re.search("PART_([0-9]*)", bfname)
        nsnap = '000'
        if len(mm.groups()) > 0:
            nsnap = mm.groups()[0]
        hdf5name = os.path.join(hdf5name, "snap_"+nsnap)
    except AttributeError:
        hdf5name = os.path.join(hdf5name, bfname)
    print("Writing %d hdf snapshot files to %s" % (nfiles, hdf5name))
    for nn in range(nfiles):
        write_hdf_file(bf, hdf5name, nn, nfiles)
        print("Wrote file %d" % nn)

def write_bigfile_header(hdf5, bf):
    """Write out a header in the bigfile format. Default units are assumed."""
    bf.create("Header")
    battr = bf["Header"].attrs
    hattr = hdf5["Header"].attrs
    battr["BoxSize"] = hattr["BoxSize"]
    #As a relic from Gadget-1, the total particle numbers
    #are written as two separate 32 bit integers.
    battr["TotNumPart"] = np.uint64(hattr["NumPart_Total_HighWord"])*2**32 + np.uint64(hattr["NumPart_Total"])
    #Guess at the initial particle numbers
    battr["TotNumPartInit"] = battr["TotNumPart"]
    battr["TotNumPartInit"][0] += battr["TotNumPartInit"][4]
    battr["TotNumPartInit"][4] = 0
    battr["TotNumPartInit"][5] = 0
    #Guess at this. It only really matters for the neutrino model, which isn't present in Gadget-3.
    battr["TimeIC"] = np.min([hattr["Time"], 0.01])
    try:
        for attr in ["UnitLength_in_cm", "UnitMass_in_g", "UnitVelocity_in_cm_per_s"]:
            battr[attr] = hattr[attr]
    #Fall back to default unit system
    except KeyError:
        battr["UnitLength_in_cm"] = 3.085678e+21
        battr["UnitMass_in_g"] = 1.989e43
        battr["UnitVelocity_in_cm_per_s"] = 100000.
    #Some flags
    battr["UsePeculiarVelocity"] = 1
    #Should be 1/Time^2/Hubble, but we don't know cosmology.
    battr["RSDFactor"] = np.nan
    #Pass other keys through unchanged. We whitelist expected keys to avoid confusing Gadget.
    hdfats = ["MassTable", "Time", "BoxSize", "Omega0", "OmegaLambda", "HubbleParam"]
    for attr in hdfats:
        battr[attr] = hattr[attr]
    return hattr["Time"]

def write_bf_segment(bf, hfile, startpart, atime):
    """Write the data arrays to an HDF5 file."""
    #Open the file
    with h5py.File(hfile,'r') as hdf5:
        endpart = startpart + hdf5["Header"].attrs["NumPart_ThisFile"]
        for ptype in range(6):
            try:
                htype = hdf5["PartType"+str(ptype)]
            except KeyError:
                continue
            for hname in htype.keys():
                block = names.get_bigfile_name(hname)
                bname = "%d/%s" % (ptype, block)
                harray = np.array(hdf5["PartType"+str(ptype)][hname])
                # Convert velocity units to peculiar velocity, as MP-Gadget expects.
                if hname == "Velocities":
                    harray *= np.sqrt(atime)
                #Beware this is not checked.
                bf[bname].write(startpart[ptype], harray)
        return endpart

def create_big_file_arrays(bf, hfile):
    """Pre-create all the big file arrays with the desired sizes."""
    npart_tot = bf["Header"].attrs["TotNumPart"]
    nfile = hfile["Header"].attrs["NumFilesPerSnapshot"]
    for ptype in range(6):
        try:
            htype = hfile["PartType"+str(ptype)]
        except KeyError:
            continue
        for hname in htype.keys():
            hshape = np.shape(htype[hname])
            dtype = htype[hname].dtype
            block = names.get_bigfile_name(hname)
            bname = "%d/%s" % (ptype, block)
            try:
                rows = hshape[1]
            except IndexError:
                rows = 1
            bf.create(bname, dtype=(dtype, rows), size=npart_tot[ptype], Nfile=nfile)
        #We need these (generally non-present) arrays but the simulation will start even if they are zero.
        if ptype == 0 and "Generation" not in htype.keys():
            genzero = np.zeros(npart_tot[ptype],dtype=np.uint8)
            bf.create_from_array("0/Generation", genzero)
        #MP-Gadget needs a mass array: Gadget-3 reads it from the header if all masses are the same.
        if "Mass" not in htype.keys():
            genzero = hfile["Header"].attrs["MassTable"][ptype] * np.ones(npart_tot[ptype],dtype=np.float32)
            bf.create_from_array("%d/Mass" % ptype, genzero)


def write_big_file(bfname, hdf5name):
    """Find all the HDF5 files in the snapshot and merge them into a bigfile."""
    #Find all the HDF5 snapshot set.
    hdf5_files = glob.glob(hdf5name)
    if len(hdf5_files) == 0:
        hdf5_files = glob.glob(hdf5name+".*.hdf5")
    elif os.path.isdir(hdf5_files[0]):
        hdf5_files = glob.glob(os.path.join(hdf5name,"*_[0-9][0-9][0-9].*.hdf5"))
    if len(hdf5_files) == 0:
        raise IOError("Could not find hdF5 snapshot as %s (.*.hdf5)" % hdf5name)
    #Sort so we get a consistent answer each time.
    hdf5_files = sorted(hdf5_files)
    if not h5py.is_hdf5(hdf5_files[0]):
        raise IOError("%s is not hdf5!" % hdf5_files[0])
    hdf5 = h5py.File(hdf5_files[0], 'r')
    bf = bigfile.BigFile(bfname, create=True)
    atime = write_bigfile_header(hdf5, bf)
    for n in range(6):
        bf.create(str(n))
    create_big_file_arrays(bf, hdf5)
    hdf5.close()
    startpart = np.zeros(6, dtype=int)
    for hfile in hdf5_files:
        startpart = write_bf_segment(bf, hfile, startpart, atime)
        print("Copied HDF file %s" % hfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Input snapshot to convert.')
    parser.add_argument('--output', type=str, help='Output directory for converted files. HDF5 will create $output/snap_$n.hdf5.')
    parser.add_argument('--oformat', type=str, default="hdf5", help='Output format. Should be hdf5 or bigfile.',required=False)
    parser.add_argument('--iformat', type=str, default="bigfile", help='Input format. Should be bigfile or hdf5.',required=False)

    args = parser.parse_args()
    assert args.oformat == "hdf5" or args.oformat == "bigfile"
    assert args.iformat == "bigfile" or args.iformat == "hdf5"
    assert args.iformat != args.oformat
    if args.oformat == "bigfile":
        write_big_file(args.output, args.input)
    else:
        write_all_hdf_files(args.output, args.input)
