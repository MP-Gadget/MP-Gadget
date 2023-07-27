"""A script to read binary files BlackholeDetails/* in MP-GADGET simulation box
and store desired properties in text file"""

# importing required modules
import numpy as np
import glob
import struct
import os

# Dictionary mapping black hole properties to their corresponding data types
# BHinfo struct contents,
# check bhinfo.c to make sure that they are consistent!
# dict description: key --> BH property, value --> datatype of property as stored in C
content = {
    "BHID": "Q",
    "BHMass": "d",
    "Mdot": "d",
    "Density": "d",
    "Timebin": "i",
    "Encounter": "i",
    "MinPos": "3d",
    "MinPot": "d",
    "Entropy": "d",
    "GasVel": "3d",
    "acMom": "3d",
    "acMass": "d",
    "acBHMass": "d",
    "FdbkWgtSum": "d",
    "SPHSwallowID": "Q",
    "SwallowID": "Q",
    "CountProgs": "i",
    "Swallowed": "i",
    "BHpos": "3d",
    "BH_SurroundingDensity": "d",
    "BH_SurroundingParticles": "d",
    "BH_SurroundingVel": "3d",
    "BH_SurroundingRmsVel": "d",
    "DFAccel": "3d",
    "DragAccel": "3d",
    "GravAccel": "3d",
    "BHvel": "3d",
    "Mtrack": "d",
    "Mdyn": "d",
    "KineticFdbkEnergy": "d",
    "NumDM": "d",
    "V1sumDM": "3d",
    "V2sumDM": "d",
    "MgasEnc": "d",
    "KEflag": "i",
    "time": "d",
}


def get_bh_info(bhfile_list, selected_keys, searchID=None):
    """
    Extracts black hole information from binary files in BlackholeDetails* folder

    Parameters:
        bhfile_list (list): List of black hole binary file paths.
        selected_keys (list): List of properties to extract from each entry.
        searchID (int64, optional): Specific BHID to read. If None, all BHs are read. Default is None.

    Returns:
        numpy.ndarray: Structured array containing the selected properties.
    """

    keys = np.array(list(content.items()))[:, 0]  # Names of each field in the struct
    sizes = [
        np.dtype(x).itemsize for x in np.array(list(content.items()))[:, 1]
    ]  # Size of each field in the struct
    chunk_size = np.sum(sizes) + 8  # Total size of the struct, including padding
    offset = (
        np.append(0, np.cumsum(np.array(sizes))) + 4
    )  # Starting position of each field in the struct

    ixs = [
        np.where(keys == x)[0][0] for x in selected_keys
    ]  # Indices of selected properties
    s_sizes = [sizes[x] for x in ixs]  # Sizes of selected properties
    s_offset = [offset[x] for x in ixs]  # Starting positions of selected properties
    s_types = [content[x] for x in selected_keys]  # Data types of selected properties

    if searchID is not None:
        off0 = offset[np.where(keys == "BHID")[0][0]]  # Starting position of BHID
        s0 = sizes[np.where(keys == "BHID")[0][0]]  # Size of BHID

    data_dict = {
        key: [] for key in selected_keys
    }  # Dictionary to store selected properties

    for filename in bhfile_list:
        try:
            file_size = os.path.getsize(filename)  # Get the size of the file
        except OSError as e:
            print(f"Failed to get the size of file '{filename}': {e}")
            continue

        if file_size % 436 != 0:
            print(
                f"File '{filename}' does not have a size that is a multiple of 436 bytes."
            )
        else:
            try:
                with open(filename, "rb") as f:
                    while True:
                        buf = f.read(chunk_size)  # Read a chunk of binary data
                        if not buf:
                            break

                        if searchID is None:
                            # Unpack the selected properties from the binary chunk
                            for x, lgt, off, tp in zip(
                                selected_keys, s_sizes, s_offset, s_types
                            ):
                                data_dict[x].append(
                                    struct.unpack(tp, buf[off : off + lgt])
                                )
                        else:
                            sID = struct.unpack("q", buf[off0 : off0 + s0])[
                                0
                            ]  # Read BHID from the binary chunk
                            if sID == searchID:
                                # Unpack the selected properties from the binary chunk
                                for x, lgt, off, tp in zip(
                                    selected_keys, s_sizes, s_offset, s_types
                                ):
                                    data_dict[x].append(
                                        struct.unpack(tp, buf[off : off + lgt])
                                    )
            except OSError as e:
                print(f"Failed to open file '{filename}': {e}")
                continue

    # Create a structured NumPy array to store the extracted properties
    data = np.zeros(
        len(data_dict[selected_keys[0]]),
        dtype={"names": tuple(selected_keys), "formats": tuple(s_types)},
    )

    if len(data_dict) == 0:
        print("No BH found!")
        return

    # Assign the extracted data to the structured array
    for x, tp in zip(selected_keys, s_types):
        d = np.array(data_dict[x])
        if tp != "3d":
            d = np.concatenate(d)
        data[x] = d

    # Add a "redshift" field based on the "time" field
    dt = data.dtype.descr + [("redshift", "<f8")]
    data_z = np.empty(data.shape, dt)
    for name in data.dtype.names:
        data_z[name] = data[name]
    data_z["redshift"] = 1.0 / data_z["time"] - 1

    return data_z
