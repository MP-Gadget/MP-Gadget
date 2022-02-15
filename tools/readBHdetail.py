"""A short function to read BlackholeDetails/* binary files"""

import numpy as np
import struct
import glob

# the items in BHinfo struct, 
# check blackhole.c to make sure that they are consistent!
#-------------------------------------------
content = {'BHID':'q', 'BHMass':'d', 'Mdot':'d', 'Density':'d', 'Timebin':'i', 'Encounter':'i',
           'MinPos':'3d', 'MinPot':'d', 'Entropy':'d', 'GasVel':'3d', 'acMom':'3d', 'acMass':'d',
           'acBHMass':'d', 'FdbkWgtSum':'d', 'SPHSwallowID':'q', 'SwallowID':'q', 
           'CountProgs':'i', 'Swallowed':'i', 'BHpos':'3d', 'BH_SurroundingDensity':'d',
           'BH_SurroundingParticles':'d', 'BH_SurroundingVel':'3d', 'BH_SurroundingRmsVel':'d',
           'DFAccel':'3d','DragAccel':'3d','GravAccel':'3d','BHvel':'3d','Mtrack':'d','Mdyn':'d',
           'KineticFdbkEnergy':'d', 'NumDM':'d', 'V1sumDM':'3d', 
           'V2sumDM':'d', 'MgasEnc':'d', 'KEflag':'d','time':'d'}


def get_bh_info(bhfile_zoo,selected_keys,searchID=None):
    """
    Parameters
    ----------
    bhfile_zoo : list of the BH detail binary files
        For example, glob.glob('test-run/output/BlackholeDetails/*')
    selected_keys: list of the fields of interest
        For example, selected_keys = ['BHID','BHMass','Mdot','BHpos','time']
    searchID: int64, one particular BHID that will be read out, optional
        If set, BH with this searchID will be read out
        if searchID is None, all BHs will be read out
        Default is None
    
    Returns
    ----------
    A numpy structed array that stores the fields according to selected_keys
    """    
    
    # all fields
    keys = np.array(list(content.items()))[:,0] # name of each field in struct
    sizes = [np.dtype(x).itemsize for x in np.array(list(content.items()))[:,1]] # size of each field in struct
    chunk_size = np.sum(sizes)+8 # we pad the struct with chunksize
    offset = np.append(0,np.cumsum(np.array(sizes))) # starting of each field
    offset += 4 # the first 4 byte stores the chunksize 
    
    # selected fields
    ixs = [np.where(keys==x)[0][0] for x in selected_keys]
    s_sizes = [sizes[x] for x in ixs]
    s_offset = [offset[x] for x in ixs]
    s_types = [content[x] for x in selected_keys]
    
    if searchID is not None:
        off0 = offset[np.where(keys=='BHID')[0][0]]
        s0 = sizes[np.where(keys=='BHID')[0][0]]

    dct = {}
    for x in selected_keys:
        dct['%s'%x] = []
        
    def split(chunk):
        for x,lgt,off,tp in zip(selected_keys,s_sizes,s_offset,s_types):
            dct['%s'%x].append(struct.unpack(tp,chunk[off:off+lgt]))
            
    for filename in bhfile_zoo:
        f = open(filename,'rb')
        while True:
            buf = f.read(chunk_size)
            if not buf:
                f.close()
                break
            
            if searchID is None:              
                split(buf) 
            else:
                sID = struct.unpack('q',buf[off0:off0+s0])[0]
                if sID == searchID:
                    split(buf)
    #---------------------------------
    data = np.zeros(len(dct[selected_keys[0]]),
                dtype={'names':tuple(selected_keys),'formats':tuple(s_types)})
    
    if (len(dct)==0):
        print ("no BH found!")
        return
    
    for x,tp in zip(selected_keys,s_types):
        d = np.array(dct['%s'%x])
        if tp != '3d':
            d = np.concatenate(d)
        data[x] = d
    
    if np.isin('time',selected_keys):
        data['time'] = 1./data['time'] - 1 # convert from a_scale to z
    return data
  
