'''
simple grid plotting code
takes bigfiles with datasets named {prefix}_grid_z{redshift}
like those generated from get_xgrids.py and plots slices of them
in a number of grid types x number of redshifts grid
'''

import argparse
import numpy as np
import matplotlib
matplotlib.use('pdf')
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import gridspec as gs
import bigfile as bf

ap = argparse.ArgumentParser("get_xgrids.py")
ap.add_argument("gridfile", help='path to the grid bigfile')
ap.add_argument("--output", help='path to save the plots')
ap.add_argument("--snapshots", help='comma separated list of redshifts to plot')
ap.add_argument("--gridnames", default="d,xhi"
                , help='comma separated list of prefixes of the desired grids')
ap.add_argument("--slice-idx", type=int, default=5, help='index of the plotted slice')
ap.add_argument("--slice-depth", type=int, default=5, help='depth of the slice')
ap.add_argument("--show-plot", help="show plot with matplotlib", action="store_true")

ns = ap.parse_args()
snapshots = ns.snapshots.split(",")
grid_prefix = ns.gridnames.split(",")

n_snap = len(snapshots)
n_types = len(grid_prefix)

X_H = 0.76
Y_He = 1 - X_H

def u_to_t(uin,xhi):
    helium = 0.24
    #assuming hei ion with HI
    nep = (1-3/4*helium)*(1 - xhi)
    hy_mass = 1 - helium
    muienergy = 4 / (hy_mass * (3 + 4*nep) + 1)*uin
    temp = 2/3 * 1.6726e-24 / 1.38066e-16 * muienergy * 1e10
    return temp

#mapping of grids to plot position and prefix/redshift number in the array
grid_map = np.zeros((n_snap*n_types, 2), dtype=int)
c_lims = []
c_maps = []
#build the list of grid names by redshift and variable
#and save a map of where it exists in redshift/type space
grid_names = []
count = 0
for i, pref in enumerate(grid_prefix):
    for j, snap in enumerate(snapshots):
        grid_names.append(f'{pref}_grid_{snap}')
        grid_map[count,0] = j
        grid_map[count,1] = i
        count += 1

#open the file
fin = bf.File(ns.gridfile)
dset = bf.Dataset(fin, grid_names)

#build list of grid slices to plot
slab_idx = lambda x, i, d: x[i-d//2:i+d//2+1,:,:].mean(axis=0)
slab_list = []

for i, grid in enumerate(grid_names):
    #read in the grid
    buf = dset[grid].read(0, dset[grid].size)
    print(f'Mean value of {grid} is {buf.mean()}')

    #assuming a cube here
    length = int(np.cbrt(buf.size))
    buf = buf.reshape((length, length, length))
    
    if 'NeutralHydrogenFraction' in grid:
        print(f'vol weighted xhi {(buf > 0.9).mean()}')
    #grab the required slice
    buf = slab_idx(buf, ns.slice_idx, ns.slice_depth)
    if 'InternalEnergy' in grid:
        buf = u_to_t(buf,0.) #note, assuming ionised so neutral temps will be a bit off (not super important)
        grid = grid.replace('InternalEnergy','Temperature')

    slab_list.append(buf)



fin.close()

for i,p in enumerate(grid_prefix):
    if 'Value' in p:
        c_lims.append(matplotlib.colors.LogNorm(vmin=1e-1,vmax=1e1))
        c_maps.append(cm.Purples)
    elif 'NeutralHydrogenFraction' in p:
        c_lims.append(matplotlib.colors.Normalize(vmin=0,vmax=1))
        c_maps.append(cm.Blues)
    elif 'J21' in p:
        c_lims.append(matplotlib.colors.LogNorm(vmin=1e-3,vmax=1e1))
        c_maps.append(cm.viridis)
    elif 'InternalEnergy' in p:
        #its temperature now
        c_lims.append(matplotlib.colors.LogNorm(vmin=1e2,vmax=1e5))
        c_maps.append(cm.plasma)
    elif 'ZReionized' in p:
        c_lims.append(matplotlib.colors.Normalize(vmin=5,vmax=12))
        c_maps.append(cm.gist_rainbow_r)
    else:
        c_lims.append(matplotlib.colors.LogNorm())
        c_maps.append(cm.Purples)

#define length and layout of the plot grid
ext = [0, length, 0, length]
gs = gs.GridSpec(n_snap, n_types)

fig = plt.figure(figsize=(12, 10*n_snap/n_types))

#put each slice in the correct place based on redshift / type
for i, slab in enumerate(slab_list):
    pos = grid_map[i, :]
    ax = fig.add_subplot(gs[pos[0],pos[1]])

    im = ax.imshow(slab, cmap=c_maps[pos[1]], norm=c_lims[pos[1]], origin='lower', extent=ext)
    plt.colorbar(im)
    ax.set_title(f'{grid_names[i]}',fontsize=10)
    ax.tick_params(left=False,labelleft=False,bottom=False,labelbottom=False)

#fig.tight_layout()

if ns.output is not None:
    fig.savefig(ns.output)

if ns.show_plot:
    plt.show()
