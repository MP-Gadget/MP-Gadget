"""Quick script to plot an image of structure."""

import argparse
import os.path
import numpy as np
from nbodykit.lab import BigFileCatalog
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt

def plot_image(snapshot, dataset=1, colorbar=True, Nmesh=None):
    """Make a pretty picture of the mass distribution."""
    cat = BigFileCatalog(snapshot, dataset=str(dataset)+'/', header='Header')
    if Nmesh is None:
        Nmesh = 2*int(np.round(np.cbrt(cat.attrs['TotNumPart'][dataset])))
    box = cat.attrs['BoxSize']/1000
    mesh = cat.to_mesh(Nmesh=Nmesh)
    plt.clf()
    plt.imshow(np.log10(mesh.preview(axes=(0, 1))/Nmesh), extent=(0,box,0,box))
    if colorbar:
        plt.colorbar()
    plt.xlabel("x (Mpc/h)")
    plt.ylabel("y (Mpc/h)")
    plt.tight_layout()
    snap = os.path.basename(os.path.normpath(snapshot))
    plt.savefig("dens-plt-type"+str(dataset)+snap+".pdf")
    plt.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('snapshot', type=str, help='snapshot directory')
    parser.add_argument('--type', type=int, default=1, help='type of particle to plot',required=False)
    parser.add_argument('--nmesh', type=int, default=None, help='mesh size',required=False)
    args = parser.parse_args()
    plot_image(args.snapshot, dataset=args.type, Nmesh=args.nmesh)
