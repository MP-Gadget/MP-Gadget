"""Quick script to plot an image of structure."""

import argparse
import os.path
import numpy as np
from nbodykit.lab import BigFileCatalog
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt

"""Quick script to plot an image of structure."""

import argparse
import os.path
import numpy as np
from nbodykit.lab import BigFileCatalog
import matplotlib
matplotlib.use('PDF')  
import matplotlib.pyplot as plt

def plot_image(snapshot, dataset=1, colorbar=True, Nmesh=None):
    """
    Generates a 2D image of the mass distribution from a BigFile snapshot.

    Parameters:
        snapshot (str): Path to the snapshot directory.
        dataset (int): Particle type to visualize (default: 1). Options: 0 = gas particles (baryons), 1 = CDM, 2 = neutrinos, 3 = unused, 4 = stars and 5 = black holes.
        colorbar (bool): Whether to show a colorbar.
        Nmesh (int): Resolution of the mesh (if None, computed from particle count).
    """
    # Load the particle catalog
    cat = BigFileCatalog(snapshot, dataset=str(dataset)+'/', header='Header')

    # If Nmesh not provided, estimate based on particle number
    if Nmesh is None:
        Nmesh = 2 * int(np.round(np.cbrt(cat.attrs['TotNumPart'][dataset])))

    # Ensure box size is a float, not an  - important for plotting
    box_attr = cat.attrs['BoxSize']
    box = float(box_attr[0]) / 1000 if isinstance(box_attr, np.ndarray) else box_attr / 1000

    # Create the density mesh
    mesh = cat.to_mesh(Nmesh=Nmesh)
    data = mesh.preview(axes=(0, 1)) / Nmesh  # Normalize
    data_log = np.log10(data.astype(np.float64))  # Use float64 for stability

    # Setup extent for plotting
    extent_tuple = (0., box, 0., box)

    # Plotting
    plt.clf()
    plt.imshow(data_log, extent=extent_tuple)
    if colorbar:
        plt.colorbar()
    plt.xlabel("x (Mpc/h)")
    plt.ylabel("y (Mpc/h)")
    plt.tight_layout()

    # Save figure
    snap = os.path.basename(os.path.normpath(snapshot))
    plt.savefig(f"dens-plt-type{dataset}{snap}.pdf")
    plt.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('snapshot', type=str, help='Snapshot directory')
    parser.add_argument('--type', type=int, default=1, help='Type of particle to plot')
    parser.add_argument('--nmesh', type=int, default=None, help='Mesh size for density field')
    args = parser.parse_args()
    plot_image(args.snapshot, dataset=args.type, Nmesh=args.nmesh)

