from nbodykit.lab import *
import os
import os.path
import numpy as np
import scipy.interpolate

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

def compute_power(output):
    """Compute the compensated power spectrum from a catalogue."""
    catcdm = BigFileCatalog(output, dataset='1/', header='Header')

    try:
        catb = MultipleSpeciesCatalog(["gas", "star"],
                BigFileCatalog(output, dataset='0/', header='Header'),
                BigFileCatalog(output, dataset='4/', header='Header'))
    except:
        catb = BigFileCatalog(output, dataset='0/', header='Header')
    box = catcdm.attrs['BoxSize']
    Nmesh = 2*int(np.round(np.cbrt(catcdm.attrs['TotNumPart'][0])))

    meshcdm = catcdm.to_mesh(Nmesh=Nmesh, window='cic', compensated=True, interlaced=True)
    meshb = catb.to_mesh(Nmesh=Nmesh, window='cic', compensated=True, interlaced=True)
    pkb = FFTPower(catb, mode='1d', Nmesh=Nmesh)
    pkcdm = FFTPower(catcdm, mode='1d', Nmesh=Nmesh)
    z = 1. / catcdm.attrs['Time'][0] - 1
    box = catcdm.attrs['BoxSize']
    omega0 = catcdm.attrs['Omega0']
    omegab = catcdm.attrs['OmegaBaryon']
    return pkcdm.power, pkb.power, z, box, omega0, omegab

def test_power(output, camb_transfer, IC=False):
    """Check the initial power against linear theory and a linearly grown IC power"""
    print('testing', output)
    pkcdm, pkb, z, box,omega0, omegab = compute_power(output)

    box/=1e3
    #Check types have the same power
    fig = Figure()
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    zstr = "-"+str(np.round(z,1))
    if IC:
        zstr = ""

    try:
        txt = camb_transfer + "_transfer.dat"+zstr
        camb_trans = np.loadtxt(txt)
        ax.plot(camb_trans[:,0], (camb_trans[:,2]/camb_trans[:,3])**2, ls="--", label='CLASS bar / DM')
        #Build the total power to exclude radiation
        camb_trans[:,5] = (omegab * camb_trans[:,2] + (omega0 - omegab) * camb_trans[:,3])/omega0
    except (IOError,FileNotFoundError):
        pass

    #Note k in kpc/h
    ax.plot(pkcdm['k'][1:]*1e3, pkb['power'][1:].real / pkcdm['power'][1:].real, label="Sim bar / DM")

    ax.set_title("z="+str(z))
    ax.set_xscale('log')
    ax.set_xlim(2*np.pi/box/2, 2*np.pi/box*4000)
    ax.set_ylim(np.min(pkb['power'][1:].real/pkcdm['power'][1:].real),1.05)
    ax.legend()
    fig.savefig(output + '.png')
    fig = Figure()
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    try:
        camb_mat = np.loadtxt(camb_transfer+"_matterpow.dat"+zstr)
        intpdm = scipy.interpolate.interp1d(camb_trans[:,0], camb_trans[:,3]/camb_trans[:,5])
        intpbar = scipy.interpolate.interp1d(camb_trans[:,0], camb_trans[:,2]/camb_trans[:,5])
        intpdmpk = scipy.interpolate.interp1d(camb_mat[:,0], intpdm(camb_mat[:,0])**2 * camb_mat[:,1])
        intpbarpk = scipy.interpolate.interp1d(camb_mat[:,0], intpbar(camb_mat[:,0])**2 * camb_mat[:,1])
        ax.plot(pkb['k'][1:]*1e3, pkb['power'][1:].real/1e9/intpbarpk(pkb['k'][1:]*1e3) , label="Sim bar / CAMB bar")
        ax.plot(pkcdm['k'][1:]*1e3, pkcdm['power'][1:].real/1e9/intpdmpk(pkcdm['k'][1:]*1e3) , label="Sim DM / CAMB DM")
        ax.axhline(1, ls="--")
    except (IOError,FileNotFoundError):
        pass
    ax.set_xscale('log')
    ax.set_ylim(0.85, 1.10)
    ax.set_xlim(2*np.pi/box/2, 2*np.pi/box*4000)
    ax.legend()
    fig.savefig(output + '-full.png')

#This checks that the power spectrum loading and rescaling code is working.
#This checks that the output is working
test_power('output/IC', "linear_growth", IC=True)
for pp in range(8):
    test_power('output/PART_'+str(pp).rjust(3,'0'), "linear_growth")
