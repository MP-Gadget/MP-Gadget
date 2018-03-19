from nbodykit.lab import *
import glob
import os
import os.path
import numpy as np
import scipy.interpolate
from numpy.testing import assert_allclose

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
    z = 1. / catcdm.attrs['Time'] - 1
    box = catcdm.attrs['BoxSize']
    return pkcdm.power, pkb.power, z, box

def test_power(output, camb_transfer):
    """Check the initial power against linear theory and a linearly grown IC power"""
    print('testing', output)
    pkcdm, pkb, z, box = compute_power(output)

    #Check types have the same power
    fig = Figure()
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    ctf = os.path.join(camb_transfer, "camb_transfer_99.dat")
    if z < 98.5:
        ctf += "-"+str(int(z[0]))+"*"
    camb_trans = np.loadtxt(glob.glob(ctf)[0])
    ax.plot(camb_trans[:,0], (camb_trans[:,2]/camb_trans[:,1])**2, ls="--", label='CAMB bar / DM')

    #Note k in kpc/h
    ax.plot(pkcdm['k'][1:], pkb['power'][1:].real / pkcdm['power'][1:].real, label="Sim bar / DM")

    ax.set_title("z="+str(z))
    ax.set_xscale('log')
    ax.set_xlim(2*np.pi/box/2, 2*np.pi/box*4000)
    ax.set_ylim(0.35, 1.05)
    ax.legend()
    fig.savefig(output + '.png')
    fig = Figure()
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
#     ax.plot(pkcdm['k'][1:]*1e3, pkb['power'][1:].real/1e9, label="bar")
#     ax.plot(pkcdm['k'][1:]*1e3, pkcdm['power'][1:].real/1e9, label="DM")
    cmf = os.path.join(camb_transfer, "camb_matterpow_99.dat")
    if z < 98.5:
        cmf += "-"+str(int(z[0]))+"*"
    camb_mat = np.loadtxt(glob.glob(cmf)[0])
    intpdm = scipy.interpolate.interp1d(camb_trans[:,0], camb_trans[:,1]/camb_trans[:,6])
    intpbar = scipy.interpolate.interp1d(camb_trans[:,0], camb_trans[:,2]/camb_trans[:,6])
    intpdmpk = scipy.interpolate.interp1d(camb_mat[:,0], intpdm(camb_mat[:,0])**2 * camb_mat[:,1])
    intpbarpk = scipy.interpolate.interp1d(camb_mat[:,0], intpbar(camb_mat[:,0])**2 * camb_mat[:,1])
    ax.plot(pkb['k'][1:], pkb['power'][1:].real/intpbarpk(pkb['k'][1:]) , label="Sim bar / CAMB bar")
    ax.plot(pkcdm['k'][1:], pkcdm['power'][1:].real/intpdmpk(pkcdm['k'][1:]) , label="Sim DM / CAMB DM")
    ax.axhline(1, ls="--")
    ax.set_xscale('log')
    ax.set_ylim(0.8, 1.2)
    ax.set_xlim(2*np.pi/box/2, 2*np.pi/box*4000)
    ax.legend()
    fig.savefig(output + '-full.png')

    final = 10
    # asserting the linear growth is reasonably accurate
    assert_allclose(pkcdm['power'][2:final],
            intpdmpk(pkcdm['k'][2:final]),
                        rtol=0.04, atol=0.0)

    # We don't have enough particles to get the BAO wiggles
    assert_allclose(pkb['power'][5:final],
                    intpbarpk(pkb['k'][5:final]),
                        rtol=0.05, atol=0.0)


#This checks that the power spectrum loading and rescaling code is working.
# asserting the initial power spectrum is 1% accurate
print("testing IC power")
refcdm, refb, ref_z, box= compute_power('output/IC')
pkin = np.loadtxt("camb_matterpow_99.dat")
pklin = scipy.interpolate.interp1d(pkin[:,0], pkin[:,1])
#This checks that the power spectrum loading and rescaling code is working.
genpk = numpy.loadtxt("output/inputspec_IC.txt")
assert_allclose(pklin(genpk[:,0]), genpk[:,1], rtol=2e-2, atol=0.0)
#This checks that the output is working
#This checks that the output is working
test_power('output/IC', ".")
for pp in range(3):
    test_power('output/PART_'+str(pp).rjust(3,'0'), ".")
