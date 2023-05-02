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

    meshcdm = catcdm.to_mesh(Nmesh=Nmesh, resampler='cic', compensated=True, interlaced=True)
    meshb = catb.to_mesh(Nmesh=Nmesh, resampler='cic', compensated=True, interlaced=True)
    pkb = FFTPower(catb, mode='1d', Nmesh=Nmesh)
    pkcdm = FFTPower(catcdm, mode='1d', Nmesh=Nmesh)
    z = 1. / catcdm.attrs['Time'] - 1
    box = catcdm.attrs['BoxSize']
    omegab = catcdm.attrs['OmegaBaryon']
    omega0 = catcdm.attrs['Omega0']
    return pkcdm.power, pkb.power, z, box, omegab, omega0

def test_power(output, transfer, IC=False):
    """Check the initial power against linear theory and a linearly grown IC power"""
    print('testing', output)
    pkcdm, pkb, z, box, omegab, omega0 = compute_power(output)

    #Check types have the same power
    fig = Figure()
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    ctf = os.path.join(transfer, "class_tk_99.dat")
    if not IC:
        ctf += "-"+str(int(z[0]))+"*"
    # Transfer: 1:k (h/Mpc)              2:d_g                    3:d_b                    4:d_cdm                  5:d_ur        6:d_ncdm[0]              7:d_ncdm[1]              8:d_ncdm[2]              9:d_tot                 10:phi     11:psi                   12:h                     13:h_prime               14:eta                   15:eta_prime     16:t_g                   17:t_b                   18:t_ur        19:t_ncdm[0]             20:t_ncdm[1]             21:t_ncdm[2]             22:t_tot
    trans = np.loadtxt(glob.glob(ctf)[0])
    ax.plot(trans[:,0], (trans[:,2]/trans[:,3])**2, ls="--", label='CLASS bar / DM')

    #Note k in kpc/h
    ax.plot(pkcdm['k'][1:], pkb['power'][1:].real / pkcdm['power'][1:].real, label="Sim bar / DM")

    ax.set_title("z="+str(z))
    ax.set_xscale('log')
    ax.set_xlim([2*np.pi/box/2, 2*np.pi/box*4000])
    ax.set_ylim(0.35, 1.05)
    ax.legend()
    fig.savefig(output + '.png')
    fig = Figure()
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
#     ax.plot(pkcdm['k'][1:]*1e3, pkb['power'][1:].real/1e9, label="bar")
#     ax.plot(pkcdm['k'][1:]*1e3, pkcdm['power'][1:].real/1e9, label="DM")
    cmf = os.path.join(transfer, "class_pk_99.dat")
    if not IC:
        cmf += "-"+str(int(z[0]))+"*"
    mat = np.loadtxt(glob.glob(cmf)[0])
    ttot = (omegab * trans[:,2] + (omega0 - omegab) * trans[:,3])/omega0
    intpbar = scipy.interpolate.interp1d(trans[:,0], trans[:,2]/ttot)
    intpdm = scipy.interpolate.interp1d(trans[:,0], trans[:,3]/ttot)
    intpbarpk = scipy.interpolate.interp1d(mat[:,0], intpbar(mat[:,0])**2 * mat[:,1])
    intpdmpk = scipy.interpolate.interp1d(mat[:,0], intpdm(mat[:,0])**2 * mat[:,1])
    ax.plot(pkb['k'][1:], pkb['power'][1:].real/intpbarpk(pkb['k'][1:]) , label="Sim bar / CLASS bar")
    ax.plot(pkcdm['k'][1:], pkcdm['power'][1:].real/intpdmpk(pkcdm['k'][1:]) , label="Sim DM / CLASS DM")
    ax.axhline(1, ls="--")
    ax.set_xscale('log')
    ax.set_ylim(0.8, 1.2)
    ax.set_xlim(2*np.pi/box/2, 2*np.pi/box*4000)
    ax.legend()
    fig.savefig(output + '-full.png')

    final = 5
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
refcdm, refb, ref_z, box, omegab, omega0 = compute_power('output/IC')
pkin = np.loadtxt("class_pk_99.dat")
pklin = scipy.interpolate.interp1d(pkin[:,0], pkin[:,1])
#This checks that the power spectrum loading and rescaling code is working.
genpk = numpy.loadtxt("output/inputspec_IC.txt")
ii = np.where(genpk[:,0] < pkin[-1, 0])
assert_allclose(pklin(genpk[ii,0]), genpk[ii,1], rtol=2e-2, atol=0.0)
#This checks that the output is working
test_power('output/IC', ".", IC=True)
for pp in range(3):
    test_power('output/PART_'+str(pp).rjust(3,'0'), ".", IC=(pp==0))
