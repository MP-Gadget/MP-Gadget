"""Checks output of DM simulation with HMF and power"""
import os
import os.path
import numpy as np
from numpy.testing import assert_allclose
import scipy.interpolate
import bigfile

def massfunc(m,Lbox):
    """Get a mass function from a list of halo masses. Lbox should be in comoving Mpc (not Mpc/h!)"""
    mbin = np.logspace(6,12,18)
    binmid=np.log10(mbin)[:-1]+np.diff(np.log10(mbin))/2
    mhis = np.histogram(m,mbin)
    mask = mhis[0]>0
    Volumndlog = np.diff(np.log10(mbin))*(Lbox)**3
    yy = mhis[0]/Volumndlog
    err = yy[mask]/np.sqrt(mhis[0][mask])
    y1 = np.log10(yy[mask]+err)
    y2 = yy[mask]-err
    y2[y2<=0] = 1e-50
    return (binmid[mask]),np.log10(yy[mask]), y1, np.log10(y2)

def get_hmf(pig,Lbox, hh):
    """Get a conventionally unitted halo mass function in a resolved region.
    Lbox is box size in Mpc (not Mpc/h!)"""
    #Change units to M_sun
    fofmasses = pig['FOFGroups/Mass'][:]*10**10/hh
    #Find minimum halo mass
    rsl = 2*min(fofmasses[fofmasses>0])
    #Find the mass function
    smf = massfunc(fofmasses[fofmasses>rsl],Lbox)
    return smf

def check_hmf(pig):
    """Check we have the mass functions."""
    bff = bigfile.BigFile(pig)
    lbox = float(bff["Header"].attrs["BoxSize"])
    hh = bff["Header"].attrs["HubbleParam"]
    hmf = get_hmf(pig, lbox, hh)
    assert np.max(hmf) > 0

def modecount_rebin(kk, pk, modes, minmodes=2, ndesired=20):
    """Rebins a power spectrum so that there are sufficient modes in each bin"""
    assert np.all(kk) > 0
    logkk=np.log10(kk)
    mdlogk = (np.max(logkk) - np.min(logkk))/ndesired
    istart=iend=1
    count=0
    k_list=[kk[0]]
    pk_list=[pk[0]]
    targetlogk=mdlogk+logkk[istart]
    while iend < np.size(logkk)-1:
        count+=modes[iend]
        iend+=1
        if count >= minmodes and logkk[iend-1] >= targetlogk:
            pk1 = np.sum(modes[istart:iend]*pk[istart:iend])/count
            kk1 = np.sum(modes[istart:iend]*kk[istart:iend])/count
            k_list.append(kk1)
            pk_list.append(pk1)
            istart=iend
            targetlogk=mdlogk+logkk[istart]
            count=0
    k_list = np.array(k_list)
    pk_list = np.array(pk_list)
    return (k_list, pk_list)

def get_power(matpow, rebin=True):
    """Plot the power spectrum from CAMB
    (or anything else where no changes are needed)"""
    data = np.loadtxt(matpow)
    kk = data[:,0]
    ii = np.where(kk > 0.)
    #Rebin power so that there are enough modes in each bin
    kk = kk[ii]
    pk = data[:,1][ii]
    if rebin:
        modes = data[:,2][ii]
        return modecount_rebin(kk, pk, modes)
    return (kk,pk)

def check_power(scalefactor):
    """Check power spectrum is right"""
    matpow = "output/powerspectrum-%.4f.txt" % scalefactor
    kk_sim, pk_sim = get_power(matpow)
    zz = 1/scalefactor - 1
    pk_camb = np.loadtxt("class_pk_9.dat-%.1f" % zz)
    pk_camb_int = scipy.interpolate.interp1d(pk_camb[:,0], pk_camb[:,1])
    assert_allclose(pk_sim, pk_camb_int(kk_sim), rtol=0.04, atol=0.)

# Check that the power spectrum output is sensible and that the mass functions are right.
# asserting the initial power spectrum is 1% accurate
check_power(0.2)
check_power(0.25)
for pp in range(3):
    check_hmf('output/PART_'+str(pp).rjust(3,'0'))
