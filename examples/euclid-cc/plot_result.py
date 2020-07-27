"""Script to check power spectrum against pkdgrav"""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def modecount_rebin(kk, pk, modes, minmodes=20, ndesired=200):
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

mpg = get_power("measurePk_6144-mpgadget_N2048_L500.z0.0.txt")

pkggrav3 = np.loadtxt("measurePk_4096-pkdgrav3_N2048_L500.z0.0.txt")
mpgintp = interp1d(mpg[0], mpg[1])
ramses = np.loadtxt("measurePk_4096-ramses_N2048_L500.z0.0.txt")
gadget3 = np.loadtxt("measurePk_4096-gadget3_N2048_L500.z0.0.txt")
plt.semilogx(pkggrav3[:,0], mpgintp(pkggrav3[:,0])/pkggrav3[:,1], label="MP-Gadget (6144)")
plt.semilogx(pkggrav3[:,0], gadget3[:,1]/pkggrav3[:,1], label="Gadget3")
plt.semilogx(pkggrav3[:,0], ramses[:,1]/pkggrav3[:,1], label="RAMSES")
plt.ylim(0.98,1.02)
plt.legend()
plt.xlabel("k (h/Mpc")
plt.ylabel(r"$\Delta P(k)$")
