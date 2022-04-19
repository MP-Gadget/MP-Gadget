"""Compare two output directories with Lyman alpha forest statistics. Make plots for :
- P(k)
- GSMF
- HMF
- BHMF
- Flux power spectrum with and without mean flux rescaling."""
import sys
import os.path
import numpy as np
import matplotlib.pyplot as plt
import bigfile
#from fake_spectra import spectra
import plotGSMF

def make_figures():
    """Make the subplots for the figures"""
    fig = plt.figure()
    ax_abs = fig.add_subplot(2, 1,1)
    ax2_rel = fig.add_subplot(2, 1,2)
    return ax_abs, ax2_rel

def plot_power(output1, output2, scalefactor):
    """Make the plots"""
    matpow1 = os.path.join(output1, "powerspectrum-%.4f.txt" % scalefactor)
    pk1 = get_power(matpow1)
    matpow2 = os.path.join(output2, "powerspectrum-%.4f.txt" % scalefactor)
    pk2 = get_power(matpow2)
    ax_abs, ax2_rel = make_figures()
    ax_abs.loglog(pk1[0], pk1[1], label=output1)
    ax_abs.loglog(pk2[0], pk2[1], label=output2)
    ax2_rel.semilogx(pk2[0], pk2[1]/pk1[1])
    ax_abs.legend()
    plt.savefig("powerspectrum-%.4f.pdf" % scalefactor)
    plt.clf()

def plot_mass_functions(output1, output2, atime):
    """Plot the mass functions."""
    pig1 = plotGSMF.find_redshift(1/atime-1, output1)
    pig2 = plotGSMF.find_redshift(1/atime-1, output2)
    bff = bigfile.BigFile(pig1)
    scalefactor = bff["Header"].attrs["Time"]
    plotGSMF.plot_gsmf(pig1, label=output1, plot_data=False)
    plotGSMF.plot_gsmf(pig2, label=output2, plot_data=True)
    plt.savefig("gsmf-%.4f.pdf" % scalefactor)
    plt.clf()
    plotGSMF.plot_bhmf(pig1, label=output1)
    plotGSMF.plot_bhmf(pig2, label=output2)
    plt.savefig("bhmf-%.4f.pdf" % scalefactor)
    plt.clf()
    lbox = float(bff["Header"].attrs["BoxSize"])
    hh = bff["Header"].attrs["HubbleParam"]
    hmf1 = plotGSMF.get_hmf(bff, lbox, hh)
    bff2 = bigfile.BigFile(pig2)
    hmf2 = plotGSMF.get_hmf(bff2, lbox, hh)
    ax_abs, ax2_rel = make_figures()
    ax_abs.loglog(hmf1[0], hmf1[1], label=output1)
    ax_abs.loglog(hmf2[0], hmf2[1], label=output2)
    ax2_rel.semilogx(hmf2[0], hmf2[1]/hmf1[1])
    ax_abs.legend()
    plt.savefig("hmf-%.4f.pdf" % scalefactor)
    plt.clf()

def plot_flux_power(output1, output2, snapnum):
    """Plot the flux power spectrum change."""
    spec1 = spectra.Spectra(snapnum, output1, None, None, res=10, savefile="lya_forest_spectra.hdf5")
    spec2 = spectra.Spectra(snapnum, output2, None, None, res=10, savefile="lya_forest_spectra.hdf5")
    #Without rescaling
    fpk1 = spec1.get_flux_power_1D(tau_thresh = 1e3)
    fpk2 = spec2.get_flux_power_1D(tau_thresh = 1e3)
    plt.semilogx(fpk1[0], fpk1[1]/fpk2[1])
    plt.savefig("fpk-%.4f.pdf" % spec1.atime)
    plt.clf()

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

if __name__ == "__main__":
    oldoutput = sys.argv[1]
    newoutput = sys.argv[2]
    atime = float(sys.argv[3])
    snap = int(sys.argv[4])
    plot_power(oldoutput, newoutput, atime)
    plot_mass_functions(oldoutput, newoutput, atime)
#    plot_flux_power(oldoutput, newoutput, snap)
