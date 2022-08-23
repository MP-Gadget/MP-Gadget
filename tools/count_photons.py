'''
This code compares the global ratio of stellar mass to gas mass
with the global neutral fraction over time, as a test of the
global statistics of reionisation.
'''

import argparse
import bigfile as bf
import numpy as np
from os.path import exists
from astropy import cosmology, constants as C, units as U
from scipy import integrate
from mpi4py import MPI
from nbodykit.lab import BigFileCatalog
import dask.array as da

import logging
logger = logging
logging.basicConfig(level=logging.INFO)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

ap = argparse.ArgumentParser("get_xgrids.py")
ap.add_argument("bigfile", help='comma separated list of paths to the MP-Gadget output directories')
ap.add_argument("--output", help='path to save the plots')
ap.add_argument("--dataname", help='path to save the data')
ap.add_argument("--reiongrid", help='comma separated paths to zreion grids')
ap.add_argument("--gridname", help='comma separated names of datasets in bigfiles')
ap.add_argument("--nion", type=int, default=4000, help='photons per stellar baryon')
ap.add_argument("--fesc-n", type=float, help='ionising photon escape fraction norm')
ap.add_argument("--fesc-s", type=float, help='ionising photon escape fraction halo mass scaling')
ap.add_argument("--zlist", help='comma separated list of redshifts to plot')
ap.add_argument("--show-plot", help="show plot with matplotlib", action="store_true")

ns = ap.parse_args()

import matplotlib
if not ns.show_plot:
    matplotlib.use('pdf')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 10
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["text.usetex"] = False

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

h = 0.7186
Om0 = 0.2814
Ob0 = 0.0464
m_nu = [0., 0., 0.]*U.Unit('eV')
Tcmb0 = 2.7255
Ode0 = 0.7186

cosmo = cosmology.FlatLambdaCDM(H0=h*100,Om0=Om0,Ob0=Ob0,Tcmb0=Tcmb0,m_nu=m_nu)

def read_globalreion_info(fname,reiongrid=False):
    #set up the list of snapshots, and redshifts
    time_list = np.loadtxt(f'{fname}/Snapshots.txt', dtype=float, ndmin=2)
    snapshot_list = time_list[:,0].astype(int)

    #find which snapshots are in your list
    if ns.zlist is not None:
        zlist = np.fromstring(ns.zlist,dtype=float,sep=',')
        mask = np.any(np.fabs(1/time_list[:,1][:,None] - 1 - zlist[None,:]) < 0.01,axis=1)

    else:
        mask = np.ones(snapshot_list.shape,dtype=bool)

    snapshot_list = snapshot_list[mask]
    time_list = time_list[mask,1]
    redshift_list = 1/time_list - 1

    #define mass / neutral fraction arrays
    gas_mass = np.zeros(len(snapshot_list))
    star_mass = np.zeros(len(snapshot_list))
    gas_xhi = np.zeros(len(snapshot_list))
    vol_xhi = np.zeros(len(snapshot_list))
    phot_fof = np.zeros(len(snapshot_list))
    J21_avg = np.zeros(len(snapshot_list))
    J21_ion = np.zeros(len(snapshot_list))
    T0_avg = np.zeros(len(snapshot_list))

    #snapshot mask to account for missing snapshots
    snap_mask = np.ones(len(snapshot_list),dtype=bool)

    comm = MPI.COMM_WORLD
    nrank = comm.Get_size()

    if comm.rank == 0:
        logger.info("plotting snapshots %s, z = %s",snapshot_list,redshift_list)

    mean_bary_dens = (cosmo.critical_density(0) * cosmo.h**(-2) * cosmo.Ob0).to('g cm-3').value

    for i, snap in enumerate(snapshot_list):
        #read in the particle file
        filename = f'{fname}/PART_{snap:03d}/'
        fofname = f'{fname}/PIG_{snap:03d}/'

        if not exists(filename) or not exists(fofname):
            snap_mask[i] = False
            continue

        if comm.rank == 0:
            logger.info("snapshot %s", filename)

        #read in gas mass and neutral fraction, star mass, fof mass for escape fractions
        #NOTE: all this mass-weighting doesn't do much since particles are almost all the same mass
        cat = BigFileCatalog(filename,dataset='0/',header='Header')
        gas_mass[i] = cat.compute(cat['Mass'].sum())

        #if comm.rank == 0:
        #    logger.info("Test: cat compute %.3e, compute sum %.3e, sum compute %.3e", gas_mass[i], cat['Mass'].compute().sum(), cat['Mass'].sum().compute())

        gas_xhi[i] = cat.compute((cat['Mass']*cat['NeutralHydrogenFraction']).sum())
        
        #To get volume weighted, I need to make a grid, currently using 1Mpc resolution
        if reiongrid:
            Nmesh = int(cat.attrs['BoxSize']/1000)
            mesh = cat.to_mesh(Nmesh=Nmesh,weight='Mass',value='NeutralHydrogenFraction',position='Position')    
            field = mesh.to_real_field(normalize=False)
            mesh_mass = cat.to_mesh(Nmesh=Nmesh,weight='Mass',position='Position')
            field_mass = mesh_mass.to_real_field(normalize=False)
            field[...] /= field_mass[...]

            vol_xhi[i] = field.cmean()
        else:
            vol_xhi[i] = 0 #calculated later from file

        
        J21_avg[i] = cat.compute((cat['Mass']*cat['J21']).sum())
        #particle weighted because of selection
        sel_ion = (cat['NeutralHydrogenFraction'] < 0.1)
        J21_ion[i] = cat.compute((cat['J21'])[sel_ion].mean())

        dens = cat['Density'] * cat.attrs['UnitMass_in_g'] / cat.attrs['UnitLength_in_cm']**3 #* U.Unit('M_sun kpc-3')
        delta = (dens / mean_bary_dens)

        #sel_meandens = ((da.log10(delta) < 0.1) & (da.log10(delta) > -0.1)).compute()
        sel_meandens = ((delta < 1.1) & (delta > 1/1.1))
        #particle weighted in the selected bin
        temps = u_to_t(cat['InternalEnergy'][sel_meandens],cat['NeutralHydrogenFraction'][sel_meandens])
        buf = temps.mean()
        T0_avg[i] = cat.compute(buf)

        #multiply stellar mass by photons per stellar baryon
        #and take ratio with gas mass, giving an estimate of
        #total number of photons released per hydrogen atom
        #this is a sanity check of photon conservation
        if ns.fesc_n is not None:            
            #we only want fof info if we are comparing against a given escape fraction
            if ns.fesc_s is not None and ns.fesc_n is not None:
                cat = BigFileCatalog(fofname,dataset='FOFGroups/',exclude=['GasMetalMass','StellarMetalMass'],header='Header')
                if cat.size == 0:
                    phot_fof[i] = 0
                else:
                    fof_fesc = ns.fesc_n * (cat['Mass']/h/1.)**ns.fesc_s #already in 1e10 solar
                    fof_fesc = da.minimum(fof_fesc,1.)
                    fof_star = da.transpose(cat['MassByType'])[4]
                    phot_fof[i] = cat.compute((fof_fesc*fof_star).sum()) * ns.nion / (1 - 0.75*Y_He) #escape fraction weighted GSM
            #constant escape only needs stars
            else:
                cat = BigFileCatalog(filename,dataset='4/',header='Header')
                star_mass[i] = cat.compute(cat['Mass'].sum())
                phot_fof[i] = star_mass[i] * ns.nion * ns.fesc_n / (1 - 0.75*Y_He)

    gas_mass = comm.allreduce(gas_mass,op=MPI.SUM)[snap_mask]
    gas_xhi = comm.allreduce(gas_xhi,op=MPI.SUM)[snap_mask]
    phot_fof = comm.allreduce(phot_fof,op=MPI.SUM)[snap_mask]

    J21_avg = comm.allreduce(J21_avg,op=MPI.SUM)[snap_mask]
    J21_ion = comm.allreduce(J21_ion,op=MPI.SUM)[snap_mask] / nrank #assumes ranks have equal particle number
    T0_avg = comm.allreduce(T0_avg,op=MPI.SUM)[snap_mask] / nrank

    redshift_list = redshift_list[snap_mask]

    G12_avg = J21_avg * 2.535452 #NOTE: ALPHA == 2 RATE from J21 to G12, TODO: generalise by interpolating a coefficient file
    G12_ion = J21_ion * 2.535452

    result_dict = {
        "mass_xhi" : gas_xhi / gas_mass, #assumes particles have equal mass
        "vol_xhi" : vol_xhi[snap_mask], #snap mask because it was reduced earlier
        "phot_fof" : phot_fof / gas_mass,
        "G12_avg" : G12_avg / gas_mass,
        "G12_ion" : G12_ion,
        "T0_avg" : T0_avg,
    }

    return result_dict,redshift_list

def compute_tau(xHI,redshifts):
    thomson_cross_section = C.sigma_T
    density_H = cosmo.Ob0 * X_H * cosmo.critical_density(0) / C.m_p
    density_He = cosmo.Ob0 * (1-X_H)/4 * cosmo.critical_density(0) / C.m_p

    cosmo_factor = lambda z: C.c * (1+z)**2 / cosmo.H(z) * thomson_cross_section
    
    #tau calculation from the meraxes package
    def d_te_postsim(z):
        """This is d/dz scattering depth for redshifts greater than the final
        redshift of the run.
        N.B. THIS ASSUMES THAT THE NEUTRAL FRACTION IS ZERO BY THE END OF THE
        INPUT RUN!
        """
        if z <= 4:
            return (cosmo_factor(z) * (density_H + 2.0*density_He)).decompose()
        else:
            return (cosmo_factor(z) * (density_H + density_He)).decompose()
    
    def d_te_sim(z, xHII):
        """This is d/dz scattering depth for redshifts covered by the run.
        """
        prefac = cosmo_factor(z)
        return (prefac * (density_H*xHII + density_He*xHII)).decompose()
    
    xHII = 1 - np.copy(xHI)[::-1]
    z_integ = redshifts[::-1]
    scattering_depth = np.zeros_like(z_integ)
    
    for i,z in enumerate(z_integ):
        post_sim_contrib = integrate.quad(d_te_postsim, 0, np.minimum(z,z_integ[0]))[0]
        sim_contrib = integrate.trapz(d_te_sim(z_integ,xHII)[:i+1],z_integ[:i+1],axis=0)

        scattering_depth[i] = sim_contrib + post_sim_contrib

    return scattering_depth[::-1]
    
def plot_globalreion(result_dicts,redshifts):
    #plot the neutral fraction and photon ratio vs snapshot
    fig = plt.figure(figsize=(4,8))
    axxH = fig.add_subplot(411)

    colors = ['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10']
    cnt = 0
    for res,z in zip(result_dicts,redshifts):
        logger.info(f"MASS PHOT RATIO: {(1-res['phot_fof'])/res['mass_xhi']}")
        logger.info(f"VOL PHOT RATIO: {(1-res['phot_fof'])/res['vol_xhi']}")
        if ns.fesc_n is not None:
            axxH.plot(z, 1 - res['phot_fof'], label=f'photons / H (fn = {ns.fesc_n}, fs = {ns.fesc_s})')
        axxH.plot(z,res['mass_xhi'],color=colors[cnt],linewidth=1,linestyle=':')
        axxH.plot(z,res['vol_xhi'],color=colors[cnt],label='Simulation')
        cnt += 1
    
    #Yuxiang's compiled XHI observations:
    #Dark Pixels
    axxH.errorbar([5.6,6.07], [0.04,0.38], yerr=[[0,0],[0.05,0.20]], fmt='o',color='k', label='McGreer+15',mfc='white',capsize=3, markeredgewidth=1, elinewidth=1,alpha=1)
    axxH.errorbar([5.6,6.07], [0.04,0.38], yerr=[[0.03,0.1],[0,0]], mfc='white',uplims=True, fmt=' ',color='k', markeredgewidth=1,alpha=1)
    axxH.errorbar([5.9], [0.06], yerr=[[0],[0.05]],mfc='white', fmt='o',color='k',capsize=3, elinewidth=1, markeredgewidth=1,alpha=1)
    axxH.errorbar([5.9], [0.06], yerr=[[0.03],[0]], uplims=True, fmt=' ',color='k',alpha=1)
    axxH.errorbar([5.61,5.8,5.99,6.21,6.35], [0.42, 0.53,0.67,0.53,0.69], yerr=[[0,0,0,0,0],[0.05,0.07,0.07,0.11,0.15]], markersize=3,fmt='o',color='k', label='Campo+in prep.',capsize=3, elinewidth=1,alpha=1)
    axxH.errorbar([5.61,5.8,5.99,6.21,6.35], [0.42, 0.53,0.67,0.53,0.69], yerr=[[0.1,0.1,0.1,0.1,0.1],[0,0,0,0,0]], markersize=3,uplims=True, fmt=' ',color='k',alpha=1)

    # QSO damping
    axxH.errorbar([7.0], [0.70], yerr=[[0.23],[0.20]], fmt='p',color='b', label='Wang+20',markersize=3,capsize=3, elinewidth=1,mfc='white', markeredgewidth=1,alpha=1)
    axxH.errorbar([7.5413], [0.56], yerr=[[0.18],[0.21]], fmt='h',color='b', label='BaÃ±ados+18',markersize=3,capsize=3, elinewidth=1,mfc='white', markeredgewidth=1,alpha=1)
    #axxH.errorbar([7.0851,7.5413], [0.40,0.21], yerr=[[0.19,0.19],[0.21,0.17]], fmt='*',color='b', label='Greig+17/19',markersize=3,capsize=3, elinewidth=1, mfc='white',markeredgewidth=1,alpha=1)
    axxH.errorbar([7.0851,7.5413], [0.48,0.60], yerr=[[0.26,0.23],[0.26,0.20]], fmt='s',color='b', label='Davies+18',markersize=3,capsize=3, elinewidth=1,mfc='white', markeredgewidth=1,alpha=1)
    axxH.errorbar([7.29], [0.49], yerr=[0.11], xerr=[0.20], fmt='8',color='b', label='Greig+22.',markersize=3,capsize=3, elinewidth=1, markeredgewidth=1,alpha=1)

    # LAE fraction
    ## LF
    axxH.errorbar([6.9], [0.33], yerr=[[0.1],[0.]], uplims=True, fmt='<',color='purple', label='Wold+21',markersize=3,capsize=3, elinewidth=1,  mfc='white',markeredgewidth=1,alpha=1)
    axxH.errorbar([7.3], [0.5], yerr=[[0.3],[0.1]], fmt='>',color='purple', label='Inoue+18',markersize=3,capsize=3, elinewidth=1, mfc='white',markeredgewidth=1,alpha=1)
    #axxH.errorbar([5.7,6.6,7.0], [0.4,0.4,0.4], yerr=[[0.1,0.1,0.1],[0,0,0]], uplims=True, fmt='s',color='black',alpha=0.6)
    axxH.errorbar([6.6,7.0,7.3], [0.08,0.28,0.83], yerr=[[0.05,0.05,0.07],[0.08,0.05,0.06]], fmt='^',color='purple', label='Morales+21',markersize=3,capsize=3, elinewidth=1,  mfc='white',markeredgewidth=1,alpha=1)
    ## clustering
    axxH.errorbar([6.6], [0.15], yerr=[0.15], fmt='v',color='purple', label='Ouchi+18',markersize=3, capsize=3,mfc='white', elinewidth=1, markeredgewidth=1,alpha=1)

    ## EW
    axxH.errorbar([7.0], [0.55], yerr=[[0.13],[0.11]], fmt='v',color='g', label='Whitler+20',markersize=3,capsize=3, elinewidth=1, mfc='white',markeredgewidth=1,alpha=1) #EW
    axxH.errorbar([7.9], [0.76], xerr=[0.6], yerr=[[0.],[0.1]], lolims=True,fmt='<',label='Mason+19',color='g', markersize=3, capsize=3, elinewidth=1, mfc='white',markeredgewidth=1,alpha=1) #EW
    axxH.errorbar([7.6], [0.88], yerr=[[0.1],[0.05]], xerr=[0.6],fmt='>',color='g', label='Hoag+19',markersize=3, capsize=3,mfc='white', elinewidth=1, markeredgewidth=1,alpha=1) #EW
    axxH.errorbar([7.6], [0.36], yerr=[[0.14],[0.10]], fmt='^',color='g', label='Jung+21',markersize=3, capsize=3, mfc='white', elinewidth=1, markeredgewidth=1,alpha=1)

    #axxH.errorbar([7.0], [0.4], yerr=[[0],[0.2]], fmt='^',color='g', label='Mesinger+15',markersize=10, capsize=5,mfc='white', elinewidth=2, markeredgewidth=2,alpha=1)

    axxH.set_ylabel(r'$\overline{x}_{\rm HI}$')
    #axxH.tick_params(bottom=False,labelbottom=False)
    #axxH.set_xlabel('redshift',fontsize=15)
    
    legend_elements = [Line2D([0], [0], marker='o', color='k', label='Dark Pixel'),
                        Line2D([0], [0], marker='o', color='b', label='QSO Damping'),
                        Line2D([0], [0], marker='o', color='purple', label='LAE Fraction'),
                        Line2D([0], [0], marker='o', color='g', label='LA EW')]
    #axxH.legend(loc='upper right',ncol=1,frameon=False).set_zorder(300)
    axxH.legend(handles=legend_elements)

    axxH.set_ylim(1e-5, 1)
    axxH.set_xlim(5,9)
    axxH.grid()
    
    P18_mean = 0.0544
    P18_bounds = P18_mean + np.array([-0.0081,0.0070])
    
    ax = fig.add_subplot(412)
    cnt=0
    for res,z in zip(result_dicts,redshifts):
        ax.plot(z,res['tau'],color=colors[cnt])
        cnt += 1
    
    ax.axhline(P18_mean,linestyle=':',color='k')
    ax.fill_between(np.array([5,12]),P18_bounds[0],P18_bounds[1],facecolor='black',alpha=0.2,label='Planck18')
    ax.set_xlim(5,12)
    ax.legend()
    #ax.set_xlabel('Redshift')
    ax.set_ylabel(r'$\tau_{e}$')
    #ax.tick_params(bottom=False,labelbottom=False)
    ax.grid()

    G12_D18_z = np.array([4.8,5.0,5.2,5.4,5.6,5.8])
    G12_D18_val = np.array([0.58,0.53,0.48,0.47,0.45,0.29,])
    G12_D18_eru = np.array([0.08,0.09,0.10,0.12,0.14,0.11])
    G12_D18_erl = np.array([0.20,0.19,0.18,0.18,0.17,0.11])

    G12_B21_z = np.array([5.1,6.0])
    G12_B21_val = np.array([0.7,0.3])
    G12_B21_eru = G12_B21_val*10**(0.15) - G12_B21_val #0.15 dex
    G12_B21_erl = G12_B21_val - G12_B21_val*10**(-0.15)

    G12_W10_z = np.array([5.0,6.0])
    G12_W10_val = np.array([0.47,0.18])
    G12_W10_eru = np.array([0.3,0.18])
    G12_W10_erl = np.array([0.2,0.09])
    
    G12_C11_z = np.array([5.04,6.09])
    G12_C11_val = 10**np.array([-12.15,-12.84]) / 1e-12
    G12_C11_eru = 10**np.array([-12.15+0.16,-12.84+0.18]) / 1e-12 - G12_C11_val
    G12_C11_erl = G12_C11_val - 10**np.array([-12.15-0.16,-12.84-0.18]) / 1e-12

    ax = fig.add_subplot(413)
    
    cnt = 0
    for res,z in zip(result_dicts,redshifts):
        ax.semilogy(z,res['G12_avg'],color=colors[cnt],linestyle='-')
        ax.semilogy(z,res['G12_ion'],color=colors[cnt],linestyle=':')
        cnt += 1

    ax.errorbar(G12_D18_z,G12_D18_val,yerr=[G12_D18_erl,G12_D18_eru],fmt='ro',markersize=3,elinewidth=1,capsize=3,label='D\'Aloisio+ 18')
    ax.errorbar(G12_B21_z,G12_B21_val,yerr=[G12_B21_erl,G12_B21_eru],fmt='bo',markersize=3,elinewidth=1,capsize=3,label='Becker+ 21')
    ax.errorbar(G12_W10_z,G12_W10_val,yerr=[G12_W10_erl,G12_W10_eru],fmt='go',markersize=3,elinewidth=1,capsize=3,label='Wyithe+ 10')
    ax.errorbar(G12_C11_z,G12_C11_val,yerr=[G12_C11_erl,G12_C11_eru],fmt='mo',markersize=3,elinewidth=1,capsize=3,label='Calverley+ 11')
    ax.set_xlim(5,7)
    ax.set_ylim(1e-2,1e1)
    ax.legend()
    ax.grid()
    #ax.set_xlabel('Redshift')
    ax.set_ylabel(r'$\Gamma_{12} (10^{-12} s^{-1})$')
    #ax.tick_params(bottom=False,labelbottom=False)

    T0_G20_z = np.array([5.4,5.6,5.8])
    T0_G20_val = np.array([11000,10500,12000])
    T0_G20_erl = np.array([1600,2100,2200])
    T0_G20_eru = np.array([1600,2100,2200])
    
    T0_B19_z = np.array([4.2,4.6,5.0])
    T0_B19_val = np.array([8130,7310,7370])
    T0_B19_erl = np.array([970,880,1390])
    T0_B19_eru = np.array([1340,1350,1670])

    ax = fig.add_subplot(414)
    cnt = 0
    for res,z in zip(result_dicts,redshifts):
        ax.plot(z,res['T0_avg'],color=colors[cnt])
        cnt += 1

    ax.errorbar(T0_G20_z,T0_G20_val,xerr=0.1,yerr=[T0_G20_erl,T0_G20_eru],fmt='ro',markersize=3,elinewidth=1,capsize=3,label='Gaikwad+ 20')
    ax.errorbar(T0_B19_z,T0_B19_val,xerr=0.2,yerr=[T0_B19_erl,T0_B19_eru],fmt='bo',markersize=3,elinewidth=1,capsize=3,label='Boera+ 19')
    ax.set_xlim(4,7)
    ax.grid()
    ax.legend()
    ax.set_xlabel('Redshift')
    ax.set_ylabel(r'$T_{0} (K)$')

    #fig.suptitle('',fontsize=14)
    #fig.tight_layout()
    fig.subplots_adjust(top=0.99,left=0.12,right=0.97,bottom=0.05,hspace=0.15)

    if ns.dataname is not None:
        np.savez(ns.dataname,gas_xhi=gas_mass,expected_xhi=phot_fof,G12_avg=G12_avg,thomson=scattering_depth,T0_avg=T0_avg,redshift=redshift_list)
    
    if ns.output is not None:
        fig.savefig(ns.output)
    
    if ns.show_plot:
        plt.show()

if __name__ == "__main__":
    #READ HERE
    comm = MPI.COMM_WORLD
    bigfiles = ns.bigfile.split(',')
    
    if ns.reiongrid is not None:
        gridfiles = ns.reiongrid.split(',')
        gridnames = ns.gridname.split(',')
        if len(gridfiles) != len(bigfiles):
            print('incompatible reion grids and bigfiles')
            quit()
        calc_reion_grid = False
    else:
        calc_reion_grid = True

    dicts, redshifts = zip(*[read_globalreion_info(bf,calc_reion_grid) for bf in bigfiles])

    #COMUPUTE TAU, vol xhi AND ADD TO DICT
    if comm.rank == 0:

        for i,d in enumerate(dicts):
            d['tau'] = compute_tau(d['mass_xhi'],redshifts[i])
              
            if not calc_reion_grid:
                #load zreion grid (made with get_xgrids.py)
                reion_file = bf.File(gridfiles[i])
                dset = reion_file[gridnames[i]]
                reion_grid = dset.read(0,dset.size)
                reion_file.close()

                for j,z in enumerate(redshifts[i]):
                    d['vol_xhi'][j] = (reion_grid < z).sum() / reion_grid.size

        plot_globalreion(dicts,redshifts)
