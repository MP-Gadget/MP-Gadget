"""A short function to plot a GSMF from a simulation, and compare to some public data"""
import glob
import os
from bigfile import BigFile
import numpy as np
from astrodatapy.number_density import number_density
import matplotlib.pyplot as plt

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

def get_gsmf(pig,Lbox, hh):
    """Get a conventionally unitted galaxy stellar mass function in a resolved region.
    Lbox is box size in Mpc (not Mpc/h!)"""
    #Change units to M_sun
    fofmasses = np.transpose(pig['FOFGroups/MassByType'][:])[4]*10**10/hh
    #Find minimum halo mass
    rsl = 2*min(fofmasses[fofmasses>0])
    #Find the mass function
    smf = massfunc(fofmasses[fofmasses>rsl],Lbox)
    return smf

def getbmf(pig,Lbox, hh):
    """Get the black hole mass function"""
    #Swallowed black holes are never added to the FOF table
    bhmasses = (pig['5/BlackholeMass'][:])*10**10/hh
    swallowed = np.array(pig['5/Swallowed'][:])
    bmf = massfunc(bhmasses[np.where(swallowed < 1)],Lbox)
    return bmf

def plot_bhmf(pig, label=None):
    """Plot a black hole mass function from a FOF table."""
    bf = BigFile(pig)
    redshift = 1/bf['Header'].attrs['Time']-1
    hh = bf['Header'].attrs['HubbleParam']
    lbox = bf['Header'].attrs['BoxSize']/1000/hh
    lfm = getbmf(bf,lbox, hh)
    plt.plot(lfm[0],lfm[1],label=(label or '')+' z=%.1f'%redshift)
    plt.fill_between(lfm[0],lfm[2],lfm[3],alpha=0.2)
    plt.xlabel(r'$\mathrm{log}_{10} [M_{\rm BH}/M_{\odot}]$',fontsize=17)
    plt.ylabel(r'$\mathrm{log}_{10} \phi/[\mathrm{dex}^{-1} \mathrm{Mpc}^{-3}]$',fontsize=15)
    plt.xlim(6,12)
    plt.ylim(-7,-2.5)
    plt.title('BH Mass function',fontsize=15)
    plt.legend(fontsize=15)

def plot_gsmf(pig, label=None, plot_data=True):
    """Plot a galaxy stellar mass function from a FOF table, compared to some observations."""
    bf = BigFile(pig)
    redshift = 1/bf['Header'].attrs['Time']-1
    #Note! Assumes kpc units!
    hh = bf['Header'].attrs['HubbleParam']
    lbox = bf['Header'].attrs['BoxSize']/1000/hh
    print ('z=',redshift)

    lfm = get_gsmf(bf,lbox, hh)
    plt.plot(lfm[0],lfm[1], label=(label or '')+' z=%.1f' % redshift)
    plt.fill_between(lfm[0],lfm[2],lfm[3],alpha=0.2)

    color2 = {'Song2016':'#0099e6','Grazian2015':'#7f8c83','Gonzalez2011':'#ffa64d',\
          'Duncan2014':'#F08080','Stefanon2017':'#30ba52'}

    marker2 = {'Song2016':'o','Grazian2015':'s','Gonzalez2011':'v',\
          'Duncan2014':'^','Stefanon2017':'<'}

    if plot_data:
        obs = number_density(feature="GSMF",z_target=redshift,quiet=1,h=hh)
        for ii in range(obs.n_target_observation):
            data       = obs.target_observation['Data'][ii]
            label      = obs.target_observation.index[ii]
            datatype   = obs.target_observation['DataType'][ii]

            if datatype == 'data':
                data[:,1:] = np.log10(data[:,1:])
                try:
                    color      = color2[label]
                    marker     = marker2[label]
                except KeyError:
                    color = None
                    marker = 'o'
                plt.errorbar(data[:,0],  data[:,1], yerr = [data[:,1]-data[:,3],data[:,2]- data[:,1]],\
                            label=label,color=color,fmt=marker)
            else:
                continue

    plt.legend(fontsize=14)
    plt.title('GSMF,bhfdbk,z=%.1f'%redshift,fontsize=15)
    plt.ylabel(r'$\mathrm{log}_{10} \phi/[\mathrm{dex}^{-1} \mathrm{Mpc}^{-3}]$',fontsize=15)

def find_redshift(redshift, directory, pig=True):
    """Find a snapshot at a given redshift from a directory list. Returns snapshot number."""
    if pig:
        fname = "PIG_*"
    else:
        fname = "PART_*"
    globs = glob.glob(os.path.join(directory, fname))
    for gg in globs:
        bf = BigFile(gg)
        rr = 1/bf['Header'].attrs['Time']-1
        if np.abs(rr - redshift) < 0.05:
            return gg
        del bf
    return None
