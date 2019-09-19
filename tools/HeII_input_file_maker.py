import numpy as np
#import matplotlib.pyplot as plt
import scipy.integrate
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import interp1d
import os.path
import rate_network as RN
import sys
import warnings



if len(sys.argv) == 1:
    print('No input parameters detected. Please provide at minimum: (1) QSO Spectral index (2) Thresshold long-mean-free-path photon energy in eV')

alpha_q = float(sys.argv[1])
Emax = float(sys.argv[2])
if len(sys.argv) == 3:
    hist = 'quasar'
else:    
    hist = sys.argv[3]
if (hist == 'linear'):
    reion_z_i = float(sys.argv[4])
    reion_z_f = float(sys.argv[5])

   
class HeIIheating(object):
    """Thermal state of gas due to HeII reionization. HeII heating comes in two varieties-- local instantaneous heating from HeIII front (the photons just above 4Ryd are approximated as being immediately absorbed) and uniform heating from harder photons that free stream through the IGM.
    Immediate TODO: change clumping factor prescription and double check approximations"""
    def __init__(self):
        if hist == 'linear':
            self.hist = linear_history()
        else:
            self.hist = HeII_history()
        self.E0_HI = 13.6 #eV
        self.E0_HeI = 24.6 #eV
        self.E0_HeII = 54.4 #eV
        self.c = 3.0e10 #cm/s
        self.kBeV = 8.6173e-5  
        self.sigma0 = 0.25* 6.35e-18 #cm^2
        self.h_eV_s = 4.135668e-15 #eV s
        self.eVtoerg = 1.6e-12
        self.h = 0.678
        self.OmegaM = 0.3175
        self.OmegaK = 0 
        self.OmegaL = 0.6825
        self.Omegab = 0.048
        self.H0kmsMpc = 100.*self.h
        self.H0 = 3.241e-20*self.H0kmsMpc
        self.kB = 1.381e-16 #Boltzmann constant
        self.eVtoerg = 1.60217e-12
        self.protonmass = 1.67262178e-24 #g
        self.G = 6.673e-8 #cm^3/s^2/g
        self.h2rhocrit = 3./(8.*np.pi*self.G)*self.H0**2.

    def EstEmax(self, redshift, XHeII):
        """Estimation of the threshold energy at which photons are no longer 'instantaneously' absorbed"""
        Emax = (((1.+ redshift)/4.)**2.*(4. * XHeII))**(1./3.)*100.
        return Emax

    def Delta_Q_inst(self, redshift, Emax = Emax, alpha_q = alpha_q): 
        """Instantaneous heat injection from HeII reionization (absorption of photons with E<Emax) [eV/cm^3]"""
        Q_inst = self.nHe(redshift)*((alpha_q/(alpha_q-1.))*((Emax**(-alpha_q+1.)-self.E0_HeII**(-alpha_q+1.))/(Emax**(-alpha_q)-self.E0_HeII**(-alpha_q)))-self.E0_HeII)
        return Q_inst

    def nH(self, redshift, YHe = 0.25): 
        """Mean cosmic H density"""
        nh = (1.-YHe)*(self.h2rhocrit)*(self.Omegab)/(self.protonmass)*(1.+ redshift)**3. 
        return nh

    def nHe(self, redshift, YHe = 0.25):
        """Mean cosmic He density"""
        nhe = YHe*(self.h2rhocrit*self.Omegab)/(4.*self.protonmass)*(1.+ redshift)**3.
        return nhe

    def ne(self, redshift , temp): 
        """Approximate!! pre-HeII reionization electron density-- this is used in computing the uniform Quasar heating"""
        rn = RN.RateNetwork(redshift)
        ne = (self.nH(redshift) + self.nHe(redshift))/((self.nH(redshift)*rn.recomb.alphaHp(temp)/rn.photo.gH0(redshift))+(self.nHe(redshift)*rn.recomb.alphaHep(temp)/rn.photo.gHe0(redshift))+1.)
        return ne
       
    def H(self,redshift): 
        """H(z), multiplied by km/Mpc to get s^-1"""
        H_z = (self.H0**2.*(self.OmegaM*(1+redshift)**3. + self.OmegaK*(1+redshift)**2. + self.OmegaL))**0.5
        return H_z


    def sigmaHI(self, E):
        """Fit for the photoionization cross section of HI (Hui & Gnedin '97)"""
        E0 = 0.4298
        sigma0 = 5.475e-14
        P = 2.963 
        ya = 32.88
        return sigma0*(E/E0 - 1.)*(E/E0-1.)*(E/E0)**(0.5*P-5.5)/(1.+np.sqrt(E/(E0*ya)))**P


    def sigmaHeII(self, E):
        """Fit for the photoionization cross section of HeII (Hui & Gnedin '97)"""
        E0 = 1.720
        sigma0 = 1.369e-14
        P = 2.963 
        ya = 32.88
        return sigma0*(E/E0 - 1.)*(E/E0-1.)*(E/E0)**(0.5*P-5.5)/(1.+np.sqrt(E/(E0*ya)))**P

 
    def tau(self,z,z0,E):
        """Approximate optical depth that long MFP photons see-- may want to replace this with something better later but it's probably good enough right now"""
        xHII = 1. #appropriate for HeII reionization redshifts
        xHI = np.amax([1. - xHII, 0.]) #follows from above (xHI is effectively 0)
        xHeI = xHI #HeI should be ionized with HI by first galaxies
        xHeII = np.amax([1 - xHeI - self.hist.XHeIII(z), 0.]) #This essentially follows the evolution of HeIII fraction.
        func = lambda z: self.c/(self.H(z)*(1+z))*self.sigmaHeII(E*(1.+z)/(1.+z0))*self.nHe(z)*xHeII
        t = scipy.integrate.quad(func,z0,z)
        return t[0]


    def a_norm(self, redshift, alpha_q = alpha_q, clumping_fac = 2., T_est = 15000.):
        """Normalization of emissivity-- requires that the total ionizing emissivity of ionizing photons balances the number of ionizations plus recombinations. To do: put in better clumping factor prescription and T_est"""
        rn = RN.RateNetwork(redshift)
        A = ((alpha_q)*self.nHe(redshift))/(self.E0_HeII**(-alpha_q))*(self.hist.dXHeIIIdz(redshift)*(-self.H(redshift)*(1+redshift))+ clumping_fac*rn.recomb.alphaHepp(T_est)*self.hist.XHeIII(redshift)*self.ne(redshift,T_est)) 
        return A


    def JE(self, z0, E, alpha_q = alpha_q, clumping_fac = 2.):
        """Specific intensity based on powerlaw QSO spectrum"""
        func = lambda z: (self.c/(4.*np.pi))*(1./(self.H(z)*(1+z)))*(1+z0)**3./((1+z)**3.)*self.a_norm(z)*((E)**(-alpha_q))*np.exp(-self.tau(z,z0,E))
        J_E = scipy.integrate.quad(func,z0,10.)
        return J_E[0]
    
      
    def dQ_hard_dz(self, redshift, Emax = Emax, E_lim = 1000.):   
        """Uniform heating rate from long MFP hard photons only (E_gamma > Emax), dQ/dz. This is making the assumption that all Helium is in the form of HeII."""  
        func = lambda E: ((E-self.E0_HeII)/E)*self.JE(redshift,E)*self.sigmaHeII(E)
        w = scipy.integrate.quad(func,Emax,E_lim)
        dQdz = 4.*np.pi*self.eVtoerg*self.nHe(redshift)*w[0]*1./(self.H(redshift)*(1+redshift)) 
        return dQdz 



    def dGamma_hard_dt(self, redshift, Emax = Emax, E_lim = 1000.):
        """Photoionization heating of hard photons only (E_gamma > Emax), dGamma/dt. Units are erg/s/cm^3."""
        func = lambda E: ((E-self.E0_HeII)/E)*self.JE(redshift,E)*self.sigmaHeII(E)
        w = scipy.integrate.quad(func,Emax,E_lim)
        dGammadt = 4.*np.pi*w[0]*self.eVtoerg*self.nHe(redshift)
        return dGammadt

    def setUpInterpTable(self, Emax = Emax, alpha_q = alpha_q, clumping_fac = 2., numz = 100.):
        print("Setting up interpolation table!")
        directory = '.'
        filename = directory + '/HeIIReionizationTable'
        
        z_quasar = np.logspace(np.log10(6.0),np.log10(2.8),numz)
        dQ_LMFP_dat = np.zeros(len(z_quasar))
        XHeIII = np.zeros(len(z_quasar))
        
        if hist != 'linear':
            reion_z_i = 6. 
            reion_z_f = 2.
            xHeII_interp = self.hist.makexHeIIInterp(reion_z_f, reion_z_i)
            for i in range(len(z_quasar)):
                dQ_LMFP_dat[i] = self.dGamma_hard_dt(z_quasar[i])
                XHeIII[i]  = xHeII_interp(z_quasar[i])
                print(i, z_quasar[i], dQ_LMFP_dat[i])
        else:
            for i in range(len(z_quasar)):
                dQ_LMFP_dat[i] = self.dGamma_hard_dt(z_quasar[i])
                XHeIII[i]  = self.hist.XHeIII(z_quasar[i])
                print(i, z_quasar[i], dQ_LMFP_dat[i])
        print('Creating table ',filename)
        
        f = open(filename, 'w')
        f.write('#File parameters for this input file: Emax = ' + str(Emax) + ', alpha_q = ' + str(alpha_q)+ ', Clumping factor = ' + str(clumping_fac) +  ', Simple linear history (1) or QSO history (0) = ' + str(hist) + '\n')
        f.write('#Units of heating rate (3rd column) are erg/s/cm^3 \n')            
        f.write('{0:f} \n'.format(alpha_q))
        f.write('{0:f} \n'.format(Emax))
        for i in range(len(z_quasar)):
            f.write('{0:e} {1:e} {2:e} \n'.format(z_quasar[i], XHeIII[i], dQ_LMFP_dat[i]))
        f.close()

        print('Done!')


class HeII_history(object):
    """Determines the HeII reionization history from a quasar emissivity function"""
    def __init__(self):
        self.h_erg_s = 6.626e-27 #erg s
        self.mpctocm = 3.086e24
        try:
            self.xHeII_table = np.genfromtxt('xHeII.dat')
        except OSError:
            self.xHeII_table = np.zeros(0)


    
    def XHeIII(self, redshift, reion_z_f = 2, reion_z_i = 6, numz = 1000.):
        """HeIII fraction over cosmic time based on a QSO emissivity function."""
        try:
            self.xHeII_table = np.genfromtxt('xHeII.dat')
            table = self.xHeII_table
            xHeII_interp = interp1d(table[:,0],table[:,1], bounds_error=False, fill_value=0.0)
        except IndexError:
            xHeII_interp = self.makexHeIIInterp(reion_z_f, reion_z_i)
        return xHeII_interp(redshift)
    
    
    def dXHeIIIdz(self, redshift, dz = 0.01):
        """Change in XHeIII, where XHeIII evolves based on a QSO emissivity function fit."""
        return (self.XHeIII(redshift + dz) - self.XHeIII(redshift))/dz
          
        

    def dXHeIIIdz_int(self, xHeIII, redshift, clumping_fac = 2., T_est = 15000.):
        """Sets up differential eq."""
        rn = RN.RateNetwork(redshift)
        HH = HeIIheating()
        dXHeIIIdz = -(self.quasar_emissivity_Kulkarni19(redshift) - clumping_fac*rn.recomb.alphaHepp(T_est)*HH.ne(redshift, T_est)*xHeIII*HH.nHe(redshift))/HH.nHe(redshift)/(HH.H(redshift)*(1+redshift))
        return dXHeIIIdz


    
    def xHeIII_quasar(self, zmin, zmax, numz = 1000):
        """Makes a table of HeII reionization history: z, XHeIII, and number of ionizing photons per nHe produced."""
        dataarr = np.zeros([3,numz])
        dataarr[0,:] = np.linspace(zmax,zmin, numz)
        x = scipy.integrate.odeint(self.dXHeIIIdz_int, np.zeros(numz), dataarr[0,:])
        dataarr[1,:] = [min(x[i,0], 1) for i in range(numz)]
        dataarr[2,:] = scipy.integrate.odeint(self.dXHeIIIdz_int, np.zeros(numz), dataarr[0,:])[:,0]
        return dataarr


    def makexHeIIInterp(self, reion_z_f, reion_z_i):
        """Produces outfile where columns are z, xHeIII, and number of ionizing photons per nHe produced. Returns an interpolation function."""
        dataarr = self.xHeIII_quasar(reion_z_f, reion_z_i)
        filename = 'xHeII.dat'
        np.savetxt(filename, np.column_stack([dataarr[0,:], dataarr[1,:], dataarr[2,:]]), fmt='%.4e')
        print('Saved xHeII history to ', filename)
 
        return scipy.interpolate.interp1d(dataarr[0,:], dataarr[1,:], bounds_error=False, fill_value=0.0)


    def quasar_emissivity_HM12(self, redshift, alpha_q = alpha_q):
        """Proper emissivity of HeII ionizing photons from Haardt & Madau (2012) (1105.2039.pdf eqn 37)"""
        enhance_fac=1
        epsilon_nu = enhance_fac*3.98e24*(1+redshift)**7.68*np.exp(-0.28*redshift)/(np.exp(1.77*redshift) + 26.3) #erg s^-1 MPc^-3 Hz^-1 
        e = epsilon_nu/(self.h_erg_s*alpha_q)/(self.mpctocm**3)*4.**(-alpha_q)
        return e
        

    def quasar_emissivity_K15(self, redshift, alpha_q = alpha_q):
        """Proper emissivity of HeII ionizing photons from Khaire + (2015)"""
        epsilon_nu = 10.**(24.6)*(1.+redshift)**8.9 * np.exp(-0.36*redshift)/(np.exp(2.2*redshift)+25.1)  #erg s^-1 MPc^-3 Hz^-1
        e = epsilon_nu/(self.h_erg_s*alpha_q)/(self.mpctocm**3)*4.**(-alpha_q)
        return e
        
    def quasar_emissivity_Kulkarni19(self, redshift, alpha_q = alpha_q):
        """Proper emissivity of HeII ionizing photons from Kulkarni + (2019)"""
        epsilon_nu = 10.**(24.72)*(1.+redshift)**8.42 * np.exp(-2.1*redshift)/(np.exp(1.09*redshift)+38.56)  #erg s^-1 MPc^-3 Hz^-1
        e = epsilon_nu/(self.h_erg_s*alpha_q)/(self.mpctocm**3)*4.**(-alpha_q)
        return e

        
        	    

class linear_history(object):
    """Makes a HeII reionization history where X_HeIII is a linear function of redshift"""
    
    def XHeIII(self, redshift):
        """XHeIII history that is linear with redshift. Set initial and final HeII reion redshifts (highly recommend reion_z_f=2.8)."""
        if (redshift > reion_z_i):
            func = 0.   
        elif (redshift <= reion_z_i) and (redshift >= reion_z_f):
            func = (1./(reion_z_f-reion_z_i))*redshift + (1. - (1./(reion_z_f-reion_z_i))*reion_z_f)
        else:    
            func = 1.
        return func
            
    def dXHeIIIdz(self, redshift):
        """Change in XHeIII, where XHeIII evolves linearly with redshift. Set initial and final HeII reion redshifts (highly recommend reion_z_f=2.8)."""    
        if (redshift > reion_z_i):
            func = 0.
        elif (redshift <= reion_z_i) and (redshift >= reion_z_f): 
            func = (1./(reion_z_f-reion_z_i))
        else:    
            func = 0.
        return func


HeIIheating().setUpInterpTable()
