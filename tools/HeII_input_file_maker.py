"""This module creates a reionization history table with desired parameters of HeII reionization for the helium reionization module of MP-Gadget.

The parameters of HeII reionization must be input as command line arguments. These parameters are:

(1) Spectral index of quasars. The recommended range for this parameter is 1.1-2.0.

(2) The threshold energy that separates long-mean-free-path photons (photons that heat the IGM uniformly) from short-mean-free-path
photons (photons that contribute to the creation of HeIII bubbles) in electron volts. The thermal history is weakly dependent on this parameter, but recommended values are ~100-200.

(3) The clumping factor of the gas. This is dependent on the simulation, but shouldn't change too much between models. Recommended values are ~1.5-4.5-- 3 is a safe choice.

(4) Duration and timing of HeII reionization. The options are 'quasar' and 'linear'. The 'quasar' option uses a quasar emissivity function to determine the reionization history.
The quasar emissivity histories are from Khaire et al. (2015) and Haardt and Madau (2012). The default is Khaire et al. (2015).
The 'linear' option allows the user to select the starting and ending redshift of HeII reionization. The HeIII fraction will be a linear function in redshift between
these two redshifts. This parameter is optional and will default to 'linear' if not provided.

(5) If 'linear', the starting HeII reionization redshift. Only to be used when (3) is 'linear'. Default is 4.0

(6) If 'linear', the ending HeII reionization redshift. HeII reionization is observed to end at z ~ 2.8. Only to be used when (3) is 'linear'.

The parameters used will be printed in a comment at the top of the output table."""

import argparse
import numpy as np
import scipy.integrate
import scipy.interpolate

class Cosmology:
    """Little module to specify the mean densities and the Hubble rate.
    Neglects radiation and neutrinos, because at z < 2 they are usually not important!"""
    def __init__(self, hub=0.678, OmegaM = 0.3175, Omegab = 0.048):
        self.hub = hub
        self.OmegaM = OmegaM
        self.OmegaK = 0
        self.OmegaL = 1. - self.OmegaK - self.OmegaM
        self.Omegab = Omegab
        self.H0kmsMpc = 100.*self.hub
        self.H0 = 3.241e-20*self.H0kmsMpc
        self.protonmass = 1.67262178e-24 #g
        Newton_G = 6.673e-8 #cm^3/s^2/g
        self.h2rhocrit = 3./(8.*np.pi*Newton_G)*self.H0**2.

    def Hubble(self,redshift):
        """H(z), multiplied by km/Mpc to get s^-1"""
        H_z = (self.H0**2.*(self.OmegaM*(1+redshift)**3. + self.OmegaK*(1+redshift)**2. + self.OmegaL))**0.5
        return H_z

    def nH(self, redshift, YHe = 0.25):
        """Mean cosmic H number density"""
        nh = (1.-YHe)*(self.h2rhocrit)*(self.Omegab)/(self.protonmass)*(1.+ redshift)**3.
        return nh

    def nHe(self, redshift, YHe = 0.25):
        """Mean cosmic He number density"""
        nhe = YHe*(self.h2rhocrit*self.Omegab)/(4.*self.protonmass)*(1.+ redshift)**3.
        return nhe

    def ne(self, redshift):
        """Approximate pre-HeII reionization electron density -- this is used in computing the uniform Quasar heating.
        This always appears with a clumping factor correction in front of it. It is just an ansatz, and assumes 1 e- for each H and He."""
        ne = self.nH(redshift) + self.nHe(redshift)
        return ne

#Some recombination rates, used below.
def _Verner96Fit(temp, aa, bb, temp0, temp1):
    """Formula used as a fitting function in Verner & Ferland 1996 (astro-ph/9509083)."""
    sqrttt0 = np.sqrt(temp/temp0)
    sqrttt1 = np.sqrt(temp/temp1)
    return aa / ( sqrttt0 * (1 + sqrttt0)**(1-bb)*(1+sqrttt1)**(1+bb) )

def alphaHepp(temp):
    """Recombination rate for doubly ionized helium, in cm^3/s. Accurate to 2%.
    Temp in K."""
    #See line 4 of V&F96 table 1.
    return _Verner96Fit(temp, aa=1.891e-10, bb=0.7524, temp0=9.370, temp1=2.774e6)

class HeIIheating:
    """Thermal state of gas due to HeII reionization.
        HeII heating comes in two varieties:
            1) local instantaneous heating from HeIII front (the photons just above 4Ryd are approximated as being immediately absorbed)
            2) uniform heating from harder photons that free stream through the IGM.
    Immediate TODO: double check approximations."""
    def __init__(self, hist=None, hub=0.678, OmegaM = 0.3175, Omegab = 0.048, z_i = 4.0, z_f = 2.8, alpha_q = 1.7, Emax = 150, clumping_fac = 3.):
        self.cosmo = Cosmology(hub=hub, OmegaM = OmegaM, Omegab = Omegab)
        if hist == 'quasar':
            self.hist = QuasarHistory(z_i=z_i, z_f=z_f, alpha_q = alpha_q, cosmo=self.cosmo, clumping_fac = clumping_fac)
        else:       
            self.hist = LinearHistory(z_i=z_i, z_f=z_f)
        self.alpha_q = alpha_q
        self.Emax = Emax
        self.clumping_fac = clumping_fac
        self.E0_HI = 13.6 #eV
        self.E0_HeI = 24.6 #eV
        self.E0_HeII = 54.4 #eV
        self.speed_of_light = 3.0e10 #cm/s
        self.sigma0 = 0.25* 6.35e-18 #cm^2
        self.alphaHeppTest = alphaHepp(15000)
        self.eVtoerg = 1.60217e-12

    def EstEmax(self, redshift, XHeII):
        """Estimation of the threshold energy at which photons are no longer 'instantaneously' absorbed"""
        Emax = (((1.+ redshift)/4.)**2.*(4. * XHeII))**(1./3.)*100.
        return Emax

    def Delta_Q_inst(self, redshift):
        """Instantaneous heat injection from HeII reionization (absorption of photons with E<Emax) [eV/cm^3]"""
        Q_inst = self.cosmo.nHe(redshift)*((self.alpha_q/(self.alpha_q-1.))*((self.Emax**(-self.alpha_q+1.)-self.E0_HeII**(-self.alpha_q+1.))/(self.Emax**(-self.alpha_q)-self.E0_HeII**(-self.alpha_q)))-self.E0_HeII)
        return Q_inst

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
        """Approximate optical depth that long MFP photons see-- may want to replace this with something better later
        but it's probably good enough right now"""
        xHeI = 0 #HeI should be ionized with HI by first galaxies
        xHeII = np.amax([1 - xHeI - self.hist.XHeIII(z), 0.]) #This essentially follows the evolution of HeIII fraction.
        func = lambda z: self.speed_of_light/(self.cosmo.Hubble(z)*(1+z))*self.sigmaHeII(E*(1.+z)/(1.+z0))*self.cosmo.nHe(z)*xHeII
        t = scipy.integrate.quad(func,z0,z)
        return t[0]

    def a_norm(self, redshift):
        """Normalization of emissivity-- requires that the total ionizing emissivity of ionizing photons
        balances the number of ionizations plus recombinations.
        TODO: put in better clumping factor prescription and T_est"""
        absfac = self.clumping_fac*self.alphaHeppTest*self.hist.XHeIII(redshift)*self.cosmo.ne(redshift)
        A = ((self.alpha_q)*self.cosmo.nHe(redshift))/(self.E0_HeII**(-self.alpha_q))*(self.hist.dXHeIIIdz(redshift)*(-self.cosmo.Hubble(redshift)*(1+redshift))+ absfac)
        return A

    def specific_intensity(self, z0, E):
        """Specific intensity based on powerlaw QSO spectrum"""
        func = lambda z: (self.speed_of_light/(4.*np.pi))*(1./(self.cosmo.Hubble(z)*(1+z)))*(1+z0)**3./((1+z)**3.)*self.a_norm(z)*E**(-self.alpha_q)*np.exp(-self.tau(z,z0,E))
        J_E = scipy.integrate.quad(func,z0,10., limit=100)
        return J_E[0]

    def dQ_hard_dz(self, redshift, E_lim = 1000.):
        """Uniform heating rate from long MFP hard photons only (E_gamma > Emax), dQ/dz. This is making the assumption that all Helium is in the form of HeII."""
        func = lambda E: ((E-self.E0_HeII)/E)*self.specific_intensity(redshift,E)*self.sigmaHeII(E)*(E)**(-self.alpha_q)
        w = scipy.integrate.quad(func,self.Emax,E_lim)
        dQdz = 4.*np.pi*self.eVtoerg*self.cosmo.nHe(redshift)*w[0]*1./(self.cosmo.Hubble(redshift)*(1+redshift))
        return dQdz

    def dGamma_hard_dt(self, redshift, E_lim = 1000.):
        """Photoionization heating of hard photons only (E_gamma > Emax), dGamma/dt. Units are erg/s/cm^3."""
        def _dGamma_dt_int(zz, E):
            """Integrand for the the double integration over E and J(E,z)"""
            intensity = (self.speed_of_light/(4.*np.pi))*(1./(self.cosmo.Hubble(zz)*(1+zz)))*(1+redshift)**3./((1+zz)**3.)*self.a_norm(zz)*np.exp(-self.tau(zz,redshift,E))
            dGamma_int = ((E-self.E0_HeII)/E)*intensity *self.sigmaHeII(E)*(E)**(-self.alpha_q)
            return dGamma_int
        #Do the double integral of E from Emax to Elim and zz from redshift to 10.
        w = scipy.integrate.dblquad(_dGamma_dt_int,self.Emax,E_lim, redshift, 10)
        xHeII = np.amax([1 - self.hist.XHeIII(redshift), 0.])
        dGammadt = 4.*np.pi*w[0]*self.eVtoerg*self.cosmo.nHe(redshift)*xHeII
        return dGammadt

    def WriteInterpTable(self, outfile, numz = 100):
        """Built the interpolation table file, the main output of this code, loadable by the MP-Gadget reionization module."""
        print("Setting up interpolation table!")

        z_quasar = np.logspace(np.log10(self.hist.z_i),np.log10(self.hist.z_f),numz)
        dQ_LMFP_dat = [self.dGamma_hard_dt(zqso) for zqso in z_quasar]
        XHeIII = [self.hist.XHeIII(zqso) for zqso in z_quasar]

        print('Creating table ',outfile)
        with open(outfile , 'w') as f:
            header = "#File parameters for this input file: Emax = %g, alpha_q = %g, Clumping factor = %g, Simple linear history or QSO history = %s\n" % (self.Emax, self.alpha_q, self.clumping_fac, self.hist)
            f.write(header)
            f.write('#Units of heating rate (3rd column) are erg/s/cm^3 \n')
            f.write('{0:f} \n'.format(self.alpha_q))
            f.write('{0:f} \n'.format(self.Emax))
            for zqso, xHe, dQ_LMFP in zip(z_quasar, XHeIII, dQ_LMFP_dat):
                f.write('{0:e} {1:e} {2:e} \n'.format(zqso, xHe, dQ_LMFP))

class LinearHistory:
    """Makes a HeII reionization history where X_HeIII is a linear function of redshift"""
    def __init__(self, z_i, z_f):
        self.z_i = z_i
        self.z_f = z_f

    def XHeIII(self, redshift):
        """XHeIII history that is linear with redshift.
           Set initial and final HeII reion redshifts (default is reion_z_f=2.8)."""
        return np.clip((redshift - self.z_i) /(self.z_f-self.z_i), a_min = 0, a_max = 1)

    def dXHeIIIdz(self, redshift):
        """Change in the derivative of XHeIII, where XHeIII evolves linearly with redshift.
           Set initial and final HeII reion redshifts (default is reion_z_f=2.8)."""
        if self.z_f <= redshift <= self.z_i:
            return 1./(self.z_f-self.z_i)
        return 0.

class QuasarHistory:
    """Determines the HeII reionization history from a quasar emissivity function.
    Note: the initial reionization redshift is when the neutral fraction is zero"""
    def __init__(self, cosmo, z_i = 6, z_f = 2, alpha_q = 1.7, clumping_fac = 3.):
        self.h_erg_s = 6.626e-27 #erg s
        self.mpctocm = 3.086e24
        self.alpha_q = alpha_q
        self.cosmo=cosmo
        self.alphaHeppTest = alphaHepp(15000)
        self.clumping_fac = clumping_fac
        self.z_i = z_i
        self.z_f = z_f
        self.xHeII_interp = self._makexHeIIInterp()

    def XHeIII(self, redshift):
        """HeIII fraction over cosmic time based on a QSO emissivity function."""
        return np.exp(self.xHeII_interp(redshift))-1e-30

    def dXHeIIIdz(self, redshift):
        """Change in XHeIII, where XHeIII evolves based on a QSO emissivity function fit."""
        return self.dXHeIIIdz_int(self.XHeIII(redshift), redshift)

    def dXHeIIIdz_int(self, xHeIII, redshift):
        """Sets up differential eq."""
        cosfac = self.cosmo.nHe(redshift)*(self.cosmo.Hubble(redshift)*(1+redshift))
        dXHeIIIdz = -(self.quasar_emissivity_Kulkarni19_21(redshift) - self.clumping_fac*self.alphaHeppTest*self.cosmo.ne(redshift)*xHeIII*self.cosmo.nHe(redshift))/cosfac
        return dXHeIIIdz

    def xHeIII_quasar(self, zmin, zmax, numz = 1000):
        """Makes an interpolation table of the HeII reionization history: z, XHeIII"""
        dataarr = np.zeros([2,numz])
        dataarr[0,:] = np.linspace(zmax,zmin, numz)
        x = scipy.integrate.odeint(self.dXHeIIIdz_int, np.zeros(numz), dataarr[0,:])
        dataarr[1,:] = [min(x[i,0], 1) for i in range(numz)]
        return dataarr

    def _makexHeIIInterp(self):
        """Produces outfile where columns are z, xHeIII, and number of ionizing photons per nHe produced. Returns an interpolation function."""
        dataarr = self.xHeIII_quasar(self.z_f, self.z_i)
        return scipy.interpolate.interp1d(dataarr[0,:], np.log(1e-30+dataarr[1,:]), bounds_error=False, fill_value=0.0)

    def quasar_emissivity_HM12(self, redshift):
        """Proper emissivity of HeII ionizing photons from Haardt & Madau (2012) (1105.2039.pdf eqn 37)"""
        enhance_fac=1
        epsilon_nu = enhance_fac*3.98e24*(1+redshift)**7.68*np.exp(-0.28*redshift)/(np.exp(1.77*redshift) + 26.3) #erg s^-1 MPc^-3 Hz^-1
        e = epsilon_nu/(self.h_erg_s*self.alpha_q)/(self.mpctocm**3)*4.**(-self.alpha_q)
        return e

    def quasar_emissivity_K15(self, redshift):
        """Proper emissivity of HeII ionizing photons from Khaire + (2015)"""
        epsilon_nu = 10.**(24.6)*(1.+redshift)**8.9 * np.exp(-0.36*redshift)/(np.exp(2.2*redshift)+25.1)  #erg s^-1 MPc^-3 Hz^-1
        e = epsilon_nu/(self.h_erg_s*self.alpha_q)/(self.mpctocm**3)*4.**(-self.alpha_q)
        return e

    def quasar_emissivity_Kulkarni19_18(self, redshift):
        """Proper emissivity of HeII ionizing photons from Kulkarni + (2019) with limiting magnitude -18"""
        epsilon_nu1450 = 10.**(24.72)*(1.+redshift)**11.42 * np.exp(-2.1*redshift)/(np.exp(1.09*redshift)+38.56)  #erg s^-1 MPc^-3 Hz^-1
        epsilon_nu = epsilon_nu1450*(912/1450)**0.61
        e = epsilon_nu/(self.h_erg_s*self.alpha_q)/(self.mpctocm**3)*4.**(-self.alpha_q)
        return e

    def quasar_emissivity_Kulkarni19_21(self, redshift):
        """Proper emissivity of HeII ionizing photons from Kulkarni + (2019) with limiting magnitude -21"""
        #Note the factor of 1+z ^3 compared to Kulkarni 2019: there is a comoving change!
        epsilon_nu1450 = 10.**(23.91)*(1.+redshift)**11.26 * np.exp(-1.3*redshift)/(np.exp(1.62*redshift)+13.6)  #erg s^-1 MPc^-3 Hz^-1
        epsilon_nu = epsilon_nu1450*(912/1450)**0.61
        e = epsilon_nu/(self.h_erg_s*self.alpha_q)/(self.mpctocm**3)*4.**(-self.alpha_q)
        return e

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--alphaq', type=float, default=1.7, help='QSO spectral index', required=True)
    parser.add_argument('--Emax', type=float, default=150, help='Threshold long-mean-free-path photon energy in eV', required=False)
    parser.add_argument('--cf', type=float, default=3., help='Subgrid clumping factor', required=False)
    parser.add_argument('--z_i', type=float, default=-1, help='Start redshift of helium reionization', required=False)
    parser.add_argument('--z_f', type=float, default=-1, help='End redshift of helium reionization', required=False)
    parser.add_argument('--hist', type=str, default="linear", help='Type of reionization history', required=True, choices=["linear", "quasar"])
    parser.add_argument('--outfile', type=str, default="HeIIReionizationTable", help='Name of file to save to', required=False)
    args = parser.parse_args()

    if args.z_i < 0:
        if args.hist == "linear":
            args.z_i = 4.0
        else:
            args.z_i = 6.0
    if args.z_f < 0:
        if args.hist == "linear":
            args.z_f = 2.8
        else:
            args.z_f = 2.5

    heat = HeIIheating(hist = args.hist, z_i = args.z_i, z_f= args.z_f, Emax=args.Emax, alpha_q = args.alphaq, clumping_fac = args.cf)
    heat.WriteInterpTable(args.outfile)
