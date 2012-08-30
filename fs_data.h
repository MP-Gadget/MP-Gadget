c
c Written by S. Glover, AMNH, 2004-2005
c
c Atomic data (transition probabilities, energies) required 
c for computing the fine structure cooling rates. It seems 
c sensible to put these in a separate header, as we only need 
c them in a couple of places (cool_l, cool_h)

#ifdef CHEMCOOL
      REAL kb 
      parameter (kb = 1.38066d-16)

      REAL oxa10, oxa20, oxa21, oxe10, oxe20, oxe21
      parameter (oxa10 = 8.865d-5) 
      parameter (oxa20 = 1.275d-10)
      parameter (oxa21 = 1.772d-5) 
      parameter (oxe10 = 2.2771d2 * kb) 
      parameter (oxe21 = 9.886d1  * kb)
      parameter (oxe20 = oxe10 + oxe21)

      REAL cIa10, cIa20, cIa21, cIe10, cIe20, cIe21
      parameter (cIa10 = 7.932d-8)
      parameter (cIa20 = 2.054d-14)
      parameter (cIa21 = 2.654d-7) 
      parameter (cIe10 = 2.360d1 * kb) 
      parameter (cIe21 = 3.884d1 * kb)
      parameter (cIe20 = cIe10 + cIe21)

      REAL cIIa10, cIIe10
      parameter (cIIa10 = 2.291d-6) 
      parameter (cIIe10 = 9.125d1 * kb) 

      REAL siIa10, siIa20, siIa21, siIe10, siIe20, siIe21
      parameter (siIa10 = 8.4d-6)
      parameter (siIa20 = 2.4d-10) 
      parameter (siIa21 = 4.2d-5) 
      parameter (siIe10 = 1.1d2 * kb) 
      parameter (siIe21 = 2.1d2 * kb)
      parameter (siIe20 = siIe10 + siIe21)

      REAL siIIa10, siIIe10
      parameter (siIIa10 = 2.17d-4)
      parameter (siIIe10 = 4.1224d2 * kb)

#endif /* CHEMCOOL */
