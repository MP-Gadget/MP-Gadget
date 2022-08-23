
#21cmFAST in MP-Gadget rundown

*Syncpoints are set up every 10 Myr or so (free parameter) in order to trigger the excursion set.
*When we reach one of the UV syncpoints, three particle meshes are created for mass, stellar mass, and star formation rate, with the same communication and layout as the long-range force code
*We then step through filters of different scales PFFT and count photons in each cell (from stellar mass * photons per baryon) to determine which cells are ionised, and calculate J21 if so. This is the excursion set model presented in Mesinger et al 2011. (astro-ph:1003.3878) with UV background from Sobacchi & Mesinger 2013 (astro-ph:1301.6781).
*This J21 is then converted into ionisation and photoheating rates, assuming a certain spectral slope (free parameter) between the HI and HeII ionisation thresholds. This is then passed into the existing cooling functions.


##Notes:

*new parameters:
..*ExcursionSetReionOn : 0 or 1 to disable/enable the excursion set
..*ExcursionSetZStop : redshift at which the cooling code reverts to global ionising background
..*ExcursionSetZStart : redshift at which the excursion set starts
..*UVBGdim : dimension of the excursion set grids (can be different from the force grids)
..*ReionRBubbleMax : largest filter radius for the excursion set
..*ReionRBubbleMin : smallest filter radius for the excursion set
..*ReionDeltaRFactor : relative size of subsequent filter radii
..*ReionGammaHaloBias : **UNUSED** bias used in the meraxes grid model for calculating reionisation feedback in halos
..*ReionNionPhotPerBary : photons per stellar baryon, usually set to 4000 (TODO: cite)
..*AlphaUV : spectral slope for ionising radiation, currently a global constant
..*EscapeFractionNorm : escape fraction of photons from ionising sources at a 10^10 solar halo mass
..*EscapeFractionScaling : escape fraction power-law scaling with halo mass
..*UVBGTimestep : time in Myr between UVBG timesteps
..*ReionFilterType : Filter to use with 21CMFAST, 0 = real-space top-hat, 1 = k-space top-hat, 2 = gaussian
..*RtoMFilterType : Filter to use with 21CMFAST radius to mass, 0 = top-hat, 1 = gaussian
..*J21CoeffFile : path to the table containing photo-ionisaiton and heating rates versus spectral slope at J21 == 1

This excursion set model takes negligible time compared to the rest of the simulation. Tested up to 100Mpc boxes with 0.25 Mpc UV grid resolution.
