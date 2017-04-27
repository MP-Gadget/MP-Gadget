-- parameter file
------ Size of the simulation -------- 

-- For Testing
nc = 64
boxsize = 32.0

-------- Time Sequence ----
-- linspace: Uniform time steps in a
-- time_step = linspace(0.025, 1.0, 39)
-- logspace: Uniform time steps in loga
-- time_step = linspace(0.01, 1.0, 10)
time_step = linspace(0.1, 1, 10)

output_redshifts= {9.0}  -- redshifts of output

-- Cosmology --
omega_m = 0.307494
h       = 0.6774

-- Start with a power spectrum file
-- Initial power spectrum: k P(k) in Mpc/h units
-- Must be compatible with the Cosmology parameter
read_powerspectrum= "planck_camb_56106182_matterpower_z0.dat"
random_seed= 181170

-------- Approximation Method ---------------
force_mode = "fastpm"

pm_nc_factor = 2

np_alloc_factor= 4.0      -- Amount of memory allocated for particle

-------- Output ---------------

-- Dark matter particle outputs (all particles)
write_snapshot= "fastpm_reference/fastpm" 
-- 1d power spectrum (raw), without shotnoise correction
write_powerspectrum = "fastpm_reference/powerspec"

