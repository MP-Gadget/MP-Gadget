"""Checks output of star-forming hydro simulation with star formation and BHs."""
import numpy as np
from numpy.testing import assert_allclose
import bigfile

def check_snapshot(pig, stars=0, bh=0):
    """Check we have formed stars and bhs."""
    bff = bigfile.BigFile(pig)
    partingroup = bff["Header"].attrs["NumPartInGroupTotal"]
    assert partingroup[0] > partingroup[4]
    assert partingroup[1] > partingroup[4]
    #Tolerance may be too tight: there is significant variance in a small box.
    assert_allclose(partingroup[4], stars, rtol=0.1, atol=2)
    assert_allclose(partingroup[5], bh, rtol=0.1, atol=2)
    #Check the stars and BHs ended up in the group"""
    gsmf = bff["FOFGroups/LengthByType"][:]
    for i in (4,5):
        assert np.sum(gsmf[:,i]) == partingroup[i]

def check_sfr(sfrfile="output/sfr.txt"):
    """Check that sfr.txt behaves as expected."""
    sfr = np.loadtxt(sfrfile)
    #Make sure some stars formed!
    istars = np.where(sfr[:,-1] > 0)
    #z=7.7
    assert np.min(sfr[:, 0][istars]) < 0.115
    #The format of sfr.txt is:
    #0 - scale factor.
    #1 - Total expected mass formed in stars
    #2 - totsfrrate = current star formation rate in active particles in Msun/year,
    #3 - rate_in_msunperyear = expected stellar mass formation rate in Msun/year from total_sm,
    #4 - total_sum_mass_stars = actual mass of stars formed this timestep (discretized total_sm)
    #5. avg_dtime = the average timsetep (dt) for the currently active star particles
    #6. total_sum_part: the number of actively star-forming particles
    #7. tot_new stars: number of new star particles spawned or converted this timestep
    assert 0.5 < np.median(sfr[:,2])/np.median(sfr[:,3]) < 1.5
    #Always a PM step
    ii = np.argmax(sfr[:, -1])
    assert 0.25 < sfr[ii, 1] / sfr[ii, 4] < 1

def check_bh(bhfile="output/blackholes.txt"):
    """Check that blackholes.txt is as expected."""
    #Format is:
    #0 - scale factor
    #1 - Total number of black holes
    #2 - Total mass of black holes
    #3 - Total mdot
    #4 - Total mdot converted to msun/year
    #5 - Sum of Mdot / Mass converted to units of the Eddington ratio
    bh = np.loadtxt(bhfile)
    #1 BH, no later than this, with seed mass.
    seedmass = 2.00062e-05
    assert 0.15 < bh[0, 0] < 0.16
    assert bh[0, 1] == 1
    assert np.abs(bh[0, 2] - seedmass) < 1e-7
    #Some accretion by the end, but not too much.
    assert bh[-1, 1] >= 4
    assert bh[-1, 1] * seedmass* 1.1 > bh[-1, 2] > bh[-1, 1] * seedmass

# Mass functions
if __name__ == "__main__":
    check_snapshot('output/PIG_000', stars=10, bh=0)
    check_snapshot('output/PIG_001', stars=58, bh=0)
    check_snapshot('output/PIG_002', stars=501, bh=4)
    check_sfr()
    check_bh()
