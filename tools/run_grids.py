import get_xgrids
import sys
import numpy as np

d_arr = ['1/','0/','4/','0/','0/','0/']
v_arr = ['Value','NeutralHydrogenFraction','Value','J21','StarFormationRate','InternalEnergy']
w_arr = ['Mass','Mass','Mass','Mass','Weight','Mass']
n_arr = ['global','local','none','local','none','local']
redshifts = [8,7.8,7.6,7.4,7.2,7,6.8,6.6,6.4,6.2,6]
res = 100/400.

get_xgrids.run_multiple(datadir=sys.argv[1],outdir=sys.argv[2],datasets=d_arr,values=v_arr,weightings=w_arr,normtypes=n_arr,resolution=res,redshifts=redshifts)
