import bigfile
from numpy.testing import assert_array_equal
import numpy

fc = bigfile.BigFile('PIG_003_correct')
dc = bigfile.BigData(fc['FOFGroups/'], ['LengthByType', 'MassCenterPosition'])[:]
gasc = bigfile.BigData(fc['0/'], ['GroupID'])[:]['GroupID']

f1 = bigfile.BigFile('PIG_003')
d1 = bigfile.BigData(f1['FOFGroups/'], ['LengthByType', 'MassCenterPosition'])[:]
gas1 = bigfile.BigData(f1['0/'], ['GroupID'])[:]['GroupID']

assert_array_equal(gasc, 
    numpy.arange(1, len(dc) + 1).repeat(dc['LengthByType'][:, 0]))

assert_array_equal(gas1, 
    numpy.arange(1, len(d1) + 1).repeat(d1['LengthByType'][:, 0]))

bad = (dc['LengthByType'] != d1['LengthByType']).any(axis=-1)
print dc[bad]
