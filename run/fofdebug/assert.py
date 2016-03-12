import bigfile
from numpy.testing import assert_array_equal
import numpy

fc = bigfile.BigFile('PIG_003_correct')
dc = bigfile.BigData(fc['FOFGroups/'], ['LengthByType', 'MassCenterPosition'])[:]
gasc = bigfile.BigData(fc['0/'], ['GroupID', 'ID'])[:]['GroupID']

f1 = bigfile.BigFile('PIG_003')
d1 = bigfile.BigData(f1['FOFGroups/'], ['LengthByType', 'MassCenterPosition'])[:]
gas1 = bigfile.BigData(f1['0/'], ['GroupID', 'ID'])[:]['GroupID']

gascgid = numpy.arange(1, len(dc) + 1).repeat(dc['LengthByType'][:, 0])
assert_array_equal(gasc, gascgid)

gas1gid = numpy.arange(1, len(d1) + 1).repeat(d1['LengthByType'][:, 0])
assert_array_equal(gas1, gas1gid)

print len(dc), len(d1)
print len(gasc), len(gas1)


d1.sort(order='LengthByType')
dc.sort(order='LengthByType')
d1 = d1[::-1]
dc = dc[::-1]
print '----- good -----'
for i in range(5):
    print dc[i], d1[i]
print '----- end good ----'
bad = (dc['LengthByType'] != d1['LengthByType']).any(axis=-1)
print '----- bad -----'
for i in bad.nonzero()[0]:
    print dc[i], d1[i]
print '----- end bad ----'

