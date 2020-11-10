from bigfile import BigFile
from bigfile import BigBlock
from bigfile import BigFileMPI
from bigfile import BigData
from bigfile import BigFileClosedError
from bigfile import BigBlockClosedError
from bigfile import BigFileError

from bigfile import Dataset

import tempfile
import numpy
import shutil

from numpy.testing import assert_equal
from numpy.testing import assert_raises
from numpy.testing import assert_array_equal

import pytest

dtypes = [
    ('boolean', '?'),
    ('i4', 'i4'),
    ('u4', 'u4'),
    ('u8', 'u8'),
    ('f4', 'f4'),
    ('f8', 'f8'),
#   ('f4v', ('f4', (1,))),  # This case is not well defined. Bigfile will treat it as 'f4', thus automated tests will not work.
    ('f4s', ('f4', 1)),  # This case will start to fail when numpy starts to ('f4', 1) as ('f4').
    ('f4_2', ('f4', (2,))),
    ('c8', ('complex64')),
    ('c16', ('complex128')),
    ('c16_2', ('complex128', (2, ))),
]

from runtests.mpi import MPITest
import re

def sanitized_name(s):
    return re.sub("[^a-z0-9]", '_', s)

@MPITest([1])
def test_create(comm):
    fname = tempfile.mkdtemp()
    x = BigFile(fname, create=True)
    x.create('.')

    for name, d in dtypes:
        d = numpy.dtype(d)
        numpy.random.seed(1234)

        # test creating
        with x.create(name, Nfile=1, dtype=d, size=128) as b:
            data = numpy.random.uniform(100000, size=128*128).view(dtype=b.dtype.base).reshape([-1] + list(d.shape))[:b.size]
            b.write(0, data)

        with x[name] as b:
            assert_equal(b[:], data.astype(d.base))
            assert_equal(b[:],  b[...])

        # test creating
        data = numpy.random.uniform(100000, size=128*128).view(dtype=d.base).reshape([-1] + list(d.shape))[:128]
        with x.create_from_array(name, data) as b:
            pass

        with x[name] as b:
            assert_equal(b[:], data)

        # test writing with an offset
        with x[name] as b:
            b.write(1, data[0:1])
            assert_equal(b[1:2], data[0:1].astype(d.base))

        # test writing beyond file length
        with x[name] as b:
            caught = False
            try:
                b.write(1, data)
            except:
                caught = True
            assert caught
    assert_equal(set(x.blocks), set([name for name, d in dtypes]))
    import os
    os.system("ls -r %s" % fname)
    for b in x.blocks:
        assert b in x

    for b in x:
        assert b in x

    bd = BigData(x)
    assert set(bd.dtype.names) == set(x.blocks)
    d = bd[:]

    shutil.rmtree(fname)

@MPITest([1])
def test_create_odd(comm):
    fname = tempfile.mkdtemp()
    x = BigFile(fname, create=True)
    x.create('.')


    name = 'f4'
    d = numpy.dtype('f4')
    numpy.random.seed(1234)

    # test creating
    with x.create(name, Nfile=3, dtype=d, size=455**3) as b:
        data = numpy.random.uniform(100000, size=455**3).astype(d)
        b.write(0, data)

    import os
    os.system("ls -r %s" % fname)
    for b in x.blocks:
        assert b in x

    for b in x:
        assert b in x

    shutil.rmtree(fname)

@MPITest([1])
def test_append(comm):
    fname = tempfile.mkdtemp()
    x = BigFile(fname, create=True)

    name = 'f4'
    d = numpy.dtype(('f4', 3))
    numpy.random.seed(1234)

    data = numpy.random.uniform(100000, size=(100, 3)).astype('f4')
    # test creating
    with x.create(name, Nfile=3, dtype=d, size=100) as b:
        b.write(0, data)

        b.append(data, Nfile=2)
        with x.open(name) as bb:
            assert bb.size == 200
        assert b.size == 200

    with x.open(name) as b:
        assert b.Nfile == 5
        assert_equal(b[:100], data)
        assert_equal(b[100:], data)
        assert b.size == 200

    shutil.rmtree(fname)

@MPITest([1])
def test_fileattr(comm):
    import os.path
    fname = tempfile.mkdtemp()
    x = BigFile(fname, create=True)
    assert not os.path.exists(os.path.join(fname, 'attr-v2'))
    assert not os.path.exists(os.path.join(fname, '000000'))
    with x['.'] as bb:
        bb.attrs['value'] = 1234
        assert bb.attrs['value'] == 1234
    assert not os.path.exists(os.path.join(fname, 'header'))
    assert os.path.exists(os.path.join(fname, 'attr-v2'))

    shutil.rmtree(fname)

@MPITest([1])
def test_file_large_attr(comm):
    import os.path
    fname = tempfile.mkdtemp()
    x = BigFile(fname, create=True)
    data = numpy.ones(1024 * 128 * 8, dtype='f8')

    with x['.'] as bb:
        bb.attrs['value'] = data

    with x['.'] as bb:
        assert_equal(bb.attrs['value'], data)

    shutil.rmtree(fname)

@MPITest([1])
def test_casts(comm):
    fname = tempfile.mkdtemp()
    x = BigFile(fname, create=True)

    with x.create('block', Nfile=1, dtype='f8', size=128) as b:
        assert_raises(BigFileError, b.write, 0, numpy.array('aaaaaa'))
        b.write(0, numpy.array(True, dtype='?'))

@MPITest([1])
def test_passby(comm):
    fname = tempfile.mkdtemp()
    x = BigFile(fname, create=True)

    # half floats are pass-through types, no casting is supported
    data = numpy.array([3.0, 5.0], dtype='f2')
    with x.create('block', Nfile=1, dtype='f2', size=128) as b:
        b.write(0, data)
        assert_equal(b[:2], data)
        assert_raises(BigFileError, b.write, 0, numpy.array((30, 20.)))

@MPITest([1])
def test_dataset(comm):
    fname = tempfile.mkdtemp()
    x = BigFile(fname, create=True)
    x.create('.')

    for name, d in dtypes:
        dt = numpy.dtype(d)
        numpy.random.seed(1234)
        # test creating
        with x.create(name, Nfile=1, dtype=dt, size=128) as b:
            data = numpy.random.uniform(100000, size=128*128).view(dtype=b.dtype.base).reshape([-1] 
                    + list(dt.shape))[:b.size]
            b.write(0, data)

    bd = Dataset(x)
    assert set(bd.dtype.names) == set(x.blocks)
    assert isinstance(bd[:], numpy.ndarray)
    assert isinstance(bd['f8'], BigBlock)
    assert_equal(len(bd['f8'].dtype), 0)
    # tuple of one item is the same as non-tuple
    assert isinstance(bd[('f8',)], BigBlock)
    assert_equal(len(bd[('f8',)].dtype), 0)

    assert isinstance(bd['f8', :10], numpy.ndarray)
    assert_equal(len(bd['f8', :10]), 10)
    assert_equal(len(bd['f8', :10].dtype), 0)
    assert_equal(len(bd[['f8',], :10].dtype), 1)

    # tuple of one item is the same as non-tuple
    assert_equal(len(bd[('f8',), :10].dtype), 0)
    assert isinstance(bd[:10, 'f8'], numpy.ndarray)
    assert isinstance(bd['f8'], BigBlock)
    assert isinstance(bd[['f8', 'f4'],], Dataset)
    assert_equal(len(bd[['f8', 'f4'],].dtype), 2)
    assert isinstance(bd[['f8', 'f4'], :10], numpy.ndarray)

    for name, d in dtypes:
        assert_array_equal(x[name][:], bd[:][name])

    data1 = bd[:10]
    data2 = bd[10:20]

    bd[:10] = data2
    assert_array_equal(bd[:10], data2)

    bd[10:20] = data1
    assert_array_equal(bd[:10], data2)
    assert_array_equal(bd[10:20], data1)

    bd.append(data1)
    assert bd.size == 128 + 10
    assert_array_equal(bd[-10:], data1)

    shutil.rmtree(fname)

@MPITest([1])
def test_closed(comm):
    fname = tempfile.mkdtemp()
    x = BigFile(fname, create=True)
    x.create('.')
    x.close()
    assert x.blocks == []
    try:
        h = x['.']
    except BigFileClosedError:
        pass

@MPITest([1])
def test_attr_objects(comm):
    fname = tempfile.mkdtemp()
    x = BigFile(fname, create=True)
    with x.create('block', dtype=None) as b:
        def set_obj1():
            b.attrs['objects'] = numpy.array([object()])
        assert_raises(ValueError, set_obj1);
        def set_obj_scalar():
            b.attrs['objects'] = object()
        assert_raises(ValueError, set_obj_scalar);
    shutil.rmtree(fname)

@MPITest([1])
def test_attr(comm):
    fname = tempfile.mkdtemp()
    x = BigFile(fname, create=True)
    with x.create('.', dtype=None) as b:
        b.attrs['int'] = 128
        b.attrs['float'] = [128.0, 3, 4]
        b.attrs['string'] = 'abcdefg'
        b.attrs['complex'] = 128 + 128J
        b.attrs['bool'] = True
        b.attrs['arrayustring'] = numpy.array(u'unicode')
        b.attrs['arraysstring'] = numpy.array('str')

    with x.open('.') as b:
        assert_equal(b.attrs['int'], 128)
        assert_equal(b.attrs['float'], [128.0, 3, 4])
        assert_equal(b.attrs['string'],  'abcdefg')
        assert_equal(b.attrs['complex'],  128 + 128J)
        assert_equal(b.attrs['bool'],  True)
        b.attrs['int'] = 30
        b.attrs['float'] = [3, 4]
        b.attrs['string'] = 'defg'
        b.attrs['complex'] = 32 + 32J
        b.attrs['bool'] = False

    with x.open('.') as b:
        assert_equal(b.attrs['int'], 30)
        assert_equal(b.attrs['float'], [3, 4])
        assert_equal(b.attrs['string'],  'defg')
        assert_equal(b.attrs['complex'],  32 + 32J)
        assert_equal(b.attrs['bool'],  False)

    shutil.rmtree(fname)

@MPITest([1, 4])
def test_mpi_create(comm):
    if comm.rank == 0:
        fname = tempfile.mkdtemp()
        fname = comm.bcast(fname)
    else:
        fname = comm.bcast(None)
    x = BigFileMPI(comm, fname, create=True)
    for name, d in dtypes:
        d = numpy.dtype(d)
        numpy.random.seed(1234)

        # test creating
        with x.create(name, Nfile=1, dtype=d, size=128) as b:
            data = numpy.random.uniform(100000, size=128*128).view(dtype=b.dtype.base).reshape([-1] + list(d.shape))[:b.size]
            b.write(0, data)

        with x[name] as b:
            assert_equal(b[:], data.astype(d.base))

        # test writing with an offset
        with x[name] as b:
            b.write(1, data[0:1])
            assert_equal(b[1:2], data[0:1].astype(d.base))

        # test writing beyond file length
        with x[name] as b:
            caught = False
            try:
                b.write(1, data)
            except:
                caught = True
            assert caught
    assert_equal(set(x.blocks), set([name for name, d in dtypes]))

    for b in x.blocks:
        assert b in x

    for b in x:
        assert b in x

    bd = BigData(x)
    assert set(bd.dtype.names) == set(x.blocks)
    d = bd[:]

    comm.barrier()
    if comm.rank == 0:
        shutil.rmtree(fname)

@MPITest([1, 4])
def test_mpi_attr(comm):
    if comm.rank == 0:
        fname = tempfile.mkdtemp()
        fname = comm.bcast(fname)
    else:
        fname = comm.bcast(None)
    x = BigFileMPI(comm, fname, create=True)

    with x.create('.', dtype=None) as b:
        b.attrs['int'] = 128
        b.attrs['float'] = [128.0, 3, 4]
        b.attrs['string'] = 'abcdefg'

    with x.open('.') as b:
        assert_equal(b.attrs['int'], 128)
        assert_equal(b.attrs['float'], [128.0, 3, 4])
        assert_equal(b.attrs['string'],  'abcdefg')
        b.attrs['int'] = 30
        b.attrs['float'] = [3, 4]
        b.attrs['string'] = 'defg'

    with x.open('.') as b:
        assert_equal(b.attrs['int'], 30)
        assert_equal(b.attrs['float'], [3, 4])
        assert_equal(b.attrs['string'],  'defg')

    comm.barrier()
    if comm.rank == 0:
        shutil.rmtree(fname)

def test_version():
    import bigfile
    assert hasattr(bigfile, '__version__')

@MPITest(commsize=[1, 4])
def test_mpi_large(comm):
    if comm.rank == 0:
        fname = tempfile.mkdtemp()
        fname = comm.bcast(fname)
    else:
        fname = comm.bcast(None)
    x = BigFileMPI(comm, fname, create=True)

    size= 1024 * 1024
    for name, d in dtypes:
        d = numpy.dtype(d)
        numpy.random.seed(1234)

        # test creating with create_array; large enough for all types
        data = numpy.random.uniform(100000, size=4 * size).view(dtype=d.base).reshape([-1] + list(d.shape))[:size]
        data1 = comm.scatter(numpy.array_split(data, comm.size))

        with x.create_from_array(name, data1, memorylimit=1024 * 128) as b:
            pass

        with x[name] as b:
            assert_equal(b[:], data.astype(d.base))

    comm.barrier()
    if comm.rank == 0:
        shutil.rmtree(fname)

@MPITest(commsize=[4])
def test_mpi_badfilenames(comm):
    fname = tempfile.mkdtemp()
    fname = fname + '%d' % comm.rank
    assert_raises(BigFileError, BigFileMPI, comm, fname, create=True)

@MPITest(commsize=[1])
def test_threads(comm):
    # This test shall not core dump
    # raise many errors here and there on many threads

    from threading import Thread, Event
    import gc
    fname = tempfile.mkdtemp()
    x = BigFile(fname, create=True)

    b = x.create("Threading", Nfile=1, dtype='i8', size=128)

    old = gc.get_threshold()

    gc.set_threshold(1, 1, 1)
    E = Event()
    def func(i, b):
        E.wait()
        x['.'].attrs['v3'] = [1, 2, 3]
        err = 0
        for j in range(100 * i):
            try:
                with pytest.raises(BigFileError):
                    b.attrs['v 3'] = ['a', 'bb', 'ccc']

                b.write(0, numpy.ones(128))
            except BigBlockClosedError:
                err = err + 1

        b.close()

        x['Threading'].attrs['v3'] = [1, 2, 3]

    t = []
    for i in range(4):
        t.append(Thread(target = func, args=(i, b)))

    for i in t: i.start()

    E.set()

    for i in t: i.join()

    gc.set_threshold(*old)
    shutil.rmtree(fname)


@MPITest(commsize=[1])
def test_blank_attr(comm):
    fname = tempfile.mkdtemp()
    x = BigFile(fname, create=True)

    with x.create("Header", Nfile=1, dtype=None, size=128) as b:
        with pytest.raises(BigFileError):
            b.attrs['v 3'] = ['a', 'bb', 'ccc']

        with pytest.raises(BigFileError):
            b.attrs['v\t3'] = ['a', 'bb', 'ccc']

        with pytest.raises(BigFileError):
            b.attrs['v\n3'] = ['a', 'bb', 'ccc']

    with pytest.raises(BigFileError):
        x.create(" ", Nfile=1, dtype=None, size=128)

    with pytest.raises(BigFileError):
        x.create("\t", Nfile=1, dtype=None, size=128)

    with pytest.raises(BigFileError):
        x.create("\n", Nfile=1, dtype=None, size=128)
    shutil.rmtree(fname)

@MPITest(commsize=[1])
def test_pickle(comm):
    fname = tempfile.mkdtemp()
    x = BigFile(fname, create=True)

    # test creating
    column = x.create("abc", dtype='f8', size=128)
    
    import pickle
    str = pickle.dumps(column)
    column1 = pickle.loads(str)

    assert type(column) == type(column1)
    assert column.size == column1.size
    assert column.dtype == column1.dtype
    assert column.comm is column1.comm

    column.close()
    str = pickle.dumps(column)
    column1 = pickle.loads(str)

    str = pickle.dumps(x)
    x1 = pickle.loads(str)

    assert type(x) == type(x1)
    assert x1.basename == x.basename

    x.close()
    str = pickle.dumps(x)
    x1 = pickle.loads(str)
    assert tuple(sorted(x1.blocks)) == tuple(sorted(x.blocks))
    shutil.rmtree(fname)

@MPITest(commsize=[1])
def test_string(comm):
    fname = tempfile.mkdtemp()
    x = BigFile(fname, create=True)

    # test creating
    with x.create("Header", Nfile=1, dtype=None, size=128) as b:
        b.attrs['v3'] = ['a', 'bb', 'ccc']
        b.attrs['v32'] = [
                            ['a', 'bb', 'ccc'],
                            ['1', '22', '333'],]

        b.attrs['s'] = 'abc'
        b.attrs['l'] = 'a' * 65536

    with x.open("Header") as b:
        assert_equal(b.attrs['v3'], ['a', 'bb', 'ccc'])
        assert_equal(b.attrs['v32'], ['a', 'bb', 'ccc', '1', '22', '333'])
        assert_equal(b.attrs['s'], 'abc')
        assert_equal(b.attrs['l'], 'a' * 65536)

    shutil.rmtree(fname)

@MPITest([1])
def test_slicing(comm):
    fname = tempfile.mkdtemp()
    x = BigFile(fname, create=True)
    x.create('.')

    numpy.random.seed(1234)

    # test creating
    with x.create("data", Nfile=1, dtype=('f8', 32), size=128) as b:
        data = numpy.random.uniform(100000, size=(128, 32))
        junk = numpy.random.uniform(100000, size=(128, 32))

        b.write(0, data)

        with x['data'] as b:
            assert_equal(b[:], data)
            assert_equal(b[0], data[0])

        b[:len(junk)] = junk

        with x['data'] as b:
            assert_equal(b[:], junk)
            assert_equal(b[0], junk[0])

        b[3] = data[3]

        with x['data'] as b:
            assert_equal(b[3], data[3])

    shutil.rmtree(fname)
