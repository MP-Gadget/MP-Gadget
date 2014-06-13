cimport numpy
from libc.stddef cimport ptrdiff_t
from libc.string cimport strcpy
import numpy

cdef extern from "bigfile.c":
    struct BigFile:
        char * basename

    struct BigBlock:
        char * dtype
        int nmemb
        char * basename
        size_t size
        int Nfile

    struct BigBlockPtr:
        pass

    struct BigArray:
        int ndim
        char * dtype
        ptrdiff_t * dims
        ptrdiff_t * strides
        size_t size
        void * data

    void big_file_set_buffer_size(size_t bytes)
    int big_block_open(BigBlock * bb, char * basename)
    int big_block_create(BigBlock * bb, char * basename, char * dtype, int nmemb, int Nfile, size_t fsize[])
    int big_block_close(BigBlock * block)
    int big_block_seek(BigBlock * bb, BigBlockPtr * ptr, ptrdiff_t offset)
    int big_block_seek_rel(BigBlock * bb, BigBlockPtr * ptr, ptrdiff_t rel)
    int big_block_read(BigBlock * bb, BigBlockPtr * ptr, BigArray * array)
    int big_block_write(BigBlock * bb, BigBlockPtr * ptr, BigArray * array)
    int big_block_set_attr(BigBlock * block, char * attrname, void * data, char * dtype, int nmemb)
    int big_block_get_attr(BigBlock * block, char * attrname, void * data, char * dtype, int nmemb)
    int big_array_init(BigArray * array, void * buf, char * dtype, int ndim, size_t dims[], ptrdiff_t strides[])

    int big_file_open_block(BigFile * bf, BigBlock * block, char * blockname)
    int big_file_create_block(BigFile * bf, BigBlock * block, char * blockname, char * dtype, int nmemb, int Nfile, size_t fsize[])
    int big_file_open(BigFile * bf, char * basename)
    void big_file_close(BigFile * bf)

def set_buffer_size(bytes):
    big_file_set_buffer_size(bytes)

class BigFileError(Exception):
    pass
cdef class PyBigFile:
    cdef BigFile bf
    cdef int closed

    def __cinit__(self):
        self.closed = True
    def __init__(self, filename):
        big_file_open(&self.bf, filename)
        self.closed = False

    def __dealloc__(self):
        if not self.closed:
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

    def close(self):
        big_file_close(&self.bf)
        self.closed = True

    def open(self, block):
        cdef PyBigBlock rt = PyBigBlock()
        if 0 != big_file_open_block(&self.bf, &rt.bb, block):
            raise BigFileError("can't open block")
        rt.closed = False
        return rt

cdef class PyBigBlock:
    cdef BigBlock bb
    cdef int closed

    property size:
        def __get__(self):
            return self.bb.size
    
    property dtype:
        def __get__(self):
            return numpy.dtype((self.bb.dtype, self.bb.nmemb))
    def __cinit__(self):
        self.closed = True

    def __init__(self):
#        , filename, create=False, Nfile=None, dtype=None, size=None):
#        if create:
#            self.create(filename, Nfile, dtype, size)
#        else:
#            self.open(filename)
        pass

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

    @staticmethod
    def open(filename):
        cdef PyBigBlock self = PyBigBlock()
        if 0 != big_block_open(&self.bb, filename):
            raise BigFileError("Failed to open file")
        self.closed = False
        return self
    @staticmethod
    def create(filename, Nfile, dtype, size):
        cdef PyBigBlock self = PyBigBlock()
        dtype = numpy.dtype(dtype)
        assert len(dtype.shape) <= 1
        if len(dtype.shape) == 0:
            items = 1
        else:
            items = dtype.shape[0]
        cdef numpy.ndarray fsize = numpy.empty(dtype='intp', shape=Nfile)
        fsize[:] = (numpy.arange(Nfile) + 1) * size / Nfile \
                 - (numpy.arange(Nfile)) * size / Nfile
        if 0 != big_block_create(&self.bb, filename, dtype.base.str,
                items, Nfile, <size_t*> fsize.data):
            raise BigFileError("Failed to create file")
        self.closed = False
        return self

    def write(self, start, numpy.ndarray buf):
        cdef BigArray array
        cdef BigBlockPtr ptr
        big_array_init(&array, buf.data, buf.dtype.str, 
                buf.ndim, 
                <size_t *> buf.shape,
                <ptrdiff_t *> buf.strides)
        big_block_seek(&self.bb, &ptr, start)
        big_block_write(&self.bb, &ptr, &array)

    def __getitem__(self, range):
        if isinstance(range, slice):
            start, end, step = range.indices(self.size)
            return self.read(start, end - start)[::step]
        else:
            raise KeyError("only support indexing with a slice")

    def read(self, start, length):
        cdef numpy.ndarray result 
        cdef BigArray array
        cdef BigBlockPtr ptr
        cdef int i
        if length == -1:
            length = self.size - start
        if length + start > self.size:
            length = self.size - start
        result = numpy.empty(dtype=self.dtype, shape=length)
        big_array_init(&array, result.data, self.bb.dtype, 
                result.ndim, 
                <size_t *> result.shape,
                <ptrdiff_t *> result.strides)
        big_block_seek(&self.bb, &ptr, start)
        big_block_read(&self.bb, &ptr, &array)
        return result

    def close(self):
        if not self.closed:
            big_block_close(&self.bb)
            self.closed = True

    def __dealloc__(self):
        if not self.closed:
            big_block_close(&self.bb)

    def __repr__(self):
        return "<BigBlock: %s dtype=%s, size=%d>" % (self.bb.basename,
                self.dtype, self.size)


