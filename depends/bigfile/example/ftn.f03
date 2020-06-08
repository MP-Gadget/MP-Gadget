program test
    integer, parameter :: Lmax = 512
    real*4 :: data0(1:Lmax+2,1:Lmax,1:Lmax),data1D((Lmax +2)* Lmax * Lmax)
    integer :: nread, ntot
    integer :: q
    character(len=100) :: filename
    equivalence(data0,data1D)
    integer :: fileid
    INTEGER, DIMENSION(13) :: buff

    integer :: elsize = 4 ! sizeof(real*4) how to do this in fortran?

    ntot = 0
    do fileid=0, 4
        write(filename, '("/home/yfeng1/source/fastpm/tests/results-ic/IC/LinearDensityK/", Z0.6)'), fileid
        call stat(filename, buff)
        nread = buff(8) / elsize
        open(100 + fileid,file=filename,status='old',form='unformatted', access='stream')
        read(100+fileid) (data1D(ntot + q), q=1, nread)
        ntot  = ntot + nread
        print*, ntot
        close(100 + fileid)
    enddo
end program
