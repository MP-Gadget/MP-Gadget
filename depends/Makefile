CONFIG = ../Options.mk
include $(CONFIG)

.PHONY: depends
.INTERMEDIATE: pfft
# MPICC ?= mpicc
MPICCDEP ?= mpicc
OPTIMIZE ?= -O2 -g -fopenmp -Wall
LIBRARIES=lib/libbigfile-mpi.a
FFTLIBRARIES=lib/libpfft_omp.a lib/libfftw3_mpi.a lib/libfftw3_omp.a
depends: $(LIBRARIES) $(FFTLIBRARIES)
$(FFTLIBRARIES): pfft

lib/libbigfile-mpi.a: bigfile/src/bigfile-mpi.c
	mkdir -p lib; \
	mkdir -p include; \
	cd bigfile/src; \
	make install PREFIX=$(PWD) CC="$(MPICCDEP)" MPICC="$(MPICCDEP)" CFLAGS="$(OPTIMIZE)" AR="$(AR)"

pfft: install_pfft.sh
	mkdir -p lib; \
	mkdir -p include; \
	#Using -ipo causes icc to crash.
	MPICC="$(MPICCDEP)" CC="$(MPICCDEP)" CFLAGS="$(filter-out -ipo,$(OPTIMIZE)) -I $(PWD)/include -L$(PWD)/lib" AR="$(AR)" RANLIB=$(RANLIB) \
        sh $(PWD)/install_pfft.sh $(PWD)/

clean: clean-fast clean-fft

clean-fast:
	rm -rf $(LIBRARIES)
	cd bigfile/src; make clean

clean-fft:
	rm -rf $(FFTLIBRARIES)
	rm -rf tmp-pfft-*/double
	rm -rf tmp-pfft-*/single
