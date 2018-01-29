CONFIG = ../Options.mk
include $(CONFIG)

LIBRARIES=lib/libbigfile-mpi.a lib/libmpsort-mpi.a lib/libpfft_omp.a
depends: $(LIBRARIES)

lib/libclass.a: Makefile
	bash class/install_class.sh $(PWD)/class 2.6.1 $(PWD)

lib/libmpsort-mpi.a: mpsort/mpsort-mpi.c
	mkdir -p lib; \
	mkdir -p include; \
	cd mpsort; \
	make install PREFIX=$(PWD) CC="$(MPICC)" MPICC="$(MPICC)" CFLAGS="$(OPTIMIZE)"

lib/libbigfile-mpi.a: bigfile/src/bigfile-mpi.c
	mkdir -p lib; \
	mkdir -p include; \
	cd bigfile/src; \
	make install PREFIX=$(PWD) CC="$(MPICC)" MPICC="$(MPICC)" CFLAGS="$(OPTIMIZE)"

lib/libpfft_omp.a:
	mkdir -p lib; \
	mkdir -p include; \
	#Using -ipo causes icc to crash.
	MPICC="$(MPICC)" CC="$(MPICC)" CFLAGS="$(filter-out -ipo,$(OPTIMIZE)) -I $(PWD)/include -L$(PWD)/lib" \
        sh $(PWD)/install_pfft.sh $(PWD)/

clean:
	rm -rf $(LIBRARIES)
	cd bigfile/src; make clean
	cd mpsort; make clean

