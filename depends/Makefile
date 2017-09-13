CONFIG = ../Options.mk
include $(CONFIG)

LIBRARIES=lib/libbigfile-mpi.a lib/libmpsort-mpi.a lib/libpfft_omp.a lib/libcmocka.a
depends: $(LIBRARIES)

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

lib/libcmocka.a:
	mkdir -p cmocka-build; cd cmocka-build; \
	cmake -DCMAKE_INSTALL_PREFIX=$(PWD) -DCMAKE_BUILD_TYPE=Debug -DWITH_STATIC_LIB=On ../cmocka; \
	make install
	# remove the so files
	rm -f lib/libcmocka.so*

clean:
	rm -rf $(LIBRARIES)
	cd bigfile/src; make clean
	cd mpsort; make clean

