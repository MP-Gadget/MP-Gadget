include ../Options.mk

LIBRARIES=lib/libbigfile-mpi.a lib/libmpsort-mpi.a lib/libpfft_omp.a
depends: $(LIBRARIES)

lib/libmpsort-mpi.a:
	mkdir -p lib; \
	mkdir -p include; \
	cd mpsort; \
	make install PREFIX=$(PWD) CC=$(CC) MPICC=$(CC) CFLAGS="$(OPTIMIZE)"

lib/libbigfile-mpi.a:
	mkdir -p lib; \
	mkdir -p include; \
	cd bigfile/src; \
	make install PREFIX=$(PWD) CC=$(CC) MPICC=$(CC) CFLAGS="$(OPTIMIZE)"

lib/libpfft_omp.a:
	mkdir -p lib; \
	mkdir -p include; \
	MPICC="$(CC)" CFLAGS="-O3 -I $(PWD)/include -L$(PWD)/lib" \
        sh $(PWD)/install_pfft.sh $(PWD)/

clean:
	rm -rf $(LIBRARIES)

