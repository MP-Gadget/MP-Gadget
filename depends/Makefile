include ../Options.mk

LIBRARIES=lib/libbigfile-mpi.a lib/libmpsort-mpi.a lib/libpfft.a
depends: $(LIBRARIES)

lib/libmpsort-mpi.a:
	cd mpsort; \
	make install PREFIX=$(PWD) CC=$(CC) CFLAGS="$(OPTIMIZE)"

lib/libbigfile-mpi.a:
	cd bigfile/src; \
	make install PREFIX=$(PWD) CC=$(CC) MPICC=$(CC) CFLAGS="$(OPTIMIZE)"

lib/libpfft.a:
	MPICC="$(CC)" CFLAGS="$(OPTIMIZE) -I $(PWD)/include -L$(PWD)/lib" \
        sh $(PWD)/pfft-python/depends/install_pfft.sh $(PWD)/

clean:
	rm -rf $(LIBRARIES)

