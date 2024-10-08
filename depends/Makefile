CONFIG = ../Options.mk
include $(CONFIG)

.PHONY: depends
# MPICC ?= mpicc
MPICCDEP ?= mpicc
OPTIMIZE ?= -O2 -g -fopenmp -Wall
LIBRARIES=lib/libbigfile-mpi.a
depends: $(LIBRARIES)

lib/libbigfile-mpi.a: bigfile/src/bigfile-mpi.c
	mkdir -p lib; \
	mkdir -p include; \
	cd bigfile/src; \
	make install PREFIX=$(PWD) CC="$(MPICCDEP)" MPICC="$(MPICCDEP)" CFLAGS="$(OPTIMIZE)" AR="$(AR)"


clean: clean-fast

clean-fast:
	rm -rf $(LIBRARIES)
	cd bigfile/src; make clean

