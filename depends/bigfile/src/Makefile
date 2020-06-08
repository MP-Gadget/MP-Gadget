MPICC ?= mpicc
PREFIX ?=/usr
CFLAGS ?=
PIC ?= -fPIC

.PHONY: all

all: libbigfile.a libbigfile-mpi.a

install: libbigfile.a libbigfile-mpi.a
	install -d $(PREFIX)
	install -d $(PREFIX)/lib
	install -d $(PREFIX)/include
	install libbigfile.a $(PREFIX)/lib/libbigfile.a
	install libbigfile-mpi.a $(PREFIX)/lib/libbigfile-mpi.a
	install bigfile-mpi.h $(PREFIX)/include/bigfile-mpi.h
	install bigfile.h $(PREFIX)/include/bigfile.h

bigfile.o: bigfile.c bigfile.h bigfile-internal.h
	$(MPICC) $(CFLAGS) $(PIC) -o $@ -c bigfile.c
bigfile-mpi.o: bigfile-mpi.c bigfile-mpi.h bigfile-internal.h mp-mpiu.h
	$(MPICC) $(CFLAGS) $(PIC) -o $@ -c bigfile-mpi.c
mp-mpiu.o: mp-mpiu.c mp-mpiu.h
	$(MPICC) $(CFLAGS) $(PIC) -o $@ -c mp-mpiu.c

libbigfile.a: bigfile.o
	ar r $@ $^
	ranlib $@
libbigfile-mpi.a: bigfile-mpi.o mp-mpiu.o
	ar r $@ $^
	ranlib $@
clean:
	rm -f *.a *.o
