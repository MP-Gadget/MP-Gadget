# vim: set ft=make:

# Customization; see Options.mk.example

CONFIG ?= ../Options.mk

include $(CONFIG)

INCL = densitykernel.h \
	forcetree.h \
	hci.h \
	petapm.h \
	run.h \
	timebinmgr.h \
	treewalk.h \
	allvars.h \
	partmanager.h \
	cooling.h   \
	domain.h   \
	exchange.h \
	slotsmanager.h     \
	checkpoint.h \
	physconst.h   \
	sfr_eff.h \
	timefac.h \
	blackhole.h \
	cosmology.h \
	drift.h     \
	fof.h  \
	gravshort.h  \
	petaio.h  \
	powerspectrum.h  \
	timestep.h  \
	walltime.h \
	kspace-neutrinos/omega_nu_single.h \
	kspace-neutrinos/delta_tot_table.h \
utils/peano.h \
utils/interp.h \
utils/paramset.h \
utils/endrun.h \
utils/memory.h \
utils/mymalloc.h \
utils/system.h \
utils/event.h \
utils/openmpsort.h \
utils/string.h

UTILS_TESTED = memory openmpsort interp

TESTED = hci \
	slotsmanager \
	powerspectrum \
	cosmology \
	forcetree \
	timefac \
	timebinmgr \
	exchange

MPI_TESTED = exchange

TESTBIN :=$(UTILS_TESTED:%=.objs/utils/test_%) $(TESTED:%=.objs/test_%) $(MPI_TESTED:%=.objs/test_%)
SUITE?= $(TESTED:%=test_%) $(UTILS_TESTED:%=utils/test_%)
MPISUITE = $(MPI_TESTED:%=test_%)

include ../Makefile.rules

FILES = $(shell git ls-files)

# define the version
# (easier to extract from other utils, e.g. python)
include ../Makefile.version

# Main Rules
#
# make V=1 to see the command line
#

GADGET_OBJS =  \
	 gdbtools.o hci.o\
	 fof.o fofpetaio.o petaio.o \
	 domain.o exchange.o slotsmanager.o allvars.o partmanager.o \
	 blackhole.o timebinmgr.o \
	 run.o drift.o stats.o \
	 timestep.o init.o checkpoint.o sfr_eff.o \
	 cooling.o \
	 density.o \
	 treewalk.o cosmology.o \
	 gravshort-tree.o gravshort-pair.o hydra.o  timefac.o \
	 gravpm.o powerspectrum.o \
	 forcetree.o \
	 petapm.o gravity.o \
	 densitykernel.o lightcone.o walltime.o\
	 runtests.o \
     kspace-neutrinos/omega_nu_single.o \
     kspace-neutrinos/delta_pow.o kspace-neutrinos/delta_tot_table.o \
     kspace-neutrinos/transfer_init.o

GADGET_UTILS_OBJS= \
utils/endrun.o \
utils/peano.o \
utils/interp.o \
utils/memory.o \
utils/mymalloc.o \
utils/system.o \
utils/paramset.o \
utils/event.o \
utils/openmpsort.o \
utils/string.o


GADGET_OBJS := $(GADGET_OBJS:%=.objs/%)
GADGET_UTILS_OBJS := $(GADGET_UTILS_OBJS:%=.objs/%)

all: libgadget.a libgadget-utils.a

.PHONY: all test run-tests

.objs/utils/test_%: tests/test_%.c .objs/utils/%.o ../tests/stub.c ../tests/cmocka.c libgadget-utils.a
	$(MPICC) $(CFLAGS) -I../tests/ $^ $(LIBS) -o $@

.objs/test_%: tests/test_%.c .objs/%.o ../tests/stub.c ../tests/cmocka.c libgadget-utils.a
	$(MPICC) $(CFLAGS) -I../tests/ $^ $(LIBS) -o $@

build-tests: $(TESTBIN)

test : build-tests
	trap 'err=1' ERR; for tt in $(SUITE) ; do \
		if [[ "$(MPISUITE)" =~ .*$$tt.* ]]; then \
			mpirun -np 4 .objs/$$tt ;  \
		else \
			.objs/$$tt ; \
		fi ;  \
	done; exit $$err

libgadget-utils.a: $(GADGET_UTILS_OBJS)
	$(AR) rv $@ $(GADGET_UTILS_OBJS)

libgadget.a: $(GADGET_OBJS) config.h
	$(AR) rv $@ $(GADGET_OBJS)

clean:
	rm -rf .objs libgadget.a libgadget-utils.a

config.h: Makefile $(CONFIG) $(INCL)
	mkdir -p `dirname $@`
	MPICC="$(MPICC)" CFLAGS="$(CFLAGS)" OPT="$(OPT)" OPTIMIZE="$(OPTIMIZE)" VERSION="$(VERSION)" bash makeconfig.sh $@

