# vim: set ft=make:

# Customization; see Options.mk.example

CONFIG ?= ../Options.mk

include $(CONFIG)

INCL = config.h \
	densitykernel.h \
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

.PHONY: all

libgadget-utils.a: $(GADGET_UTILS_OBJS)
	$(AR) rv $@ $(GADGET_UTILS_OBJS)

libgadget.a: $(GADGET_OBJS)
	$(AR) rv $@ $(GADGET_OBJS)

clean:
	rm -rf .objs libgadget.a libgadget-utils.a

config.h: Makefile $(CONFIG)
	mkdir -p `dirname $@`
	MPICC="$(MPICC)" CFLAGS="$(CFLAGS)" OPT="$(OPT)" OPTIMIZE="$(OPTIMIZE)" VERSION="$(VERSION)" bash makeconfig.sh $@

