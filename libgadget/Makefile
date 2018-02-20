# vim: set ft=make:

# Customization; see Options.mk.example

CONFIG ?= ../Options.mk

include $(CONFIG)

INCL = config.h \
densitykernel.h  endrun.h    forcetree.h  interp.h event.h hci.h \
mymalloc.h    paramset.h  petapm.h         proto.h    timebinmgr.h  treewalk.h \
allvars.h  partmanager.h  cooling.h         domain.h         exchange.h  slotsmanager.h     \
 openmpsort.h  peano.h     physconst.h      sfr_eff.h  timefac.h     utils-string.h \
blackhole.h  cosmology.h       drift.h          fof.h       gravshort.h  memory.h   \
param.h       petaio.h    powerspectrum.h  system.h   timestep.h    walltime.h \
kspace-neutrinos/omega_nu_single.h kspace-neutrinos/delta_tot_table.h

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
	 param.o paramset.o \
	 domain.o exchange.o slotsmanager.o allvars.o partmanager.o \
	 blackhole.o timebinmgr.o \
	 run.o drift.o begrun.o global.o \
	 timestep.o init.o io.o sfr_eff.o \
	 cooling.o \
	 density.o \
	 treewalk.o cosmology.o \
	 gravshort-tree.o gravshort-pair.o hydra.o  timefac.o \
	 gravpm.o powerspectrum.o \
	 forcetree.o peano.o \
	 petapm.o longrange.o \
	 densitykernel.o interp.o lightcone.o walltime.o\
	 runtests.o \
     kspace-neutrinos/omega_nu_single.o \
     kspace-neutrinos/delta_pow.o kspace-neutrinos/delta_tot_table.o \
     kspace-neutrinos/transfer_init.o

GADGET_UTILS_OBJS= endrun.o memory.o mymalloc.o system.o event.o openmpsort.o utils-string.o


GADGET_IC_OBJS= partmanager.o \
		 petapm.o \
        paramset.o \
		walltime.o \
        cosmology.o \
        endrun.o \
        kspace-neutrinos/omega_nu_single.o


GADGET_OBJS := $(GADGET_OBJS:%=.objs/%)
GADGET_UTILS_OBJS := $(GADGET_UTILS_OBJS:%=.objs/%)
GADGET_IC_OBJS := $(GADGET_IC_OBJS:%=.objs/%)

all: builddir libgadget.a libgadget-ic.a libgadget-utils.a

.PHONY: builddir all

libgadget-utils.a: $(GADGET_UTILS_OBJS)
	$(AR) rv $@ $(GADGET_UTILS_OBJS)

libgadget.a: $(GADGET_OBJS)
	$(AR) rv $@ $(GADGET_OBJS)

libgadget-ic.a: $(GADGET_IC_OBJS)
	$(AR) rv $@ $(GADGET_IC_OBJS)


clean:
	rm -rf .objs libgadget.a libgadget-utils.a libgadget-ic.a

config.h: Makefile $(CONFIG)
	mkdir -p `dirname $@`
	MPICC="$(MPICC)" CFLAGS="$(CFLAGS)" OPT="$(OPT)" OPTIMIZE="$(OPTIMIZE)" VERSION="$(VERSION)" bash makeconfig.sh $@

