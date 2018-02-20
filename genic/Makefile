
# Customization; see Options.mk.example
CONFIG ?= ../Options.mk

include $(CONFIG)

INCL=../libgadget/config.h

include ../Makefile.rules

OBJS = main.o power.o allvars.o params.o \
	zeldovich.o save.o thermal.o

OBJS := $(OBJS:%.o=.objs/%.o)

MP-GenIC: $(OBJS) ../libgadget/libgadget-ic.a ../libgadget/libgadget-utils.a
	$(MPICC) $(OPTIMIZE) $^ $(LIBS) -o $@

clean:
	rm -rf $(OBJS) MP-GenIC
