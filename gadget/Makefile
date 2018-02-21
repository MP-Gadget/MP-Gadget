# Customization; see Options.mk.example
CONFIG ?= ../Options.mk

include $(CONFIG)

INCL=../libgadget/config.h

include ../Makefile.rules

OBJS = main.o params.o

OBJS := $(OBJS:%.o=.objs/%.o)

MP-Gadget: $(OBJS) ../libgadget/libgadget.a ../libgadget/libgadget-utils.a
	$(MPICC) $(OPTIMIZE) $^ $(LIBS) -o $@

clean:
	rm -rf MP-Gadget .objs
