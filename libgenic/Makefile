
# Customization; see Options.mk.example
CONFIG ?= ../Options.mk

include $(CONFIG)

INCL=../libgadget/config.h \
    power.h allvars.h thermal.h proto.h

include ../Makefile.rules

OBJS = power.o allvars.o \
  	zeldovich.o save.o thermal.o

OBJS := $(OBJS:%.o=.objs/%.o)

all : libgenic.a

libgenic.a: $(OBJS)
	$(AR) rv $@ $(OBJS)

clean:
	rm -rf $(OBJS) libgenic.a
