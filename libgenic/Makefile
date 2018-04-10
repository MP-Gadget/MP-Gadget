
# Customization; see Options.mk.example
CONFIG ?= ../Options.mk

include $(CONFIG)

INCL=../libgadget/config.h \
    power.h allvars.h thermal.h proto.h

TESTED = power thermal
TESTBIN := $(TESTED:%=.objs/test_%) $(MPI_TESTED:%=.objs/test_%)
SUITE?= $(TESTED:%=test_%)
MPISUITE = $(MPI_TESTED:%=test_%)

include ../Makefile.rules

OBJS = power.o allvars.o \
  	zeldovich.o save.o thermal.o

OBJS := $(OBJS:%.o=.objs/%.o)

.PHONY: all test run-tests

all : libgenic.a

libgenic.a: $(OBJS)
	$(AR) rv $@ $(OBJS)

clean:
	rm -rf .objs libgenic.a

