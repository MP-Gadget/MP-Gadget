
# Customization; see Options.mk.example
CONFIG ?= ../Options.mk

include $(CONFIG)

TESTED = power thermal
TESTBIN := $(TESTED:%=.objs/test_%) $(MPI_TESTED:%=.objs/test_%)
MPISUITE = $(MPI_TESTED:%=test_%)
SUITE = $(TESTED:%=test_%) $(MPISUITE)

include ../Makefile.rules

OBJS = power.o zeldovich.o glass.o save.o thermal.o

OBJS := $(OBJS:%.o=.objs/%.o)

.PHONY: all test run-tests

all : libgenic.a

.objs/test_%: tests/test_%.c .objs/%.o ../tests/stub.c ../tests/cmocka.c ../libgadget/libgadget-utils.a ../libgadget/libgadget.a
	$(MPICC) $(TCFLAGS) -I../tests/ $^ $(LIBS) -o $@

build-tests: $(TESTBIN)

test : build-tests
	trap 'err=1' ERR; for tt in $(SUITE) ; do \
		if [[ "$(MPISUITE)" =~ .*$$tt.* ]]; then \
			mpirun -np 4 .objs/$$tt ;  \
		else \
			.objs/$$tt ; \
		fi ;  \
	done; exit $$err

libgenic.a: $(OBJS)
	$(AR) r $@ $(OBJS)

DEPS := $(OBJS:.o=.d)
-include $(DEPS)

clean:
	rm -rf .objs libgenic.a

