# Customization; see Options.mk.example
CONFIG ?= ../Options.mk

include $(CONFIG)

include ../Makefile.rules

##
# Run all the tests.
#
#  make test
#
# run a selected suite of tests
#
#  make test SUITE=tests/test_cosmology
#
# To debug a test
#
#  make tests/test_cosmology
#  gdb tests/test_cosmology
#
##
TESTED = \
	hci \
	slotsmanager \
	interp \
	powerspectrum \
	cosmology \
	forcetree \
	timefac \
	timebinmgr \
	memory \
	openmpsort \
	genic-power \
	genic-thermal \
	exchange

MPI_TESTED = exchange

TESTBIN = $(TESTED:%=test_%) $(MPI_TESTED:%=test_%)
SUITE?= $(TESTED:%=test_%)
MPISUITE = $(MPI_TESTED:%=test_%)

.PHONY: test run-tests build-tests


build-tests: $(TESTBIN)

test : build-tests
	trap 'err=1' ERR; for tt in $(SUITE) ; do \
		if [[ "$(MPISUITE)" =~ .*$$tt.* ]]; then \
			mpirun -np 4 ./$$tt ;  \
		else \
			./$$tt ; \
		fi ;  \
	done; exit $$err

clean-tests:
	rm -f $(TESTBIN)

# this empty rule is important for tab-completion
$(TESTBIN):

test_%: test_%.c stub.c ../libgenic/libgenic.a ../libgadget/libgadget.a ../libgadget/libgadget-utils.a cmocka.c
	$(MPICC) $(CFLAGS) -I.  $^ $(LIBS) -o $@

