# Customization; see Options.mk.example
CONFIG ?= Options.mk

include $(CONFIG)

FILES = $(shell git ls-files)

# define the version
# (easier to extract from other utils, e.g. python)

# Main Rules
include Makefile.rules
#
# Testing Rules
include Makefile.tests

# Testing Rules
include Makefile.maint
