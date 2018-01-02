# Customization; see Options.mk.example
CONFIG ?= Options.mk

include $(CONFIG)

# define the version
# (easier to extract from other utils, e.g. python)
include Makefile.version

# Main Rules
include Makefile.rules
#
# Testing Rules
include Makefile.tests
