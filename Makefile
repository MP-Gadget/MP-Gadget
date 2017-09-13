# Customization; see Options.mk.example
CONFIG ?= Options.mk

include $(CONFIG)

# Main Rules
include Makefile.rules
#
# Testing Rules
include Makefile.tests
