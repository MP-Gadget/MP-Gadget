# Customization; see Options.mk.example
CONFIG ?= Options.mk

include $(CONFIG)
#
# define the version
# (easier to extract from other utils, e.g. python)
include Makefile.version
#

FILES = $(shell git ls-files)

all:
	cd depends; $(MAKE)
	cd libgadget; $(MAKE)
	cd libgenic; $(MAKE)
	cd gadget; $(MAKE)
	cd genic; $(MAKE)

clean :
	cd libgadget; $(MAKE) clean
	cd libgenic; $(MAKE) clean
	cd gadget; $(MAKE) clean
	cd genic; $(MAKE) clean

test:
	cd tests; $(MAKE) test


html: $(FILES)
	@pandoc -f rst -t markdown README.rst > README.tmp.md
	@sed -e "s;@VERSION@;$(VERSION)-`git describe --always --dirty --abbrev=10`;" \
	    -e "s;@MAINPAGE@;README.tmp.md;" \
	    -e "s;@DOT@;$(PWD)/maintainer/dot;" \
	       maintainer/Doxyfile.in > Doxyfile
	@doxygen Doxyfile
	@rm -rf README.tmp.md Doxyfile
	@echo x-ref source code generated. Start from file://$(PWD)/html/index.html

gh-pages: html
	ghp-import -p -f html

tag:
	@echo trying to tag a release $(VERSION).
	@if git describe --always --dirty --abbrev=10 | grep dirty ; then \
	    echo "FAILED: version not clean, cannot tag a release." ; exit 1; \
	else \
	    git tag $(VERSION); \
	    echo "Current list of tags : "; \
	    git tag; \
	    echo "Need to push the tag with"; \
	    echo "git push origin $(VERSION)"; \
	fi;

sdist:
	bash maintainer/git-archive-all.sh --prefix MPGadget-$(VERSION)/ -- - | gzip -c > MPGadget-$(VERSION).tar.gz
