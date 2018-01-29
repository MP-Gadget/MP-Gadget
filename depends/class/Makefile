CLASS_VERSION?=2.6.1
DEST?=_inst

TARBALL=class-v$(CLASS_VERSION).tar.gz
UNPACK=tmp-class-v$(CLASS_VERSION)
SRC=$(UNPACK)/class_public-$(CLASS_VERSION)

$(TARBALL):
	wget https://github.com/lesgourg/class_public/archive/v$(CLASS_VERSION).tar.gz \
		-O $(TARBALL)

$(SRC)/stamp.unzip: $(TARBALL)
	mkdir -p $(UNPACK)
	gzip -dc $(TARBALL) | tar xf - -C $(UNPACK)
	touch $@

$(SRC)/stamp.patch : $(SRC)/stamp.unzip \
	  class-2.6.0-a_max.patch \
	  class-2.6.0-tol-ncdm.patch
	patch -d $(SRC) -p1 < class-2.6.0-a_max.patch || exit 1 
	patch -d $(SRC) -p1 < class-2.6.0-tol-ncdm.patch || exit 1
	patch -d $(SRC) -p1 < class-2.6.1-pathsize.patch || exit 1
	touch $@

$(SRC)/libclass.a: $(SRC)/stamp.patch class.cfg Makefile
	cp Makefile.class $(SRC)/Makefile
	cp class.cfg $(SRC)/myclass.cfg
	cd $(SRC); make CLASSCFG=myclass.cfg libclass.a

$(DEST)/lib/libclass.a: $(SRC)/libclass.a Makefile
	echo $(DEST)/data
	mkdir -p $(DEST)/data
	cp -r $(SRC)/include $(DEST)
	mkdir -p $(DEST)/lib
	cp $(SRC)/libclass.a $(DEST)/lib
	mkdir -p $(DEST)/data
	./copy_data_files $(SRC) $(DEST)/data

unzip: $(SRC)/stamp.unzip
patch: $(SRC)/stamp.patch
build: $(SRC)/libclass.a
install: $(DEST)/lib/libclass.a
