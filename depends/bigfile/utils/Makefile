CC=mpicc -g -O0
all: \
	bigfile-get-attr \
	bigfile-set-attr \
	bigfile-copy \
	bigfile-copy-mpi \
	bigfile-cat \
	bigfile-create \
	bigfile-ls \
	bigfile-iosim \
	$(NULL)

bigfile-get-attr: bigfile-get-attr.c ../src/libbigfile.a
	$(CC) -o $@ $< ../src/libbigfile.a -I../src
bigfile-set-attr: bigfile-set-attr.c ../src/libbigfile.a
	$(CC) -o $@ $< ../src/libbigfile.a -I../src
bigfile-copy: bigfile-copy.c ../src/libbigfile.a
	$(CC) -o $@ $< ../src/libbigfile.a -I../src
bigfile-copy-mpi: bigfile-copy-mpi.c ../src/libbigfile.a ../src/libbigfile-mpi.a
	$(CC) -o $@ $< ../src/libbigfile-mpi.a ../src/libbigfile.a -I../src
bigfile-cat: bigfile-cat.c ../src/libbigfile.a
	$(CC) -o $@ $< ../src/libbigfile.a -I../src
bigfile-create: bigfile-create.c ../src/libbigfile.a
	$(CC) -o $@ $< ../src/libbigfile.a -I../src
bigfile-ls: bigfile-ls.c ../src/libbigfile.a
	$(CC) -o $@ $< ../src/libbigfile.a -I../src
bigfile-iosim: bigfile-iosim.c ../src/libbigfile.a ../src/libbigfile-mpi.a
	$(CC) -o $@ $< ../src/libbigfile-mpi.a ../src/libbigfile.a -I../src
