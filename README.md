= MP-Gadget3 =

Massvively parallel version of P-Gadget3.

== Pre-Installation ==

pfft3 shall be installed. It is a bit tricky to install.

Luckily, on coma, pfft3 is already installed. To use it
```
source ~yfeng1/local/bin/setup.sh
```

== Installation ==
```
git clone https://github.com/rainwoodman/MP-Gadget3

cd MP-Gadget3
git submodules init

```

Now we need to look at Makefile.example
```
copy Makefile.example Makefile
```

Edit Makefile and enable some flags. 

An important variable is SYSTYPE. On COMA, set this to Warp will compile fine.
Otherwise, reference Makefile.Warp to build your own Makefile.MyMachine file and
set SYSTYPE=MyMachine in Makefile

The defaults shall work for most cases; it enables Pressure-Entropy SPH and Blackhole, Cooling
and SFR. To run a N-Body sim, use IC files with no Gas particles.

```
make
```
== Usage, parameter files ==
There are two example runs in run/

first make the code in source tree with Makefile.example

then:
    run.sh : simulation with gas
    run-dm.sh : simulation without gas (dm only)

== Initial Condition ==
There is an IC generator in GENIC directory. It compiles similarily. You need to
set SYSTYPE variable and create a Makefile.$(SYSTYPE) file.
IC generator also depends on pfft.

