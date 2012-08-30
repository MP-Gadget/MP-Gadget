#ifdef NUCLEAR_NETWORK

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "eos.h"
#include "network.h"
#include "network_solver.h"

#ifndef STANDALONE
#include "proto.h"
#define EXIT endrun(1338)
#else
#define EXIT exit(1338)
#endif

/* to use this network you need 5 files:
 * one file containing information about which species your network should use [species.txt]
 * one file containing the partition functions for the species [part.txt]
 *     you can download it e.g. from http://www.nscl.msu.edu/~nero/db/library.php?action=download | http://www.nscl.msu.edu/~nero/db/docs/part_frdm.asc
 * one file containing the reaction rates in the REACLIB 2 format (described in http://www.nscl.msu.edu/~nero/db/docs/reaclibFormat.pdf) [rates.txt]
 *     you can download it e.g. from http://www.nscl.msu.edu/~nero/db/library.php?action=download
 * one file containing the binding energies of the nuclei [masses.txt]
 *     you can download it e.g. from http://www.nndc.bnl.gov/masses/ | http://www.nndc.bnl.gov/masses/mass.mas03
 * one file containing weak reactions e.g. by langanke 2001
 */

void network_init( char *speciesfile, char *ratesfile, char *partfile, char *massesfile, char *weakratesfile ) {
  FILE *fd;
  char cdummy[200], cdummy2[200];
  int i, j, k, len, found, needed, rc;
  int na, nz, nn, nminz;
  float spin, exm;
  char nucnames[6][6], *dummy;
  int ratetype, nallocated;
  int nucids[6];
  int perm, mult, *count, *ratecount;
  int nMatrix, nMatrix2;
  char *masses, *spins, missing;
  char nucEduct[6], nucProduct[6];
  int nucEductId, nucProductId;
  int isFirst;
  
  /* 
   * 
   * read species data
   * 
   */
   
#ifndef STANDALONE
  if (ThisTask == 0) {
#endif
    if (!(fd = fopen(speciesfile, "r"))) {
      printf( "can't open file `%s' for reading species information.\n", speciesfile );
      EXIT;
    }
    fgets( cdummy, 200, fd );
    sscanf( cdummy, "%d", &network_data.nuc_count );
#ifndef STANDALONE
  }  
  MPI_Bcast( &network_data.nuc_count, 1, MPI_INT, 0, MPI_COMM_WORLD );
#endif

#ifndef STANDALONE
  if (network_data.nuc_count != EOS_NSPECIES) {
    if(ThisTask == 0) {
      printf( "Speciesfile contains %d species, but %d expected.\n", network_data.nuc_count, EOS_NSPECIES );
    }
    EXIT;
  }
#endif

  /* allocate memory for species data */
  network_nucdata = (struct network_nucdata*)malloc( network_data.nuc_count * sizeof( struct network_nucdata ) );

#ifndef STANDALONE
  if (ThisTask == 0) {
#endif    
    for ( i=0; i<network_data.nuc_count; i++ ) {
      fgets( cdummy, 200, fd );
      sscanf( cdummy, "%5c%d%d", network_nucdata[i].name, &network_nucdata[i].na, &network_nucdata[i].nz );
      network_nucdata[i].name[5] = 0;
      network_nucdata[i].nn = network_nucdata[i].na - network_nucdata[i].nz;
      network_nucdata[i].nrates = 0;
      network_nucdata[i].nweakrates = 0;
      network_nucdata[i].spin = 0.0;
      
      len = floor( log10( network_nucdata[i].na ) ) + 1;
      strncpy( network_nucdata[i].symbol, network_nucdata[i].name, strlen( network_nucdata[i].name )-len );
	}
    fclose( fd );
#ifndef STANDALONE
  }
#endif

  /* 
   * 
   * read mass excess
   * 
   */
  
#ifndef STANDALONE
  if (ThisTask == 0) {
#endif
    masses = (char*)malloc( network_data.nuc_count );
    memset( masses, 0, network_data.nuc_count );

    if (!(fd = fopen(massesfile, "r"))) {
      printf( "can't open file `%s' for reading mass excess.\n", massesfile );
      EXIT;
    }
    
    /* skip 39 lines */
    for (i=0; i<39; i++) {
    	fgets( cdummy, 200, fd );
    }
    
    while (!feof(fd)) {
    	fgets( cdummy, 200, fd );
    	sscanf( &cdummy[1], "%d%d%d%d\n", &nminz, &nn, &nz, &na );
    	
    	for (i=0; i<network_data.nuc_count; i++) {
    		if (network_nucdata[i].na == na && network_nucdata[i].nz == nz) {
    			sscanf( &cdummy[29], "%f", &exm );
    			network_nucdata[i].exm = exm;
    			masses[i] = 1;
    			break;
			}
		}
	}
    
    fclose( fd );
    
    missing = 0;
    for (i=0; i<network_data.nuc_count; i++) {
    	if (!masses[i]) {
    		printf( "Nucleus %s missing in mass file.\n", network_nucdata[i].name );
    		missing = 1;
		}
	}
	
	if (missing) {
		EXIT;
	}
    free( masses );
#ifndef STANDALONE
  }
#endif

  /* 
   * 
   * read partition functions
   * 
   */

#ifndef STANDALONE
  if (ThisTask == 0) {
#endif
    spins = (char*)malloc( network_data.nuc_count );
    memset( spins, 0, network_data.nuc_count );

    if (!(fd = fopen(partfile, "r"))) {
      printf( "can't open file `%s' for reading partition functions.\n", partfile );
      EXIT;
    }
    
    /* default value for the partition function is 1.0 => log(part) = 0.0 */
    for (i=0; i<network_data.nuc_count; i++) {
    	for (j=0; j<24; j++) {
    		network_nucdata[i].part[j] = 0.0;
    	}
    }

    /* skip 4 lines */
    for (i=0; i<4; i++) {
    	fgets( cdummy, 200, fd );
    }
    
    while (!feof(fd)) {
    	fgets( cdummy, 200, fd ); /* skip name of the nucleus */
    	fgets( cdummy, 200, fd );
    	sscanf( cdummy, "%d%d%f\n", &nz, &na, &spin );
    	
    	found = 0;
    	for (i=0; i<network_data.nuc_count; i++) {
    		if (network_nucdata[i].na == na && network_nucdata[i].nz == nz) {
    			found = 1;
    			
    			/* read partition function data */
    			for (j=0; j<3; j++) {
				char *tmp, *next_field;

    				fgets( cdummy, 200, fd );
				next_field = cdummy;
    				for (k=0; k<8; k++) {
					network_nucdata[i].part[j*8+k] = strtod(next_field, &tmp);
					if (tmp == next_field) {
						printf("Error reading partition function data for element %s\n", network_nucdata[i].name);
						EXIT;
					}
					next_field = tmp;
    					network_nucdata[i].part[j*8+k] = log( network_nucdata[i].part[j*8+k] );
    				}
    			}
    			
    			network_nucdata[i].spin = spin;
    			spins[i] = 1;
    			break;
			}
		}
		
		/* skip 3 lines containing partition function data */
		if (!found) {
			for (i=0; i<3; i++) {
				fgets( cdummy, 200, fd );
			}
		}
	}
    
    fclose( fd );
    
    for (i=0; i<network_data.nuc_count; i++) {
    	if (!spins[i]) {
    		printf( "There are no partition function and spin data for nucleus %s. Assuming spin 0 and constant partition function of 1.\n", network_nucdata[i].name );
		}
	}
    free( spins );
#ifndef STANDALONE
  }
#endif

  /* 
   * 
   * read rate data
   * 
   */
  
  count = (int*)malloc( network_data.nuc_count * sizeof(int) );
  
#ifndef STANDALONE
  if (ThisTask == 0) {
#endif
    network_data.rate_count = 0;
    nallocated = 100;
    network_rates = (struct network_rates*)malloc( nallocated * sizeof( struct network_rates ) );

    if (!(fd = fopen(ratesfile, "r"))) {
      printf( "can't open file `%s' for reading rate data.\n", ratesfile );
      EXIT;
    }
    
    while (!feof(fd)) {
    	fgets( cdummy, 200, fd );
    	
#ifdef REACLIB1
		if (isdigit(cdummy[0])) {
			sscanf( cdummy, "%d", &ratetype );
			
			/* skip 2 lines and get new line */
			for (i=0; i<3 && !feof(fd); i++) {
				fgets( cdummy, 200, fd );
			}
		}
#else
		/* in the REACLIB2 format the ratetype is given for each rate individually */
		sscanf( cdummy, "%d", &ratetype );
		fgets( cdummy, 200, fd );
#endif
    	for (i=0; i<6; i++) {
    		sscanf( &cdummy[(i+1)*5], "%5c", nucnames[i] );
    		nucnames[i][5] = 0;
    	}
    	
    	/* check if we need this rate. we need it if it only contains nuclei we consider */
    	needed = 1;
    	
    	for (i=0; i<6; i++) {
    		nucids[i] = -1;
    		found = 0;
    		if (strncmp(nucnames[i],"     ",5)) {
    			/* do not check one nuclei again */
    			for(j=0; j<i; j++) {
    				if (!strcmp(nucnames[i], nucnames[j])) {
    					nucids[i] = nucids[j];
    					found = 1;
    					break;
    				}
    			}
    			
    			/* if found, we do not have to check it again */
    			if (found) {
    				continue;
    			}
    			
    			/* if new nuclei, check table of all nuclei used */
    			for (j=0; j<network_data.nuc_count; j++) {
    				if (!strcmp(network_nucdata[j].name, nucnames[i])) {
    					nucids[i] = j;
    					found = 1;
    					break;
    				}
				}
				
				/* if not found, we do not need this rate */
				if (!found) {
					needed = 0;
					break;
				}
			}
    	}
    	
    	if (needed) {
    		rc = network_data.rate_count;
    		
    		/* check if array to store rates is large enough */
    		if (rc == nallocated) {
    			dummy = (char*)malloc( (nallocated+100) * sizeof( struct network_rates ) );
    			memcpy( dummy, network_rates, nallocated * sizeof( struct network_rates ) );
    			free( network_rates );
    			network_rates = (struct network_rates*)dummy;
    			nallocated += 100;
    		}
			
			switch(ratetype) {
				case 1:
					network_rates[rc].ninput = 1;
					network_rates[rc].noutput = 1;
					break;
				case 2:
				    network_rates[rc].ninput = 1;
					network_rates[rc].noutput = 2;
					break;
				case 3:
				    network_rates[rc].ninput = 1;
					network_rates[rc].noutput = 3;
					break;
				case 4:
				    network_rates[rc].ninput = 2;
					network_rates[rc].noutput = 1;
					break;
				case 5:
				    network_rates[rc].ninput = 2;
					network_rates[rc].noutput = 2;
					break;
				case 6:
				    network_rates[rc].ninput = 2;
					network_rates[rc].noutput = 3;
					break;
				case 7:
				    network_rates[rc].ninput = 2;
					network_rates[rc].noutput = 4;
					break;
				case 8:
				    network_rates[rc].ninput = 3;
				    if (nucids[4] == -1) {
						network_rates[rc].noutput = 1;
					} else {
						network_rates[rc].noutput = 2;
					}
					break;
				case 9:
				    network_rates[rc].ninput = 3;
					network_rates[rc].noutput = 2;
					break;
				case 10:
				    network_rates[rc].ninput = 4;
					network_rates[rc].noutput = 2;
					break;
				case 11:
				    network_rates[rc].ninput = 1;
					network_rates[rc].noutput = 4;
					break;
				default:
					printf( "Ratetype %d is not supported\n", ratetype );
					needed = 0;
					break;
			}
			
			for (i=0; i<network_rates[rc].ninput; i++) network_rates[rc].input[i] = nucids[i];
			for (i=0; i<network_rates[rc].noutput; i++) network_rates[rc].output[i] = nucids[i+network_rates[rc].ninput];
		}
		
        if (needed) {
         	rc = network_data.rate_count;
         	
			network_rates[rc].type = ratetype;
			network_rates[rc].isWeak = (cdummy[47] == 'w');
			network_rates[rc].isReverse = (cdummy[48] == 'v');
			network_rates[rc].isElectronCapture = (strcmp(&cdummy[43],"  ec") == 0) ? 1 : 0;
			network_rates[rc].isWeakExternal = 0;
			sscanf( &cdummy[52], "%f\n", &network_rates[rc].q );
			
			fgets( cdummy2, 200, fd );
			for (i=0; i<4; i++) {
				sscanf( &cdummy2[i*13], "%f", &network_rates[rc].data[i] );
			}
			fgets( cdummy2, 200, fd );
			for (i=4; i<7; i++) {
				sscanf( &cdummy2[(i-4)*13], "%f", &network_rates[rc].data[i] );
			}
			
			/* mark nuclei that are affected by this rate */
			memset( count, 0, network_data.nuc_count * sizeof(int) );
			
			for (i=0; i<6; i++) {
				if (nucids[i] != -1) {
					if ( (ratetype < 4 && i < 1) || (ratetype >= 4 && ratetype < 8 && i < 2) || (ratetype == 8 && i < 3) ) {
						count[ nucids[i] ]--;
					} else {
						count[ nucids[i] ]++;
					}
				}
			}
			
			/* nuclei are only affected if their number is really changed 
			 * this is i.e. not the case for catalysts like protons in the reaction 
			 * p + he4 + ru88 => p + pd92 */
			for (i=0; i<network_data.nuc_count; i++) {
				if (count[i] != 0)
					network_nucdata[i].nrates++;
			}
			
			network_data.rate_count++;
		} else {
			/* skip 2 lines */
			for (i=0; i<2; i++) {
				fgets( cdummy, 200, fd );
			}
		}
	}
    
    /* free memory that is not needed */
    dummy = (char*)malloc( network_data.rate_count * sizeof( struct network_rates ) );
    memcpy( dummy, network_rates, network_data.rate_count * sizeof( struct network_rates ) );
    free( network_rates );
    network_rates = (struct network_rates*)dummy;
    
    fclose( fd );
    
#ifndef STANDALONE
  }
#endif

#if 0
  double temp;
  double t9, t9l, temp9[7];
  double rate;

  temp = 1e10;
  t9 = temp * 1.0e-9;
  t9l = log( t9 );

  temp9[0] = 1.0;
  temp9[1] = 1.0 / t9;
  temp9[2] = exp( -t9l / 3.0 );
  temp9[3] = 1.0 / temp9[2];
  temp9[4] = t9;
  temp9[5] = exp( t9l * 5.0 / 3.0 );
  temp9[6] = t9l;
  
  fd = fopen( "bad_rates.txt", "w" );
  
  for (i=0; i<network_data.rate_count; i++) {
  	rate = 0;
  	for (j=0; j<7; j++) {
  		rate += temp9[j] * network_rates[i].data[j];
  	}
  	rate = exp( rate );
  	if (rate > 1e20) {
  		fprintf( fd, "%12.6e: %s", rate, network_rates[i].cdummy );
    }
  }
  
  fclose( fd );
  
  EXIT;
#endif

  /* 
   * 
   * read weak rate data
   * 
   */
   
#ifndef STANDALONE
  if (ThisTask == 0) {
#endif
    network_data.weakrate_count = 0;
    nallocated = 100;
    network_weakrates = (struct network_weakrates*)malloc( nallocated * sizeof( struct network_weakrates ) );

    if (!(fd = fopen(weakratesfile, "r"))) {
      printf( "can't open file `%s' for reading weak rate data.\n", weakratesfile );
      EXIT;
    }
    
    isFirst = 1;
    while (!feof(fd)) {
    	fgets( cdummy, 200, fd );
    	
    	memcpy( nucEduct, cdummy, 5 );
    	nucEduct[5] = 0;
    	memcpy( nucProduct, &cdummy[5], 5 );
    	nucProduct[5] = 0;
    	
    	nucEductId = -1;
    	nucProductId = -1;
    	
    	for (i=0; i<network_data.nuc_count; i++) {
    		if (!strcmp(nucEduct, network_nucdata[i].name)) {
    			nucEductId = i;
    			if (nucProductId >= 0)
    				break;
    		}
    		if (!strcmp(nucProduct, network_nucdata[i].name)) {
    			nucProductId = i;
    			if (nucEductId >= 0)
    				break;
    		}
    	}
    	
    	if (nucEductId >= 0 && nucProductId >= 0) {
    		/* we need this rate */
    		rc = network_data.weakrate_count;
    		
    		/* check if array to store rates is large enough */
    		if (rc == nallocated) {
    			dummy = (char*)malloc( (nallocated+100) * sizeof( struct network_weakrates ) );
    			memcpy( dummy, network_weakrates, nallocated * sizeof( struct network_weakrates ) );
    			free( network_weakrates );
    			network_weakrates = (struct network_weakrates*)dummy;
    			nallocated += 100;
    		}
    		
    		network_weakrates[rc].input = nucEductId;
    		network_weakrates[rc].output = nucProductId;
    		
    		sscanf( &cdummy[12], "%f", &network_weakrates[rc].q1 );
    		sscanf( &cdummy[24], "%f", &network_weakrates[rc].q2 );
    		sscanf( &cdummy[36], "%d", &network_weakrates[rc].isReverse );
			
			for (i=0; i<143; i++) {
				fgets( cdummy, 200, fd );
				sscanf( &cdummy[13], "%f", &network_weakrates[rc].lambda1[i] );
				sscanf( &cdummy[22], "%f", &network_weakrates[rc].lambda2[i] );
				
				if (isFirst) {
					if (i < 13) sscanf( &cdummy[0], "%f", &network_data.weakTemp[i] );
					if (i % 13 == 0) sscanf( &cdummy[6], "%f", &network_data.weakRhoYe[i/13] );
				}
			}
    		
    		network_nucdata[ nucEductId ].nweakrates++;
    		network_nucdata[ nucProductId ].nweakrates++;
    		
    		network_data.weakrate_count++;
    		
    		isFirst = 0;
		} else {
			/* skip rate data, 13*11 = 143 lines */
			for (i=0; i<143; i++)
				fgets( cdummy, 200, fd );
		}
	}
	
	/* free memory that is not needed */
    dummy = (char*)malloc( network_data.weakrate_count * sizeof( struct network_weakrates ) );
    memcpy( dummy, network_weakrates, network_data.weakrate_count * sizeof( struct network_weakrates ) );
    free( network_weakrates );
    network_weakrates = (struct network_weakrates*)dummy;
    
    fclose( fd );
    
    for (i=0; i<network_data.weakrate_count; i++) {
    	nucEductId = network_weakrates[i].input;
    	nucProductId = network_weakrates[i].output;
    	for (j=0; j<network_data.rate_count; j++) {
    		if (network_rates[j].type == 1 && network_rates[j].input[0] == nucEductId && network_rates[j].output[0] == nucProductId) {
    			network_rates[j].isWeakExternal = 1;
    		}
    	}
    }
#ifndef STANDALONE
  }
#endif

#ifndef STANDALONE
  /* distribute nuclear and rate date */
  MPI_Bcast( &network_data.rate_count, 1, MPI_INT, 0, MPI_COMM_WORLD );
  MPI_Bcast( &network_data.weakrate_count, 1, MPI_INT, 0, MPI_COMM_WORLD );
  
  if (ThisTask != 0) {
  	network_rates = (struct network_rates*)malloc( network_data.rate_count * sizeof( struct network_rates ) );
  	network_weakrates = (struct network_weakrates*)malloc( network_data.weakrate_count * sizeof( struct network_weakrates ) );
  }
  
  MPI_Bcast( network_nucdata, network_data.nuc_count * sizeof( struct network_nucdata ), MPI_BYTE, 0, MPI_COMM_WORLD );
  MPI_Bcast( network_rates, network_data.rate_count * sizeof( struct network_rates ), MPI_BYTE, 0, MPI_COMM_WORLD );
  MPI_Bcast( network_weakrates, network_data.weakrate_count * sizeof( struct network_weakrates ), MPI_BYTE, 0, MPI_COMM_WORLD );
#endif

  /* save rates that influence a species on its struct for faster access */
  for (i=0; i<network_data.nuc_count; i++) {
  	network_nucdata[i].rates = (int*)malloc( network_nucdata[i].nrates * sizeof(int) );
  	network_nucdata[i].prates = (double**)malloc( network_nucdata[i].nrates * sizeof(double*) );
  	network_nucdata[i].w = (double*)malloc( network_nucdata[i].nrates * sizeof(double) );
  	network_nucdata[i].weakrates = (int*)malloc( network_nucdata[i].nweakrates * sizeof(int) );
  	network_nucdata[i].pweakrates = (double**)malloc( network_nucdata[i].nweakrates * sizeof(double*) );
  	network_nucdata[i].wweak = (double*)malloc( network_nucdata[i].nweakrates * sizeof(double) );
  }
  
  network_data.yrate = (double*)malloc( network_data.rate_count * sizeof(double) );
  network_data.yweakrate = (double*)malloc( network_data.weakrate_count * sizeof(double) );
  
  ratecount = (int*)malloc( network_data.nuc_count * sizeof(int) );
  
  memset( ratecount, 0, network_data.nuc_count * sizeof(int) );  
  for (i=0; i<network_data.rate_count; i++) {
  	memset( count, 0, network_data.nuc_count * sizeof(int) );
  	for (j=0; j<network_rates[i].ninput; j++)
  		count[ network_rates[i].input[j] ]--;
  	for (j=0; j<network_rates[i].noutput; j++) {
  		count[ network_rates[i].output[j] ]++;
  	}
  	

	/* permutation factor that is important when more than one nucleus of a kind is involved */
	perm = 1;
	for (j=0; j<network_rates[i].ninput; j++) {
		mult = 1;
		for (k=0; k<network_rates[i].ninput; k++) {
			if (network_rates[i].input[j] == network_rates[i].input[k]) {
				if (k < j) {
					/* we already did this kind */
					break;
				} else if (k > j) {
					/* another one of this kind */
					mult++;
				}
			}
		}
		
		for (k=1; k<=mult; k++) {
			perm *= k;
		}
	}
	
	for (j=0; j<network_data.nuc_count; j++) {
		if (count[j] != 0) {
			rc = ratecount[j];
			if (rc == network_nucdata[j].nrates) {
				/* this should never happen */
				printf( "rate %d, %s: ratecount too low (%d <> %d), stopping.\n", i, network_nucdata[j].name, rc, network_nucdata[j].nrates );
				EXIT;
			}
			network_nucdata[j].rates[rc] = i;
			network_nucdata[j].prates[rc] = &network_data.yrate[ i ];
			network_nucdata[j].w[rc] = (double)count[j] / (double)perm;
			ratecount[j]++;
		}
	}
  }
  
  /* do the same for weak rates */
  memset( ratecount, 0, network_data.nuc_count * sizeof(int) );
  for (i=0; i<network_data.weakrate_count; i++) {
  	network_nucdata[ network_weakrates[i].input ].weakrates[ ratecount[ network_weakrates[i].input ] ] = i;
  	network_nucdata[ network_weakrates[i].input ].pweakrates[ ratecount[ network_weakrates[i].input ] ] = &network_data.yweakrate[ i ];
  	network_nucdata[ network_weakrates[i].input ].wweak[ ratecount[ network_weakrates[i].input ] ] = -1.0;
  	ratecount[ network_weakrates[i].input ]++;
  	
  	network_nucdata[ network_weakrates[i].output ].weakrates[ ratecount[ network_weakrates[i].output ] ] = i;
  	network_nucdata[ network_weakrates[i].output ].pweakrates[ ratecount[ network_weakrates[i].output ] ] = &network_data.yweakrate[ i ];
  	network_nucdata[ network_weakrates[i].output ].wweak[ ratecount[ network_weakrates[i].output ] ] = 1.0;
  	ratecount[ network_weakrates[i].output ]++;
  }
  
  free( ratecount );
  free( count );
  
  /* do electron capture rates externally */
  network_data.nElectronCaptureRates = 0;
  for (i=0; i<network_data.rate_count; i++) {
  	if (network_rates[i].isElectronCapture)
  		network_data.nElectronCaptureRates++;
  }
  
  network_data.electronCaptureRates = (int*)malloc( network_data.nElectronCaptureRates * sizeof( int ) );
  network_data.nElectronCaptureRates = 0;
  for (i=0; i<network_data.rate_count; i++) {
  	if (network_rates[i].isElectronCapture) {
  		network_data.electronCaptureRates[ network_data.nElectronCaptureRates ] = i;
  		network_data.nElectronCaptureRates++;
  	}
  }

#ifndef FIXED_TEMPERATURE
  nMatrix = network_data.nuc_count + 1;
#else
  nMatrix = network_data.nuc_count;
#endif
  nMatrix2 = nMatrix * nMatrix;

  /* allocate some memory that is needed to calculate rates */
  network_data.gg = (double*)malloc( network_data.nuc_count * sizeof(double) );
  network_data.x = (double*)malloc( network_data.nuc_count * sizeof(double) );
  network_data.y = (double*)malloc( nMatrix * sizeof(double) );
  network_data.conv = 1.602177e-12 * 1.0e3 * 6.0221367e23; /* eV2erg * 1.0e3 [keV] * avogadro */
  network_data.na = (double*)malloc( network_data.nuc_count * sizeof(double) );
  for (i=0; i<network_data.nuc_count; i++) { network_data.na[i] = network_nucdata[i].na; }
  network_data.deriv = (double*)malloc( nMatrix * sizeof(double) );
  network_data.nMatrix = nMatrix;
  network_data.nMatrix2 = nMatrix2;
#if defined(SUPERLU) || defined(SOLVER_PARDISO)
  network_data.jacob = (double*)malloc( nMatrix2 * sizeof(double) );
#endif

  /* start values */
  network_data.oldTemp = -1.0;
  network_data.oldRho = -1.0;
  network_data.oldYe = -1.0;

  /* initialize solver */
#if defined(SUPERLU) || defined(SOLVER_PARDISO)
  network_solver_init( &network_getrhs, &network_getjacobLU, 1e-6, nMatrix, network_data.nuc_count, network_data.na );
#else
  network_solver_init( &network_getrhs, &network_getjacob, 1e-6, nMatrix, network_data.nuc_count, network_data.na );
#endif

#ifndef STANDALONE
  if (ThisTask == 0) {
    printf( "Network init done.\n" );
  }
#else
    printf( "Network init done.\n" );
#endif
}

void network_normalize(double *x, double *e)
{
  double sum, xnew;
  int i;

  sum = 0;
  for(i = 0; i < network_data.nuc_count; i++)
    {
      sum += x[i];
    }

  if (e)
    {
      for(i = 0; i < network_data.nuc_count; i++)
	{
	  xnew = x[i] / sum;
	  *e -= (xnew - x[i]) * network_nucdata[i].exm * network_data.conv;
	  x[i] = xnew;
	}
    }
  else
    {
      for(i = 0; i < network_data.nuc_count; i++)
	x[i] /= sum;
    }
}

void network_integrate( double temp, double rho, double *x, double *dx, double dt, double *dedt ) {
	double *y;
	double sum;
	int i;

	if (dt == 0 || temp < 1e7) {
	  for (i=0; i<network_data.nuc_count; i++) dx[i] = 0;
	  *dedt = 0;
	  return;
	}
	
	/* calculate number densities */
	y = network_data.y;
	for (i=0; i<network_data.nuc_count; i++) {
		y[i] = x[i] / network_nucdata[i].na;
	}
	
#ifndef FIXED_TEMPERATURE	
	y[network_data.nuc_count] = temp;
#endif
	
	/* run network */
	network_solver_integrate( temp, rho, y, dt );
	
	/* normalise */
	sum = 0.0;
	for (i=0; i<network_data.nuc_count; i++) {
      if (y[i] > 1.0) y[i] = 1.0;
      if (y[i] < 1e-30) y[i] = 1e-30;
      sum += y[i] * network_nucdata[i].na;
    }
    for (i=0; i<network_data.nuc_count; i++) {
      y[i] /= sum;
    }
	
	/* calculate change of mass fractions and energy release */
	*dedt = 0;
	for (i=0; i<network_data.nuc_count; i++) {
	  dx[i] = ( y[i] * network_nucdata[i].na - x[i] ) / dt;
	  *dedt -= dx[i] / network_nucdata[i].na * network_nucdata[i].exm;
	}
	*dedt *= network_data.conv;
}

void network_getrhs( double temp, double rho, double *y, double *rhs ) {
	int iTemp;
	int i, j;
	double e, dedT, dy;
	double newTemp, p, dpdr;
	double ne, nn, ye;
	double tmp;
	double **prate;
	double *yy, *w, *wend, *yrate, *rhsiter;
	struct network_nucdata *nucdata, *nucend;
	struct network_rates *ratedata, *rateend;
 
 	nn = 0.0;
	ne = 0.0;
	yy = y;
	nucend = &network_nucdata[ network_data.nuc_count ];
	for (nucdata = network_nucdata; nucdata != nucend; nucdata++) {
		if (*yy < 1e-30) {
			*yy = 1e-30;
		} else if (*yy > 1.0) {
			*yy = 1.0;
		}
		nn += (*yy) * (*nucdata).na;
		ne += (*yy) * (*nucdata).nz;
    	
		yy++;
	}
	ye = ne / nn;

#ifndef FIXED_TEMPERATURE
	iTemp = network_data.nuc_count;
	y[iTemp] = max(1e7, min(y[iTemp], 1e10));

	temp = y[iTemp];
#endif

	network_getrates( temp, rho, ye );

	/* dy_i/dt */
	yrate = network_data.yrate;
	rateend = &network_rates[ network_data.rate_count ];
	for (ratedata = network_rates; ratedata != rateend; ratedata++) {
		*yrate = (*ratedata).rate * y[ (*ratedata).input[0] ];
		for (j=1; j<(*ratedata).ninput; j++)
			*yrate *= y[ (*ratedata).input[j] ];
		
		yrate++;
	}
	
	/* **prate holds the adresses of the values in yrate */
	rhsiter = rhs;
	nucend = &network_nucdata[ network_data.nuc_count ];
	for (nucdata = network_nucdata; nucdata != nucend; nucdata++) {
		tmp = 0.0;
		
		prate = (*nucdata).prates;
		w = (*nucdata).w;
		wend = w + (*nucdata).nrates;
		for (; w!=wend; w++) {
			tmp += (*w) * (**prate);
			prate++;			
#if 0			
			/* debug */
			if (deriv > 1e50) {
				printf( "%3d: rate %4d, type %d, deriv: %13.6e, flags: %d %d %d\n", i, rate, network_rates[rate].type, deriv, network_rates[rate].isReverse, network_rates[rate].isWeak, network_rates[rate].isElectronCapture );
			}
			
			if (isnan(rhs[i]) || isinf(rhs[i])) {
				printf( "rhs %d (%s): %g (%g %g: %g)\n", i, network_nucdata[i].name, rhs[i], network_nucdata[i].w[j], network_rates[rate].rate, deriv );
				for (k=0; k<network_rates[rate].ninput; k++) {
					printf( "y[ %s ]: %g\n", network_nucdata[ network_rates[rate].input[k] ].name, y[ network_rates[rate].input[k] ] );
				}
				EXIT;
			}
#endif
		}

		*rhsiter = tmp;
		rhsiter++;
	}
	
	rhsiter = rhs;
	nucend = &network_nucdata[ network_data.nuc_count ];
	for (nucdata = network_nucdata; nucdata != nucend; nucdata++) {
		prate = (*nucdata).pweakrates;
		w = (*nucdata).wweak;
		for (j=0; j<(*nucdata).nweakrates; j++) {
			*rhsiter += (*w) * (**prate) * y[ network_weakrates[ (*nucdata).weakrates[j] ].input ];
			w++;
			prate++;
		}
		
		rhsiter++;
	}

#ifndef FIXED_TEMPERATURE
	/* dT/dt = dT/dE * dE/dt + sum_i ( dT/dy_i * dy_i/dt )
	 * dE/dt = sum_i ( ebind_i * dy_i/dt ) */
	for (i=0; i<network_data.nuc_count; i++) { 
		network_data.x[i] = y[i] * network_nucdata[i].na;
	}
	eos_calc_tgiven( rho, network_data.x, temp, &e, &dedT );
	rhs[iTemp] = 0;
	for (i=0; i<network_data.nuc_count; i++) {
		rhs[iTemp] += rhs[i] * network_nucdata[i].exm;
	}
	rhs[iTemp] *= -network_data.conv / dedT;

#ifndef NEGLECT_DTDY_TERMS
	/* sum_i ( dT/dy_i * dy_i/dt ) */
	for (i=0; i<network_data.nuc_count; i++) {
		dy = max( NETWORK_DIFFVAR, y[i]*NETWORK_DIFFVAR );
		network_data.x[i] = (y[i]+dy) * network_nucdata[i].na;
		newTemp = temp;
		eos_calc_egiven( rho, network_data.x, e, &newTemp, &p, &dpdr );
		rhs[iTemp] += (newTemp-temp) / dy * rhs[i];
		network_data.x[i] = y[i] * network_nucdata[i].na;
	}	
#endif
#endif
}

void network_getjacob( double temp, double rho, double h, double *y, double *rhs, double *jacob ) {
	int iTemp, nMatrix;
	int i, j, k, l, rate;
	double dTemp, yold, dy;
	double *deriv, fac, tempderiv;
	double ne, nn, ye;

#ifndef FIXED_TEMPERATURE
	temp = y[network_data.nuc_count];
#endif

	nn = 0.0;
	ne = 0.0;
	for ( i=0; i<network_data.nuc_count; i++ ) {
    	nn += y[i] * network_nucdata[i].na;
    	ne += y[i] * network_nucdata[i].nz;
  	}
  	ye = ne / nn;
	
	network_getrates( temp, rho, ye );
	
#ifndef FIXED_TEMPERATURE
	nMatrix = network_data.nuc_count + 1;
#else
	nMatrix = network_data.nuc_count;
#endif
	deriv = network_data.deriv;
	
	/* dy_i/dy_j */
	for (i=0; i<network_data.nuc_count; i++) {
		/* do row by row */
		memset( deriv, 0, network_data.nuc_count * sizeof( double ) );
		
		for (j=0; j<network_nucdata[i].nrates; j++) {
			rate = network_nucdata[i].rates[j];
			fac = network_nucdata[i].w[j] * network_rates[rate].rate;
			
			for (k=0; k<network_rates[rate].ninput; k++) {
				tempderiv = fac;
				for(l=0; l<network_rates[rate].ninput; l++) {
					if (k!=l)
						tempderiv *= y[ network_rates[rate].input[l] ];
				}
				deriv[ network_rates[rate].input[k] ] += tempderiv;
			}
		}
		
		/* move entries into the matrix */
		for (j=0; j<network_data.nuc_count; j++) {
			jacob[ i*nMatrix + j ] = deriv[j];
		}
	}

#ifndef FIXED_TEMPERATURE
	iTemp = network_data.nuc_count;

	/* dy_i/dT & dT/dT */
	dTemp = max( fabs( temp ) * NETWORK_DIFFVAR, NETWORK_DIFFVAR );
	y[iTemp] = temp + dTemp;
	network_getrhs( temp + dTemp, rho, y, deriv );
	y[iTemp] = temp;
	for (i=0; i<nMatrix; i++) {
		jacob[ i*nMatrix + iTemp ] = ( deriv[i] - rhs[i] ) / dTemp;
	}

	/* dT/dy_i */
	for (i=0; i<network_data.nuc_count; i++) {
		yold = y[i];
		dy = max( fabs( yold ) * NETWORK_DIFFVAR, NETWORK_DIFFVAR );
		y[i] = yold + dy;
		network_getrhs( temp, rho, y, deriv );
		jacob[ iTemp*nMatrix + i ] = ( deriv[iTemp] - rhs[iTemp] ) / dy;
		y[i] = yold;
	}
#endif

	for ( i=0; i<nMatrix*nMatrix; i++ ) {
    	jacob[i] = - h * jacob[i];
		if (i % (nMatrix+1) == 0) jacob[i] = jacob[i] + 1.0;
	}
}

#if defined(SUPERLU) || defined(SOLVER_PARDISO)
void network_getjacobLU( double temp, double rho, double h, double *y, double *rhs, double *values, int *columns, int *rowstart, int *usedelements ) {
	int nMatrix, nMatrix2, nColumn, nRow, nMatrixElements, nuc_count;
	double *deriv, *deriviter, *derivend;
	double *yy, *w;
	int *rate, *rateend;
	double nn, ne, fac;
	struct network_nucdata *nucdata, *nucend;
	struct network_rates *ratedata;
	struct network_weakrates *weakratedata;
	
	nMatrix = network_data.nMatrix;
	nMatrix2 = network_data.nMatrix2;
	deriv = network_data.deriv;
	derivend = &deriv[ nMatrix ];
	nuc_count = network_data.nuc_count;

#ifndef FIXED_TEMPERATURE
	temp = y[network_data.nuc_count];
#endif

    nn = 0.0;
	ne = 0.0;
	yy = y;
	nucend = &network_nucdata[ network_data.nuc_count ];
	for (nucdata = network_nucdata; nucdata != nucend; nucdata++) {
    	nn += (*yy) * (*nucdata).na;
    	ne += (*yy) * (*nucdata).nz;
    	yy++;
  	}
	
	network_getrates( temp, rho, ne / nn );
	
	nMatrixElements = 0;
	nRow = 0;
	
	/* dy_i/dy_j */
	for (nucdata = network_nucdata; nucdata != nucend; nucdata++) {
		/* do row by row */
		memset( deriv, 0, nuc_count * sizeof( double ) );
		
		w = (*nucdata).w;
		rateend = &(*nucdata).rates[ (*nucdata).nrates ];
		for (rate=(*nucdata).rates; rate!=rateend; rate++) {
			ratedata = &network_rates[ (*rate) ];
			fac = (*w) * (*ratedata).rate;
			
			switch ((*ratedata).ninput) {
				case 1:
					deriv[ (*ratedata).input[0] ] += fac;
					break;
				case 2:
					deriv[ (*ratedata).input[0] ] += fac * y[ (*ratedata).input[1] ];
					deriv[ (*ratedata).input[1] ] += fac * y[ (*ratedata).input[0] ];
					break;
				case 3:
					deriv[ (*ratedata).input[0] ] += fac * y[ (*ratedata).input[1] ] * y[ (*ratedata).input[2] ];
					deriv[ (*ratedata).input[1] ] += fac * y[ (*ratedata).input[0] ] * y[ (*ratedata).input[2] ];
					deriv[ (*ratedata).input[2] ] += fac * y[ (*ratedata).input[0] ] * y[ (*ratedata).input[1] ];
					break;
				case 4:
				  	fprintf(stderr, "this should not happen with Basel library-- Ruediger\n");
					deriv[ (*ratedata).input[0] ] += fac * y[ (*ratedata).input[1] ] * y[ (*ratedata).input[2] ] * y[ (*ratedata).input[3] ];
					deriv[ (*ratedata).input[1] ] += fac * y[ (*ratedata).input[0] ] * y[ (*ratedata).input[2] ] * y[ (*ratedata).input[3] ];
					deriv[ (*ratedata).input[2] ] += fac * y[ (*ratedata).input[0] ] * y[ (*ratedata).input[1] ] * y[ (*ratedata).input[3] ];
					deriv[ (*ratedata).input[3] ] += fac * y[ (*ratedata).input[0] ] * y[ (*ratedata).input[1] ] * y[ (*ratedata).input[2] ];
					break;
			}
			
			w++;
		}
		
		w = (*nucdata).wweak;
		rateend = &(*nucdata).weakrates[ (*nucdata).nweakrates ];
		for (rate=(*nucdata).weakrates; rate!=rateend; rate++) {
			weakratedata = &network_weakrates[ (*rate) ];
			deriv[ (*weakratedata).input ] += (*w) * (*weakratedata).rate;
			
			w++;
		}
		
		*rowstart = nMatrixElements;
		rowstart++;
		
		nColumn = 0;
		for (deriviter=deriv; deriviter!=derivend; deriviter++) {
			if ((*deriviter) != 0.0 || nColumn == nRow) {
				*values = - (*deriviter) * h;
				if (nColumn == nRow)
					*values += 1.0;
				values++;
				*columns = nColumn;
				columns++;
				nMatrixElements++;
			}
			nColumn++;
		}
		
		nRow++;
	}
	
	*rowstart = nMatrixElements;
	*usedelements = nMatrixElements;
}
#endif

void network_part( double temp ) {
	/* interpolates partition functions, given the temperature */
	int index, i;
	double tempLeft, tempRight;
	double dlgLeft, dlgRight;
	double grad;
	
	index = 0;
	temp = min( max( temp, network_parttemp[0] ), network_parttemp[23] );
	
	while (temp > network_parttemp[index]) {
		index++;
	}
	if (index > 0)
		index--;
	
	tempLeft  = network_parttemp[index];
	tempRight = network_parttemp[index+1];
    
    for (i=0; i<network_data.nuc_count; i++) {
    	dlgLeft = network_nucdata[i].part[index];
    	dlgRight = network_nucdata[i].part[index+1];
    	
    	grad = (dlgRight-dlgLeft) / (tempRight-tempLeft);
    	network_data.gg[i] = exp( dlgLeft + (temp - tempLeft)*grad );
    }
}

void network_getrates( double temp, double rho, double ye ) {
	double t9, t9l, temp9[7];
	int i, j, changed;
	double baserate;
	double logrhoye;
	int iTempLow, iTempHigh, iRhoYeLow, iRhoYeHigh;
	int idx1, idx2, idx3, idx4;
	double dt, drhoye, at, arhoye, b1, b2, rate1, rate2, rate;
	
	/* do not calculate rates again, if nothing changed */
	if (network_data.oldTemp == temp && network_data.oldRho == rho && network_data.oldYe == ye) {
		return;
	}
	
	changed = 0;
	
	if (network_data.oldTemp != temp) {
		network_part( temp );
	
		t9 = temp * 1.0e-9;
		t9l = log( t9 );
		
		temp9[0] = 1.0;
		temp9[1] = 1.0 / t9;
		temp9[2] = exp( -t9l / 3.0 );
		temp9[3] = 1.0 / temp9[2];
		temp9[4] = t9;
		temp9[5] = exp( t9l * 5.0 / 3.0 );
		temp9[6] = t9l;
		
		for (i=0; i<network_data.rate_count; i++) {
			baserate = 0.0;
			for (j=0; j<7; j++) {
				baserate += temp9[j] * network_rates[i].data[j];
			}
			network_rates[i].baserate = exp( baserate );
			
			/* account for reverse rate */
			if (network_rates[i].isReverse) {
				/* divide by input and multiply with output
				 * only normalised temperature dependent partition functions are used here
				 * as the groundstate is already part of the rate parametrization (included in a0) */
				for (j=0; j<network_rates[i].ninput; j++) {
					network_rates[i].baserate /= network_data.gg[ network_rates[i].input[j] ];
				}
				for (j=0; j<network_rates[i].noutput; j++) {
					network_rates[i].baserate *= network_data.gg[ network_rates[i].output[j] ];
				}
			}
			
			/* account for weak rate */
			if (network_rates[i].isWeakExternal) {
				network_rates[i].baserate = 0;
			}
			
#if DEBUG				
			/* debug */
			if (isnan(network_rates[i].baserate) || isinf(network_rates[i].baserate)) {
				printf( "undefined rate %d: %g %g\n", i, baserate, network_rates[i].baserate );
				for (j=0; j<network_rates[i].ninput; j++)
					printf( "%s ", network_nucdata[ network_rates[i].input[j] ].name );
				printf( " --> " );
				for (j=0; j<network_rates[i].noutput; j++)
					printf( "%s ", network_nucdata[ network_rates[i].output[j] ].name );
				printf( "\n" );
				/* network_rates[i].baserate = 0.0; */
				printf( "baserate is nan or inf\n");
				EXIT;
			}
#endif
		}
		
		network_data.oldTemp = temp;
		changed = 1;
	}
	
	if (network_data.oldRho != rho || changed) {
		for (i=0; i<network_data.rate_count; i++) {
			network_rates[i].rate = network_rates[i].baserate;
			/* multiply with rho (n-1)-times for n-body rate */
			for (j=0; j<network_rates[i].ninput-1; j++)
				network_rates[i].rate *= rho;
		}
		
		network_data.oldRho = rho;
		changed = 1;
	}
	
	if (network_data.oldYe != ye || changed) {
		for (i=0; i<network_data.nElectronCaptureRates; i++)
			network_rates[ network_data.electronCaptureRates[i] ].rate = network_rates[ network_data.electronCaptureRates[i] ].baserate * rho * ye;
		
		network_data.oldYe = ye;
		changed = 1;
	}
	
	/* if something has changed, recalculate weak rates */
	if (changed) {
		t9 = temp * 1e-9;
		iTempLow = 0;
		while ( t9 > network_data.weakTemp[iTempLow] )
			iTempLow++;
		
		if (iTempLow > 0)
			iTempLow--;
		
		iTempHigh = min( iTempLow+1, 12 );
		
		logrhoye = log10( rho * ye );
		iRhoYeLow = 0;
		while ( logrhoye > network_data.weakRhoYe[iRhoYeLow] )
			iRhoYeLow++;
		
		if (iRhoYeLow > 0)
			iRhoYeLow--;
		
		iRhoYeHigh = min( iRhoYeLow+1, 10 );
		
		idx1 = iRhoYeLow*13 + iTempLow;
		idx2 = iRhoYeHigh*13 + iTempLow;
		idx3 = iRhoYeLow*13 + iTempHigh;
		idx4 = iRhoYeHigh*13 + iTempHigh;
		
		dt = t9 - network_data.weakTemp[iTempLow];
		drhoye = logrhoye - network_data.weakRhoYe[iRhoYeLow];
		at = network_data.weakTemp[iTempHigh] - network_data.weakTemp[iTempLow];
		arhoye = network_data.weakRhoYe[iRhoYeHigh] - network_data.weakRhoYe[iRhoYeLow];
		
		for (i=0; i<network_data.weakrate_count; i++) {
			b1 = network_weakrates[i].lambda1[idx1] + (network_weakrates[i].lambda1[idx3] - network_weakrates[i].lambda1[idx1]) * dt / at;
			b2 = network_weakrates[i].lambda1[idx2] + (network_weakrates[i].lambda1[idx4] - network_weakrates[i].lambda1[idx2]) * dt / at;
			rate1 = b1 + (b2-b1) * drhoye / arhoye;
			b1 = network_weakrates[i].lambda2[idx1] + (network_weakrates[i].lambda2[idx3] - network_weakrates[i].lambda2[idx1]) * dt / at;
			b2 = network_weakrates[i].lambda2[idx2] + (network_weakrates[i].lambda2[idx4] - network_weakrates[i].lambda2[idx2]) * dt / at;
			rate2 = b1 + (b2-b1) * drhoye / arhoye;
			rate = exp( log(10) * rate1 ) + exp( log(10) * rate2 );
			
			network_weakrates[i].rate = rate;
			network_data.yweakrate[i] = rate;
		}
	}
}

#endif /* NUCLEAR_NETWORK */
