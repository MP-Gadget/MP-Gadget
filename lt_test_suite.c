
/*-------------------------------- */
/*
  to be added in main.c right before the call to run()
*/

/* #ifdef LT_STELLAREVOLUTION */
/*   int i; */
/*   if(All.TestSuite) */
/*     { */
/*       i = get_random_number(100) * NTask; */
/*       if(i == ThisTask) */
/*         test_suite(); */

/*       MPI_Barrier(MPI_COMM_WORLD); */

/*       endrun(7777); */
/*     } */
/* #endif */


#if defined(LT_METAL_COOLING)
void test_metal_cooling()
{
  int i, j;

  double logT, logZ, rate;

  FILE *testfile;

  testfile = fopen("MetalCoolingRates.dat", "w");

  fprintf(testfile, "# metallicity (log solar), temperature (log K), rate (erg/sec*cm^3)\n#\n");
  for(i = 1; i < ZBins; i++)
    {
      logZ = (CoolZvalue[i - 1] + CoolZvalue[i]) / 2;
      for(j = 1; j < TBins; j++)
	{
	  logT = (CoolTvalue[j - 1] + CoolTvalue[j]) / 2;
	  rate = GetMetalLambda(logT, logZ);
	  fprintf(testfile, "%4.3e %5.4e %g\n", logZ, logT, rate);
	}
      fprintf(testfile, "#\n#\n");
    }
  fclose(testfile);
  return;
}
#endif



int test_suite()
{
  int test_result = 0, singletest_result;

  int i;

  double test1, test2;

  FILE *tfile, *file;


  tfile = fopen("TESTS.txt", "w");

  fprintf(tfile,
	  "\n-----------------------------------------------------------\n"
	  "TESTS made by processor %d\n\n", ThisTask);


  /* ----------------------------------------------------------------------------
   * test 1
   * test the cooling: this produces an ascii file, MetalCoolingRates.dat that
   * contains the cooling functions for temperature and metallcity values that
   * are within the table boundaries but not exactly equal to the tabulated values.
   * look in test_metal_cooling() for details about the file format.
   */
#if defined(LT_METAL_COOLING)
  fprintf(tfile, "test metal cooling rates.. ");
  test_metal_cooling();
  fprintf(tfile, "done (tables in MetalCoolingRates.dat)\n");
#endif


  /* ----------------------------------------------------------------------------
   * test 2
   * test all the IMF-related functions
   */
  /* this test should be made outside the code */




  /* ----------------------------------------------------------------------------
   * test 3
   * test the chemical evolution calculations
   */

  fprintf(tfile, "test chemical evolution.. ");
  TestStellarEvolution();
  fprintf(tfile, "done (tables in SE.dbg)\n");


  /* ----------------------------------------------------------------------------
   * test 4
   * test the chemical time-stepping
   */

  /* to be done */


  /* ----------------------------------------------------------------------------
   * test 5
   * test the lifetime funcion and its inverse
   */

  fprintf(tfile, "test the lifetime function for 100 ranodm values.. ");
  for(i = 0; i < 100; i++)
    {
      test1 = All.mean_lifetime + (All.sup_lifetime - All.inf_lifetime) * get_random_number(i * 10);
      test2 = lifetime(dying_mass(test1));

      if(test2 <= 0)
	{
	  printf(">> WARN :: strange result for lifetime(dying_mass(T)):: T = %8.6e, res = %8.6e\n",
		 test1, test2);
	  singletest_result = 1;
	}
      else
	{
	  if(fabs(test2 - test1) / test2 > 1e-3)
	    {
	      printf(">> WARN :: inaccurate lifetime / lifetime^-1 for T = %8.6e"
		     "           lifetime^-1(T)           = %8.6e\n"
		     "           lifetime(lifetime^-1(T)) = %8.6e\n", test1, dying_mass(test1), test2);
	      singletest_result = 1;
	    }
	}
    }


  file = fopen("lifetime.txt", "w");
  test1 = log10(All.sup_lifetime / All.inf_lifetime) / 100;
  fprintf(file, "# time (Gyr) , dying_mass(time), lifetime(dying_amss(time)), dm(time)/dt\n");
  for(i = 0; i < 100; i++)
    {
      test2 = All.inf_lifetime * pow(10, test1 * i);
      fprintf(file, "%8.6e\t%8.6e\t%8.6e\t%8.6e\n",
	      test2, dying_mass(test2), lifetime(dying_mass(test2)), dm_dt(dying_mass(test2), test2));
    }
  fclose(file);

  test_result += singletest_result;
  fprintf(tfile, "done\n");



  fclose(tfile);
  return test_result;
}
