#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <libimf_vars.h>

#define Chabrier_MC 0.079
#define Chabrier_Sigma2 (0.69*0.69)

double Chabrier_Exp(double);
double Chabrier_byNum(double, void *);
double Chabrier_byMass(double, void *);
double Chabrier_byEgy(double, void *);

gsl_integration_workspace *my_w;

int main(int argc, char **argv)
{
  char *IMFfilename;
  double inf, sup, A;
  int i, j;

  if(argc < 4)
    {
      printf("arguments: IMF(s) file name\n"
             "           inf mass limit\n"
             "           sup mass limit\n");
      return -1;
    }

  my_w = gsl_integration_workspace_alloc(limf_gsl_intspace_dim);

  IMFfilename = (char*)malloc(strlen(*(argv+1)) + 2);
  sprintf(IMFfilename, "%s", *(argv+1));

  inf = atof(*(argv+2));
  sup = atof(*(argv+3));

  initialize_externalIMFs(1);
  set_externalIMF(0, "Chabrier", &Chabrier_byMass, &Chabrier_byNum, 0x0);

  read_imfs(IMFfilename);
  IMFp = &IMFs[0];

  printf("\n");

  for(j = 0; j < IMFs_dim; j++)
    {
      printf("------------------------------------------------------\n");
      printf("IMF :: %s :: ", IMFs[j].name);

      A = IntegrateIMF_byMass(inf, sup, &IMFs[j], INC_BH);
      if( fabs(A-1)/A > IMFs[j].Mm/1000)
        {
          printf("renormalizing.. (%g) ", fabs(A-1)/A);
          A = Renormalize_IMF(&IMFs[j], inf, sup, INC_BH);
        }
      printf("\n\n");
      if(fabs(inf - IMFs[j].Mm)/inf > 1e-2)
        printf("  >> warning : inf normalization limit is different than inf mass of IMF\n");
      if(fabs(sup - IMFs[j].MU)/sup > 1e-2)
        printf("  >> warning : sup normalization limit is different than sup mass of IMF\n");

      for(i = IMFs[j].NSlopes-1; i>=0; i--)
        printf("\tmass range :: [%.2f -> %.2f]\n\t\tnormalization :: %.7f\n", IMFs[j].Slopes.masses[i], (i>0)?IMFs[j].Slopes.masses[i-1]:IMFs[j].MU, IMFs[j].A[i]);
      
      printf("\nnormalization now reads :: %g\n\n", 1.0/A);
    }
  
  for(j = 0; j < IMFs_dim; j++)
    print_IMF(j, IMFfilename);
  return 0;
}


double Chabrier_Exp(double arg)
{
  return exp(-(log10(arg) - log10(Chabrier_MC)) * (log10(arg) - log10(Chabrier_MC))) / (2 * Chabrier_Sigma2);
}
double Chabrier_byMass(double arg, void *param)
{

  if(arg > 1)
    return IMFp->A[0] * pow(arg, -IMFp->Slopes.slopes[0]);
  else
    return IMFp->A[1] * Chabrier_Exp(arg);
}

double Chabrier_byNum(double arg, void *param)
{

  if(arg > 1)
    return IMFp->A[0] * pow(arg, -(1 + IMFp->Slopes.slopes[0]));
  else
    return IMFp->A[1] * Chabrier_Exp(arg);
}

