#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "proto.h"
#include "allvars.h"


#ifdef SCFPOTENTIAL



/* random number of unit sphere Gnml coefficient */
MyDouble SCF_Gnml_rand(int n, int l, int m, long *seed)
{
 if (n+l+m>0)
  return 1./sqrt(1.*SCF_NEFF)*sqrt(gnlm_var(n,l,m))*gasdev(seed);
 else
  return -1.+ 1./sqrt(1.*SCF_NEFF)*sqrt(gnlm_var(n,l,m))*gasdev(seed);
}

/* random number of unit sphere Hnml coefficient */
MyDouble SCF_Hnml_rand(int n, int l, int m, long *seed)
{
 return 1./sqrt(1.*SCF_NEFF)*sqrt(hnlm_var(n,l,m))*gasdev(seed);
}

/* 
 derive series expansion from random numbers.
 called every timestep, updating the seed
*/
void SCF_calc_from_random(long *seed)
{
  int n, l, m;
  for  (l=0; l<=SCF_LMAX; l++) 
    for  (m=0; m<=l; m++) 
      for  (n=0; n<=SCF_NMAX; n++) 
       {
        sinsum[nlm(n,l,m)]=SCF_Hnml_rand(n,l,m,seed);
	cossum[nlm(n,l,m)]=SCF_Gnml_rand(n,l,m,seed); 
       }
}

/* SCF init routine */
void SCF_init(void)
{
 int l, n, m;
 MyDouble K_nl, deltam0;
 Anltilde=(MyDouble*)malloc((SCF_NMAX+1)*(SCF_NMAX+1)*sizeof(MyDouble));
 coeflm=(MyDouble*)malloc((SCF_NMAX+1)*(SCF_NMAX+1)*sizeof(MyDouble));
 cosmphi=(MyDouble*)malloc((SCF_LMAX+1)*sizeof(MyDouble));
 sinmphi=(MyDouble*)malloc((SCF_LMAX+1)*sizeof(MyDouble));
 ultrasp=(MyDouble*)malloc((SCF_NMAX+1)*(SCF_LMAX+1)*sizeof(MyDouble));
 ultraspt=(MyDouble*)malloc((SCF_NMAX+1)*(SCF_LMAX+1)*sizeof(MyDouble));
 ultrasp1=(MyDouble*)malloc((SCF_NMAX+1)*(SCF_LMAX+1)*sizeof(MyDouble));
 plm=(MyDouble*)malloc((SCF_LMAX+1)*(SCF_LMAX+1)*sizeof(MyDouble));
 dplm=(MyDouble*)malloc((SCF_LMAX+1)*(SCF_LMAX+1)*sizeof(MyDouble));
 dblfact=(MyDouble*)malloc((SCF_LMAX+1)*sizeof(MyDouble));
 sinsum=(MyDouble*)malloc((SCF_NMAX+1)*(SCF_LMAX+1)*(SCF_LMAX+1)*sizeof(MyDouble));
 cossum=(MyDouble*)malloc((SCF_NMAX+1)*(SCF_LMAX+1)*(SCF_LMAX+1)*sizeof(MyDouble));

#ifdef SCF_HYBRID
 sinsum_all=(MyDouble*)malloc((SCF_NMAX+1)*(SCF_LMAX+1)*(SCF_LMAX+1)*sizeof(MyDouble));
 cossum_all=(MyDouble*)malloc((SCF_NMAX+1)*(SCF_LMAX+1)*(SCF_LMAX+1)*sizeof(MyDouble));

 int i;
 /* store masses in backup field */
 for (i = 0; i < NumPart; i++)
  P[i].MassBackup = P[i].Mass;
#endif

 Anltilde[0]=0.0;
 coeflm[0]=0.0;
 cosmphi[0]=0.0;
 sinmphi[0]=0.0;
 ultrasp[0]=0.0;
 ultraspt[0]=0.0;
 plm[0]=0.0;
 ultrasp1[0]=0.0;
 dplm[0]=0.0;
 dblfact[1]=1.; 

 /* set initial random number seed (global variable, so that all processors generate the same potential */
 scf_seed=42;
 
 for (l=2; l<=SCF_LMAX; l++) 
   dblfact[l]=dblfact[l-1]*(2.*l-1.);
 
 for (n=0; n<=SCF_NMAX; n++) 
   for (l=0; l<=SCF_LMAX; l++)
     Anltilde[nl(n,l)]=0.0;
  
 for (l=0; l<=SCF_LMAX; l++) 
   for (m=0; m<=l; m++) 
     coeflm[lm(l,m)]=0.0;

 for (n=0; n<=SCF_NMAX; n++) 
   for (l=0; l<=SCF_LMAX; l++) 
     {
      K_nl = 0.5*n*(n+4.*l+3.)+(l+1.)*(2.*l+1.); /* eq.(2.23) */
      Anltilde[nl(n,l)]=-pow(2.,8.*l+6.)*factrl(n)*(n+2.*l+1.5)*tgamma(2.*l+1.5)*tgamma(2.*l+1.5)/(4.*M_PI*K_nl*factrl(n+4*l+2)); 
     }
 
 for (l=0; l<=SCF_LMAX; l++) 
  {
   for (m=0; m<=l; ++m) 
    {
     deltam0=2.;
     if (m==0) deltam0=1.;
     coeflm[lm(l,m)]=(2.*l+1.)*deltam0*factrl(l-m)/factrl(l+m); /* N_lm   eq. (3.15) */
    }
  }
}

/* set the expansion factors back to zero */
void SCF_reset(void)
{
  int n, l, m;
  for  (l=0; l<=SCF_LMAX; l++) 
    for  (m=0; m<=l; m++) 
      for  (n=0; n<=SCF_NMAX; n++) 
       {
        sinsum[nlm(n,l,m)]=0.0;
	cossum[nlm(n,l,m)]=0.0; 
       }
}

/* SCF free routine */
void SCF_free(void)
{
 free(Anltilde);
 free(coeflm);
 free(cosmphi);
 free(sinmphi);
 free(ultrasp);
 free(ultraspt);
 free(ultrasp1);
 free(plm);
 free(dplm);
 free(dblfact);
 free(sinsum);
 free(cossum);

#ifdef SCF_HYBRID
 free(sinsum_all);
 free(cossum_all);
#endif
}

/* get force and potential based on SCF expansion of (unit!) sphere */
void SCF_evaluate(MyDouble x, MyDouble y, MyDouble z, MyDouble *potential, MyDouble *ax, MyDouble *ay, MyDouble *az)
{
 int n,l,m;
 MyDouble r,rl, costh,sinth,phi,xi;
 MyDouble ar,ath,aphi,poten;
 MyDouble un,unm1,unplusr,phinltil;
 MyDouble plm1m,plm2m;
 MyDouble temp1,temp2,temp3,temp4;
 MyDouble Clm,Dlm,Elm,Flm; 


 /* particle coordinate transformation */
 r = sqrt(x * x + y * y + z * z);    
 costh=z/r;
 phi=atan2(y,x);  
 xi=(r-1.)/(r+1.);
 sinth=sqrt(1.-costh*costh);
    

  for (m=0; m<=SCF_LMAX; m++) 
   {
     cosmphi[m]=cos(m*phi);           
     sinmphi[m]=sin(m*phi); 
    }

   ar=0.0;
   ath=0.0;
   aphi=0.0;
   poten=0.0; 

   /* Ultraspherical polynomials */
   for (l=0; l<=SCF_LMAX; l++) 
    {
     ultrasp[nl(0,l)]=1.0;
     ultrasp[nl(1,l)]=(2.0*(2.*l+1.5))*xi;
     ultrasp1[nl(0,l)]=0.0;
     ultrasp1[nl(1,l)]=1.0;
     un=ultrasp[nl(1,l)];
     unm1=1.0;
     /* recursion */
     for(n=1; n<=SCF_NMAX-1; n++)
      {
       ultrasp[nl(n+1,l)]=((2.0*n+(2.0*(2.*l+1.5)))*xi*un-(1.0*n-1.0+(2.0*(2.*l+1.5)))*unm1)*1.0/(n+1.0);
       unm1=un;
       un=ultrasp[nl(n+1,l)];
       ultrasp1[nl(n+1,l)]=(((2.0*(2.*l+1.5))+(n+1)-1.)*unm1-(n+1)*xi*ultrasp[nl(n+1,l)])/((2.0*(2.*l+1.5))*(1.-xi*xi));        
      }
    }

   /* Legendre polynomials */   
   for (m=0; m<=SCF_LMAX; m++) 
    {
     plm[lm(m,m)]=1.0;
     if(m>0) 
      plm[lm(m,m)]=pow(-1,m)*dblfact[m]*pow(sinth,m);
     plm1m=plm[lm(m,m)];
     plm2m=0.0;
     /* recursion */
     for(l=m+1; l<=SCF_LMAX; l++)
      {
       plm[lm(l,m)]=(costh*(2.*l-1.)*plm1m-(l+m-1.)*plm2m)/(l-m);
       plm2m=plm1m;
       plm1m=plm[lm(l,m)]; 
     }
    }

   /* derivatives of Legendre polynomials */	
   dplm[0]=0.0;
   for(l=1; l<=SCF_LMAX; l++)
    {
     for(m=0; m<=l; m++)
      {
       if(l==m) 
        dplm[lm(l,m)]=l*costh*plm[lm(l,m)]/(costh*costh-1.0);
       else 
        dplm[lm(l,m)]=(l*costh*plm[lm(l,m)]-(l+m)*plm[lm(l-1,m)])/(costh*costh-1.0);
      }         
    } 

   for (l=0; l<=SCF_LMAX; l++) 
    {
     temp1=0.0;
     temp2=0.0;
     temp3=0.0;
     temp4=0.0; 
     for(m=0; m<=l; m++)
      {
       Clm=0.0;
       Dlm=0.0;
       Elm=0.0;
       Flm=0.0;	 
       for(n=0; n<=SCF_NMAX; n++)
        {
           Clm += ultrasp[nl(n,l)]*cossum[nlm(n,l,m)];
           Dlm += ultrasp[nl(n,l)]*sinsum[nlm(n,l,m)];
           Elm += ultrasp1[nl(n,l)]*cossum[nlm(n,l,m)];
           Flm += ultrasp1[nl(n,l)]*sinsum[nlm(n,l,m)];
        }
       temp1 += plm[lm(l,m)]*(Clm*cosmphi[m]+Dlm*sinmphi[m]);
       temp2 += -plm[lm(l,m)]*(Elm*cosmphi[m]+Flm*sinmphi[m]);
       temp3 += -dplm[lm(l,m)]*(Clm*cosmphi[m]+Dlm*sinmphi[m]);
       temp4 += -m*plm[lm(l,m)]*(Dlm*cosmphi[m]-Clm*sinmphi[m]);
      } 
      
      
   rl=pow(r,l);
   unplusr=pow(1.+r,2*l+1); 
   phinltil=rl/unplusr;
       
   /* add contributions to potential */    
   poten += temp1*phinltil;

   /* add contributions to accelerations in spherical coordinates */
   ar  += phinltil*(-temp1*(l/r-(2.*l+1.)/(1.+r))+temp2*4.*(2.*l+1.5)/(pow(1.+r,2))); 
   ath += temp3*phinltil;
   aphi += temp4*phinltil;   	 
  }
  

  /* convert to cartesian coordinates */
  *ax=(sinth*cos(phi)*ar+costh*cos(phi)*ath-sin(phi)*aphi);
  *ay=(sinth*sin(phi)*ar+costh*sin(phi)*ath+cos(phi)*aphi);
  *az=(costh*ar-sinth*ath); 
  *potential = poten;
}




#ifdef SCF_HYBRID
static int SCF_n_type[6];
static long long SCF_ntot_type_all[6];

/* reshift positions to center of mass */
void SCF_do_center_of_mass_correction(double fac_rad, double start_rad, double fac_part, int max_iter)
{
 int i, k, n;
 MyDouble pos[3], pos_all[3];
 int num, num_all;
 int iter=0;
 double rad, max_rad=start_rad;


 if (ThisTask==0)
  printf("SCF center of mass correction...\n");

 
 for(n = 0; n < 6; n++)
   SCF_n_type[n] = 0;

 for(n = 0; n < NumPart; n++)
   SCF_n_type[P[n].Type]++;

 sumup_large_ints(6, SCF_n_type, SCF_ntot_type_all);
 
 if (ThisTask==0)
  printf("total number of DM particles=%lld\n", SCF_ntot_type_all[1]);
 
 
 
 while(iter < max_iter)
  {
   num=0;
   for(k = 0; k < 3; k++)
    {
     pos[k]=0.0;
     pos_all[k]=0.0;
    }
 
   for(i = 0; i < NumPart; i++)
    { 
     rad = sqrt(P[i].Pos[0]*P[i].Pos[0] + P[i].Pos[1]*P[i].Pos[1] + P[i].Pos[2]*P[i].Pos[2]);
     
     /* consider only DM particles of SCF coefficients */
     if (P[i].Type != 1 || rad > max_rad)
      continue;
   
     for(k = 0; k < 3; k++)
      pos[k]+=P[i].Pos[k];
     
     num++;
    }  
   
   MPI_Allreduce(&pos[0], &pos_all[0], 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); 
   MPI_Allreduce(&num, &num_all, 3, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   
#ifdef DEBUG
  if (ThisTask==0)
   {
    printf("temp. center of mass = (%g|%g|%g)     iter=%d  num_all=%d\n", pos_all[0]/num_all, pos_all[1]/num_all, pos_all[2]/num_all, iter, num_all);
    printf("done.\n");
   }

#endif  
 
   for(i = 0; i < NumPart; i++)
   {    
    for(k = 0; k < 3; k++)
     P[i].Pos[k]-=pos_all[k]/num_all;
   }  
 
   if (num_all < fac_part*SCF_ntot_type_all[1])
    break;
    
   max_rad*=fac_rad;
   iter++;
  }


  if (ThisTask==0)
   {
    printf("SCF center of mass = (%g|%g|%g)   iter=%d\n", pos_all[0]/num_all, pos_all[1]/num_all, pos_all[2]/num_all, iter);
    printf("done.\n");
   }
}



/* Calculate SCF coefficients based on particle distribution */ 
void SCF_calc_from_particles(void)
{
 double r, un, unm1, costh, sinth, phi, xi;
 double plm1m,plm2m,temp1,temp2,temp3;
 int n,l,m;
 int i;
 double x, y, z, mass;

  
 for(i = 0; i < NumPart; i++)
  { 
   /* consider only DM particles of SCF coefficients */
   if (P[i].Type != 1)
    continue;

   /* scale to unit hernquist sphere */
   to_unit(P[i].Pos[0], P[i].Pos[1], P[i].Pos[2], &x, &y, &z);
   mass = P[i].Mass /SCF_HQ_MASS;
   
   /* OR: not */
   //x = P[i].Pos[0]; y = P[i].Pos[1]; z = P[i].Pos[2];
   //mass = P[i].Mass;

   /* particle coordinate transformation */
   r = sqrt(x * x + y * y + z * z);
   costh=z/r; 
   phi=atan2(y,x);
   xi=(r-1.)/(r+1.);
   sinth=sqrt(1.-costh*costh);
      
   for (m=0; m<=SCF_LMAX; m++) 
    {
     cosmphi[m]=cos(m*phi);           
     sinmphi[m]=sin(m*phi);
    } 

   /* Ultraspherical polynomials */
   for (l=0; l<=SCF_LMAX; l++) 
    {
     ultrasp[nl(0,l)]=1.0;
     ultrasp[nl(1,l)]=(2.0*(2.*l+1.5))*xi;  /* eq. (3.4b) */
     un=ultrasp[nl(1,l)];
     unm1=1.0;
       
     /* recursion */  
     for (n=1; n<=SCF_NMAX-1; n++) 
      {
       ultrasp[nl(n+1,l)]=((2.0*n+(2.0*(2.*l+1.5)))*xi*un-(1.0*n-1.0+(2.0*(2.*l+1.5)))*unm1)*(1.0/(n+1.0)); 
       unm1=un;
       un=ultrasp[nl(n+1,l)];
      }
     
     for (n=0; n<=SCF_NMAX; n++) 
      {
       ultraspt[nl(n,l)]=ultrasp[nl(n,l)]*Anltilde[nl(n,l)];
      }
    }
  
   /* Legendre polynomials */  
   for (m=0; m<=SCF_LMAX; m++) 
    {
     plm[lm(m,m)]=1.0;
     if (m>0) 
      plm[lm(m,m)]=-pow(1,m)*dblfact[m]*pow(sinth,m);
     plm1m=plm[lm(m,m)];
     plm2m=0.0;
     /* recursion */
     for (l=m+1; l<=SCF_LMAX; l++) 
      {
       plm[lm(l,m)]=(costh*(2.*l-1.)*plm1m-(l+m-1.)*plm2m)/(l-m);
       plm2m=plm1m;
       plm1m=plm[lm(l,m)];
     }       
    }

   for (l=0; l<=SCF_LMAX; l++) 
    {
     temp1=pow(r,l)/pow(1.+r,2*l+1)*mass;        
     for (m=0; m<=l; m++) 
      {
       temp2=temp1*plm[lm(l,m)]*coeflm[lm(l,m)]*sinmphi[m];
       temp3=temp1*plm[lm(l,m)]*coeflm[lm(l,m)]*cosmphi[m];
       for (n=0; n<=SCF_NMAX; n++) 
        {
	 sinsum[nlm(n,l,m)]+=temp2*ultraspt[nl(n,l)];
	 cossum[nlm(n,l,m)]+=temp3*ultraspt[nl(n,l)];
        } 
      } 
    }
  }  
}



/* write out coefficients and check potential + accelerations (note the scaling!) */
void SCF_collect_update(void)
{
 int n, l, m;

 for  (n=0; n<=SCF_NMAX; n++)
  {
   for  (l=0; l<=SCF_LMAX; l++) 
    {
     for  (m=0; m<=l; m++)
      {
       sinsum[nlm(n,l,m)]=sinsum_all[nlm(n,l,m)];
       cossum[nlm(n,l,m)]=cossum_all[nlm(n,l,m)];
      }
    }
  }  
}

#endif





/* write out coefficients and check potential + accelerations (note the scaling!) */
void SCF_write(int task)
{
 int n, l, m;

 /* print selected task */
 if (ThisTask == task)
 {
  for  (n=0; n<=SCF_NMAX; n++)
   {
    for  (l=0; l<=SCF_LMAX; l++) 
     {
      for  (m=0; m<=l; m++)
       {
        fprintf(FdSCF, "Step %d:   Task: %d   H[%d,%d,%d] = %g\n", All.NumCurrentTiStep, ThisTask, n, l, m, sinsum[nlm(n,l,m)]);
        fprintf(FdSCF, "Step %d:   Task: %d   G[%d,%d,%d] = %g\n", All.NumCurrentTiStep, ThisTask, n, l, m, cossum[nlm(n,l,m)]);       
	fflush(FdSCF);
       }
     }
   }  
  } 

 /* print all tasks */
 if (-1 == task)
 {
  for  (n=0; n<=SCF_NMAX; n++)
   {
    for  (l=0; l<=SCF_LMAX; l++) 
     {
      for  (m=0; m<=l; m++)
       {
        fprintf(FdSCF, "Step %d:   Task: %d   H[%d,%d,%d] = %g\n", All.NumCurrentTiStep, ThisTask, n, l, m, sinsum[nlm(n,l,m)]);
        fprintf(FdSCF, "Step %d:   Task: %d   G[%d,%d,%d] = %g\n", All.NumCurrentTiStep, ThisTask, n, l, m, cossum[nlm(n,l,m)]);       
	fflush(FdSCF);
       }
     }
   }  
  } 
}



#endif
