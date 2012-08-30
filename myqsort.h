/* must define macros QSORT, KEY_TYPE, STRUCT_TYPE, KEY_COPY, GET_KEYVAL */
static void QSORT(KEY_TYPE *a,int n)
{
#pragma alloca
  KEY_TYPE *b = (KEY_TYPE *)alloca(sizeof(KEY_TYPE)*n);

  if (n > 1) {
     int i,j1,j2;
     double avg = 0.0;
     KEY_BASE_TYPE kavg;

     /* compute average key value and use as pivot point */
     
     for(i=0;i < n;i++)
        avg += KEY_GETVAL(&a[i]);

     kavg = (KEY_BASE_TYPE) (avg / (double)n);

     j1 = 0;
     j2 = n;

     for (i = 0;i < n;i++)
       if (KEY_GETVAL(&a[i]) <= kavg) {
          KEY_COPY(&a[i],&b[j1]);
          j1++;
       }
       else {
          --j2;
          KEY_COPY(&a[i],&b[j2]);
       }

     if (j1 > 0 && j2 < n) {
     
        if (j1 > 0) QSORT(b,j1);
        if (j2 < n) QSORT(&b[j1],n-j1);

        memcpy(a,b,sizeof(KEY_TYPE)*n);
     }
  }
}
