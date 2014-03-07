

typedef struct {
    int Ndim;
    int * dims;
    ptrdiff_t * strides;
    double * Min;
    double * Step;
    double * Max; 

    void * data; /* internal buffer for all pointer data */
    int fsize;
} Interp;

void interp_init(Interp * obj, int Ndim, int * dims);

/* set the upper and lower limit of dimension d */
void interp_init_dim(Interp * obj, int d, double Min, double Max);

/* returns the index to the y value on the table at xi */
ptrdiff_t interp_index(Interp * obj, int * xi);

/* interpolate the table at point x; 
 * status: array of length dimension,
 * will be -1 if below lower bound
 *         +1 if above upper bound  */
double interp_eval(Interp * obj, double * x, double * ydata, int * status);

void interp_destroy(Interp * obj);

