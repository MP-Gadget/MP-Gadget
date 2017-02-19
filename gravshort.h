
/*! length of lock-up table for short-range force kernel in TreePM algorithm */
#define NTAB 1000
/*! variables for short-range lookup table */
static float shortrange_table[NTAB], shortrange_table_potential[NTAB], shortrange_table_tidal[NTAB];

/*! toggles after first tree-memory allocation, has only influence on log-files */
static int first_flag = 0;

static void fill_ntab()
{
    if(first_flag == 0)
    {
        first_flag = 1;
        int i;
        for(i = 0; i < NTAB; i++)
        {
            double u = 3.0 / NTAB * (i + 0.5);
            shortrange_table[i] = erfc(u) + 2.0 * u / sqrt(M_PI) * exp(-u * u);
            shortrange_table_potential[i] = erfc(u);
            shortrange_table_tidal[i] = 4.0 * u * u * u / sqrt(M_PI) * exp(-u * u);
        }
    }

}

typedef struct {
    TreeWalkNgbIterBase base;
} TreeWalkNgbIterGravShort;

typedef struct
{
    TreeWalkQueryBase base;
    int Type;
#ifdef ADAPTIVE_GRAVSOFT_FORGAS
    MyFloat Soft;
#endif
    MyFloat OldAcc;
} TreeWalkQueryGravShort;

typedef struct {
    TreeWalkResultBase base;
    MyFloat Acc[3];
    MyFloat Potential;
    int Ninteractions;
} TreeWalkResultGravShort;

static int
grav_short_isactive(int i)
{
    int isactive = 1;
    /* tracer particles (5) has no gravity, they move along to pot minimium */
    isactive = isactive && (P[i].Type != 5);
    return isactive;
}

static void
grav_short_postprocess(int i)
{
    int j;

    double ax, ay, az;
    ax = P[i].GravAccel[0] + P[i].GravPM[0] / All.G;
    ay = P[i].GravAccel[1] + P[i].GravPM[1] / All.G;
    az = P[i].GravAccel[2] + P[i].GravPM[2] / All.G;

    P[i].OldAcc = sqrt(ax * ax + ay * ay + az * az);
    for(j = 0; j < 3; j++)
        P[i].GravAccel[j] *= All.G;

    /* calculate the potential */
    /* remove self-potential */
    P[i].Potential += P[i].Mass / All.SofteningTable[P[i].Type];

    P[i].Potential -= 2.8372975 * pow(P[i].Mass, 2.0 / 3) *
        pow(All.CP.Omega0 * 3 * All.Hubble * All.Hubble / (8 * M_PI * All.G), 1.0 / 3);

    P[i].Potential *= All.G;

    P[i].Potential += P[i].PM_Potential;	/* add in long-range potential */

}

static void
grav_short_copy(int place, TreeWalkQueryGravShort * input) {
    input->Type = P[place].Type;

#ifdef ADAPTIVE_GRAVSOFT_FORGAS
    if(P[place].Type == 0)
        input->Soft = P[place].Hsml;
#endif
    input->OldAcc = P[place].OldAcc;

}
static void
grav_short_reduce(int place, TreeWalkResultGravShort * result, enum TreeWalkReduceMode mode) {
    int k;
    for(k = 0; k < 3; k++)
        TREEWALK_REDUCE(P[place].GravAccel[k], result->Acc[k]);

    TREEWALK_REDUCE(P[place].GravCost, result->Ninteractions);
    TREEWALK_REDUCE(P[place].Potential, result->Potential);
}

