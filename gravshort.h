
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
grav_short_isinteracting(int i, TreeWalk * tw)
{
    int isinteracting = 1;
    /* tracer particles (5) has no gravity, they move along to pot minimium */
    isinteracting = isinteracting && (P[i].Type != 5);
    return isinteracting;
}

static void
grav_short_postprocess(int i, TreeWalk * tw)
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
grav_short_copy(int place, TreeWalkQueryGravShort * input, TreeWalk * tw)
{
    input->Type = P[place].Type;

#ifdef ADAPTIVE_GRAVSOFT_FORGAS
    if(P[place].Type == 0)
        input->Soft = P[place].Hsml;
#endif
    input->OldAcc = P[place].OldAcc;

}
static void
grav_short_reduce(int place, TreeWalkResultGravShort * result, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    int k;
    for(k = 0; k < 3; k++)
        TREEWALK_REDUCE(P[place].GravAccel[k], result->Acc[k]);

    TREEWALK_REDUCE(P[place].GravCost, result->Ninteractions);
    TREEWALK_REDUCE(P[place].Potential, result->Potential);
}

int grav_apply_short_range_window(double r, double * fac, double * facpot);
