
typedef struct {
    /* aligned at 8 bytes on 64bit machines; this is ugly shall replace with
     * int64 etc because it is used in IO. */
    size_t Ngrid;
    double BoxSize;

    /* private */
    size_t size;
    double CellSize;
    size_t strides[3];
    double * buffer;
} CIC;

void cic_init(CIC * cic, int Ngrid, double BoxSize);
void cic_add_particle(CIC * cic, double Pos[3], double mass);
void cic_destroy(CIC * cic);
