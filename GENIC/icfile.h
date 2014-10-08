typedef struct {
    double Pos[3];
    double Mass;
} ParType;

typedef struct {
    char * filename;
    struct io_header header;
    ParType * P;
    int NumPart;
} ICFile;


int read_header(struct io_header * header, char * filename);
int icfile_read(ICFile * icfile, char * filename);
void icfile_destroy(ICFile * icfile);
