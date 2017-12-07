#ifndef PROTO_H
#define PROTO_H

void begrun(int RestartSnapNum);
void
open_outputfiles(int RestartsnapNum);
void
close_outputfiles(void);

void density();
void density_update();
void energy_statistics(void);

void grav_short_tree(void);
void hydro_force(void);
void init(int RestartSnapNum);
void run(void);
void runtests(void);

void write_checkpoint(int write_snapshot, int write_fof);
void dump_snapshot(void);
int find_last_snapnum();

void long_range_init(void);
void gravpm_force(void);


#ifdef LIGHTCONE
void lightcone_init(double timeBegin);
void lightcone_cross(int p, double oldpos[3]);
void lightcone_set_time(double a);
#endif

#endif //PROTO_H
