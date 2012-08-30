 

int iddat_sort_compare(const void *a, const void *b);
int iddat_gas_sort_compare(const void *a, const void *b);
int iddat_stars_sort_compare(const void *a, const void *b);

int get_total_number_of_groups(int argc,void *argv[]);
int get_group_catalogue(int argc,void *argv[]);
int get_hash_table(int argc,void *argv[]);
int get_hash_table_size(int argc,void *argv[]);
int get_group_coordinates(int argc,void *argv[]);

int get_minimum_group_len(int argc, void *argv[]);
int get_groupcount_below_minimum_len(int argc, void *argv[]);
int get_particle_data_with_vel(int argc, void *argv[]);

int get_group_catalogue_bh(int argc, void *argv[]);


int id_sort_compare_key(const void *a, const void *b);
int id_sort_groups(const void *a, const void *b);
double fof_periodic_wrap(double x);
double fof_periodic(double x);
