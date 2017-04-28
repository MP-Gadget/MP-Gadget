#ifndef __UTILS_SYSTEM_H__
#define __UTILS_SYSTEM_H__
void x86_fix(void) ;

int
cluster_get_num_hosts();
int
cluster_get_hostid();
double
get_physmem_bytes();

#endif
