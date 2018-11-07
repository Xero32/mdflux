#ifndef PRESSURE_H
#define PRESSURE_H

#include "md_struct.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>

// ar ar;
// pzz pzz;

double force_z(double xi, double xj, double yi, double yj, double zi, double zj, double epsilon, double sigma);
// may want to optimize sign function by bitwise shifting to obtain the leading (i.e. pseudo-sign) bit
int sign(float n);
int within(double x, double a, double b);
double weight_dz(double zi, double zj, double z, double dz);
// caution! sets pz to 0
double pressure_component(ar* a, int time, int traj, int start, int num, double z, double dz, double epsilon, double sigma);
double kinetic_pressure_component(ar* a, pzz* p, double m, double u, double rho, double z, double time, int traj, int start);
int pressure_tensor(ar* a, pzz* p, int time, int traj, int *prev, double epsilon, double sigma);
void kinetic_pressure_tensor(ar* a, pzz* p, int time, int traj, int prev, double u, double rho, double m);
void calc_pressure(ar* a, pzz* p, int time, double u, double rho, double epsilon, double sigma, double m);
void write_pressure(pzz* p, const char* fname);

#endif /* PRESSURE_H */
