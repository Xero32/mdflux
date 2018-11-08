#ifndef P_STRUCT
#define P_STRUCT

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <immintrin.h>

#define NO_OF_ENTRIES 1000000

// constants
extern double kB;
extern double e0;
extern double pi;
extern double au;
extern double atm;
extern double NA;
// lattice properties
extern const double area;
extern const double lattice_const; // for gold
extern double zSurface;
extern double x_max; // lattice_const * 9.0 * sqrt(3.0) / sqrt(2.0);
extern double y_max; // lattice_const * 12. / sqrt(2.0);
extern double BoxHeight; // zSurface + 60

typedef struct ar{
  int *traj;
  int *step;
  double *x;
  double *y;
  double *z;
  double *ix;
  double *iy;
  double *vx;
  double *vy;
  double *vz;
  int n;
  int maxtime;
} ar;

ar* new_ar(int n);
void getMaxTime(ar *a, int stw, int *max);
void del_ar(ar *a);

typedef struct pzz{
  double *grid;
  double *pz_u; // potential component
  double *pz_k; // kinetic component
  double dz;
  int l;
} pzz;

pzz* new_pzz(int l);
void make_pgrid(pzz *p, double start, double end);
void del_pzz(pzz *p);

typedef struct gofr{
  double *r; // radius space
  double *g; // value of g(r)
  int t1; // starttime of evaluation
  int t2; // endtime of evaluation
  int l; // number of grid points
  int rmax; // maximal r value
  double dr; // resolution of grid
} gofr;

gofr* new_gofr(int l);
void make_ggrid(gofr* g, double start, double end);
void del_gofr(gofr* g);

int readfile(const char* name, ar* a);
int readfileSingleTraj(const char* name, ar* a, int trajectory, int *index);
int readfileBoundParticles(const char* name, ar* a, int *number);
int readfileCountBoundParticles(const char* name, ar* a, int *number);
void transferData(ar *a, ar *a2, int traj);
void writedata(const char* fname, ar *a, int maxline);
void printdata_i(ar *a, int i);
void setParams(  double *angle, int *pressure, int *temp_S, int *temp_P, int arg_c, char **arg_v);
#endif /* P_STRUCT */
