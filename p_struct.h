#ifndef P_STRUCT
#define P_STRUCT

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <immintrin.h>
#include "p_struct.h"



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
} ar;

ar* new_ar(int n);
void del_ar(ar *a);

typedef struct pzz{
  double *grid;
  double *pz_u; // potential component
  double *pz_k; // kinetic component
  double dz;
  int l;
} pzz;

pzz* new_pzz(int l);
void make_grid(pzz *p, double start, double end);
void del_pzz(pzz *p);

typedef struct gofr{
  double *r;
  double *g;
  int t1;
  int t2;
  int l;
  double dr;
} gofr;

gofr* new_gofr(int n, int t);
void make_grid(gofr* g, double start, double end);
void del_gofr(gofr* g);

int readfile(const char* name, ar* a);
void writedata(const char* fname, ar *a, int maxline);
void printdata_i(ar *a, int i);

#endif /* P_STRUCT */
