#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <immintrin.h>
#include "pressure.h"
#include "p_struct.h"

double kB = 1.3806503e-23;
double e0 = 1.60217662e-19;
double pi = 3.14159265359;
double au = 1.66053904e-27;
double atm = 101325;
double NA = 6.02214086e23;
const double area = 1557e-20;
const double lattice_const = 4.08; // for gold
double zSurface = 11.8;

pzz* new_pzz(int l){
  pzz *p;
  p = (pzz*) malloc(sizeof(pzz));

  p->grid = (double*) _mm_malloc(l * sizeof(double), 64);
  p->pz_u = (double*) _mm_malloc(l * sizeof(double), 64);
  p->pz_k = (double*) _mm_malloc(l * sizeof(double), 64);
  p->l = l;
  p->dz = 0.0;

  return p;
}

void make_grid(pzz *p, double start, double end){
  int i;
  int l = p->l;
  double step = (end - start) / l;
  for(i = 0; i < l; i++){
    p->grid[i] = start + i * step;
  }
  p->dz = step;
}

void del_pzz(pzz *p){
  _mm_free(p->grid);
  _mm_free(p->pz_u);
  _mm_free(p->pz_k);
  free(p);
}

ar* new_ar(int n){
  ar *a;
  a = (ar*) malloc(sizeof(ar));

  a->traj = (int*) _mm_malloc(n * sizeof(int), 64);
  a->step = (int*) _mm_malloc(n * sizeof(int), 64);

  a->x = (double*) _mm_malloc(n * sizeof(double), 64);
  a->y = (double*) _mm_malloc(n * sizeof(double), 64);
  a->z = (double*) _mm_malloc(n * sizeof(double), 64);

  a->ix = (double*) _mm_malloc(n * sizeof(double), 64);
  a->iy = (double*) _mm_malloc(n * sizeof(double), 64);

  a->vx = (double*) _mm_malloc(n * sizeof(double), 64);
  a->vy = (double*) _mm_malloc(n * sizeof(double), 64);
  a->vz = (double*) _mm_malloc(n * sizeof(double), 64);
  a->n = n;

  return a;
}

void del_ar(ar *a){
  _mm_free(a->traj);
  _mm_free(a->step);

  _mm_free(a->x);
  _mm_free(a->y);
  _mm_free(a->z);

  _mm_free(a->ix);
  _mm_free(a->iy);

  _mm_free(a->vx);
  _mm_free(a->vy);
  _mm_free(a->vz);
  free(a);
}

int readfile(const char* name, ar* a){
  FILE* f = fopen(name, "r");
  int i;
  int n = a->n;
  char line[256];
  int percent = 0;
  if(f){
    fgets(line, sizeof(line), f); // omit header
    for(i = 0; i < n; i++){
      if (i % (n / 10) == 0){
        printf("%d percent\n",percent);
        percent += 10;
      }
      fgets(line, sizeof(line), f);
      sscanf(line, "%d, %d, %*d, %lf, %lf, %lf, %lf, %lf, %*f, %lf, %lf, %lf, %*f\n",
            &a->traj[i], &a->step[i],
            &a->x[i], &a->y[i], &a->z[i],
            &a->ix[i], &a->iy[i],
            &a->vx[i], &a->vy[i], &a->vz[i]);
      a->z[i] -= zSurface;
    }
    fclose(f);
  }else{
    return -1;
  }
  printf("%d percent\n\n",percent);
  return 0;
}

void writedata(const char* fname, ar *a, int maxline){
  FILE* f = fopen(fname, "w");
  int i;
  for(i = 0; i < maxline; i++){
    fprintf(f, "%d, %d, \
                %lf, %lf, %lf, \
                %lf, %lf, \
                %lf, %lf, %lf\n",
                a->traj[i], a->step[i],
                a->x[i], a->y[i], a->z[i],
                a->ix[i], a->iy[i],
                a->vx[i], a->vy[i], a->vz[i]);
  }
  fclose(f);
}

void printdata_i(ar *a, int i){
    printf("%d, %d, \
                %lf, %lf, %lf, \
                %lf, %lf, \
                %lf, %lf, %lf\n",
                a->traj[i], a->step[i],
                a->x[i], a->y[i], a->z[i],
                a->ix[i], a->iy[i],
                a->vx[i], a->vy[i], a->vz[i]);
}
