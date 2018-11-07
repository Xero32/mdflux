#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <immintrin.h>
#include "md_struct.h"

// constants
double kB = 1.3806503e-23;
double e0 = 1.60217662e-19;
double pi = 3.14159265359;
double au = 1.66053904e-27;
double atm = 101325;
double NA = 6.02214086e23;
// lattice properties
const double area = 1557e-20;
const double lattice_const = 4.08; // for gold
double zSurface = 11.8;
double Boundary = 5.0;
double x_max; // lattice_const * 9.0 * sqrt(3.0) / sqrt(2.0);
double y_max; // lattice_const * 12. / sqrt(2.0);
double BoxHeight; // zSurface + 60

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

void make_pgrid(pzz *p, double start, double end){
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

gofr* new_gofr(int l){
  gofr* g;
  g = (gofr*) malloc(sizeof(g));

  g->r = (double*) _mm_malloc(l * sizeof(double), 64);
  g->g = (double*) _mm_malloc(l * sizeof(double), 64);

  g->t1 = 0;
  g->t2 = 0;
  g->l = l;
  g->dr = 0.0;

  return g;
}

void make_ggrid(gofr* g, double start, double end){
  int i;
  int l = g->l;
  double step = (end - start) / l;
  for(i = 0; i < l; i++){
    g->r[i] = start + i * step;
  }
  g->dr = step;
}

void del_gofr(gofr* g){
  _mm_free(g->r);
  _mm_free(g->g);
  free(g);
}

int readfile(const char* name, ar* a){
  FILE* f = fopen(name, "r");
  int i;
  int n = a->n;
  char line[256];
  int percent = 0;
  int ctr = 0;
  if(f){
    fgets(line, sizeof(line), f); // omit header
    for(i = 0; i < n; i++){
      ctr++;
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
    a->n = ctr;
    fclose(f);
  }else{
    return -1;
  }
  printf("%d percent\n\n",percent);
  return 0;
}

int readfileBoundParticles(const char* name, ar* a){
  FILE* f = fopen(name, "r");
  int i;
  int n = a->n;
  char line[256];
  int percent = 0;
  int ctr = 0;
  double z;
  if(f){
    fgets(line, sizeof(line), f); // omit header
    for(i = 0; i < n; i++){

      if (i % (n / 10) == 0){
        printf("%d percent\n",percent);
        percent += 10;
      }
      fgets(line, sizeof(line), f);
      sscanf(line, "%*d, %*d, %*d, %*f, %*f, %lf, %*f, %*f, %*f, %*f, %*f, %*f, %*f\n", &z);
      if (z <= zSurface+Boundary){
        ctr++;
        sscanf(line, "%d, %d, %*d, %lf, %lf, %lf, %lf, %lf, %*f, %lf, %lf, %lf, %*f\n",
              &a->traj[i], &a->step[i],
              &a->x[i], &a->y[i], &a->z[i],
              &a->ix[i], &a->iy[i],
              &a->vx[i], &a->vy[i], &a->vz[i]);
        a->z[i] -= zSurface;
      }
    }
    a->n = ctr;
    printf("%d\n",ctr);
    fclose(f);
  }else{
    return -1;
  }
  printf("%d percent\n\n",percent);
  return 0;
}

int readfileSingleTraj(const char* name, ar* a, int trajectory, int *index){
  FILE* f = fopen(name, "r");
  int i;
  int n = a->n;
  char line[256];
  int ctr = 0;
  int trajtest;
  if(f){
    fgets(line, sizeof(line), f); // omit header
    for(i = 0; i < n; i++){

      fgets(line, sizeof(line), f);
      sscanf(line, "%d, %*d, %*d, %*f, %*f, %*f, %*f, %*f, %*f, %*f, %*f, %*f, %*f\n", &trajtest);
      if (trajtest == trajectory){
        sscanf(line, "%d, %d, %*d, %lf, %lf, %lf, %lf, %lf, %*f, %lf, %lf, %lf, %*f\n",
              &a->traj[i], &a->step[i],
              &a->x[i], &a->y[i], &a->z[i],
              &a->ix[i], &a->iy[i],
              &a->vx[i], &a->vy[i], &a->vz[i]);
        a->z[i] -= zSurface;
        ctr++;
      }
    }
    a->n = ctr;
    *index = ctr;
    fclose(f);
  }else{
    return -1;
  }
  return 0;
}

void getMaxTime(ar *a, int stw, int *max){
  int i;
  int n = a->n;
  for(i = 0; i < n; i++){
    *max = (a->step[i] > *max) ? a->step[i] : *max;
  }
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

void setParams(
  double *angle, int *pressure, int *temp_S, int *temp_P,
  int arg_c, char **arg_v){
  // read input parameters. for angle and pressure we need to do a conversion.
  // in main program we continue using the lower case variables for filenames
  int Angle, Pressure;
  if (arg_c == 5){
    Angle = atoi(arg_v[1]);
    *temp_S = atoi(arg_v[2]);
    *temp_P = atoi(arg_v[3]);
    Pressure = atof(arg_v[4]);
    *pressure = (int) (Pressure * 10);
  }else{
    Angle = 30;
    *angle = 0.52;
    *temp_S = 300;
    *temp_P = 300;
    Pressure = 1.0;
    *pressure = 10;
  }
  switch(Angle){
    case 0:
      *angle = 0.00;
      break;
    case 30:
      *angle = 0.52;
      break;
    case 45:
      *angle = 0.80;
      break;
    case 60:
      *angle = 1.05;
      break;
    default:
      *angle = 0.52;
  }
}
