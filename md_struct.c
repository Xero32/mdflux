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
double BoundaryBound = 5.0;
double Boundary = 10.0;
double x_max; // lattice_const * 9.0 * sqrt(3.0) / sqrt(2.0);
double y_max; // lattice_const * 12. / sqrt(2.0);
double BoxHeight; // zSurface + 60
double area_inv;
double max_traj_inv;

pzz* new_pzz(int l){
  pzz *p;
  p = (pzz*) malloc(sizeof(pzz));

  p->grid = (double*) _mm_malloc(l * sizeof(double), 64);
  p->pz_u = (double*) _mm_malloc(l * sizeof(double), 64);
  p->pz_k = (double*) _mm_malloc(l * sizeof(double), 64);
  p->pz_s = (double*) _mm_malloc(l * sizeof(double), 64);
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
  _mm_free(p->pz_s);
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

Au* new_Au(int n){
  Au *a;
  a = (Au*) malloc(sizeof(Au));

  a->x = (double*) _mm_malloc(n * sizeof(double), 64);
  a->y = (double*) _mm_malloc(n * sizeof(double), 64);
  a->z = (double*) _mm_malloc(n * sizeof(double), 64);

  a->n = n;
  return a;
}

void make_lattice(Au *a){
  double dx = 6.66261 - 1.66565;
  double dy = 2.885;
  double dz = 2.283;
  double b1x, b2x, b1y, b2y, bz;
  int NX = 9;
  int NY = 12;
  int NZ = 6;
  int i,j,k;
  bz = -zSurface;
  int n = a->n;
  int ctr = 0;
  int ctr1 = 0;
  while(ctr < n){
    ctr1 = ctr + 1;
    for(k = 0; k < NZ; k++){
      //update z component afterwards, see below
      for(i = 0; i < NX; i++){
        // 2 lattice base vectors for fcc
        // for each layer a new set
        if(k % 3 == 0){
          b1x = 1.6656;
          b1y = 0.0;
          b2x = 4.04166;
          b2y = 1.40007;
        }else if(k % 3 == 1){
          b1x = 0.808332;
          b1y = 1.4007;
          b2x = 3.23333;
          b2y = 0.0;
        }else{
          b1x = 0.0;
          b1y = 0.0;
          b2x = 2.42499;
          b2y = 1.40007;
        }

        // as it is we reset x coordinates every time
        // therefore we add i * dx to the standard value
        // rather than update the coordinate w/o resetting
        // might be a bit tedious but in a python script this method worked
        // and you never change a running system!

        // turns out, it needs to be done this way, since for each layer we have new base vectors anyway
        b1x += i * dx;
        b2x += i * dx;

        for(j = 0; j < NY; j++){
          a->x[ctr] = b1x;
          a->y[ctr] = b1y;
          a->z[ctr] = bz;

          a->x[ctr1] = b1x;
          a->y[ctr1] = b1y;
          a->z[ctr1] = bz;

          b1y += dy;
          b2y += dy;
          ctr += 2;
        }
      }
      // update z component afterwards for next iteration
      bz += dz;
    }
  }
  return;
}

void del_Au(Au *a){
  _mm_free(a->x);
  _mm_free(a->y);
  _mm_free(a->z);
  free(a);
}

gofr* new_gofr(int l){
  gofr* g;
  g = (gofr*) malloc(sizeof(g));

  g->r = (double*) _mm_malloc(l * sizeof(double), 64);
  g->g = (double*) _mm_malloc(l * sizeof(double), 64);
  g->rmax = 0;
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
  g->rmax = end;
}

void del_gofr(gofr* g){
  printf("free g->r\n");
  _mm_free(g->r);
  printf("free g->r\n");
  _mm_free(g->g);
  printf("free g-\n");
  free(g);
}

int readfile(const char* name, ar* a){
  FILE* f = fopen(name, "r");
  int i = 0;
  char line[256];
  char *err;

  if(f){
    err = fgets(line, sizeof(line), f); // omit header
    if(err){
      while(fgets(line, sizeof(line), f) != NULL){
        sscanf(line, "%d, %d, %*d, %lf, %lf, %lf, %lf, %lf, %*f, %lf, %lf, %lf, %*f\n",
              &a->traj[i], &a->step[i],
              &a->x[i], &a->y[i], &a->z[i],
              &a->ix[i], &a->iy[i],
              &a->vx[i], &a->vy[i], &a->vz[i]);
        a->z[i] -= zSurface;
        i++;
      }
      a->n = i;
      fclose(f);
    }
  } else {
    return -1;
  }
  return 0;
}

/**
* Read the whole trajectory data file
* and save all data into one struct
*/
int readfileCountBoundParticles(const char* name, ar* a, int *number){
  FILE* f = fopen(name, "r");
  int i = 0;
  char line[256];
  int ctr = 0;
  char *err;
  if(f){
    err = fgets(line, sizeof(line), f); // omit header
    if(err){
      while(fgets(line, sizeof(line), f) != NULL){
          sscanf(line, "%d, %d, %*d, %lf, %lf, %lf, %lf, %lf, %*f, %lf, %lf, %lf, %*f\n",
                &a->traj[i], &a->step[i],
                &a->x[i], &a->y[i], &a->z[i],
                &a->ix[i], &a->iy[i],
                &a->vx[i], &a->vy[i], &a->vz[i]);
          a->z[i] -= zSurface;
          if(a->z[i] < Boundary){
            ctr++;
          }
          i++;
      }
      printf("lines: %d\n", i);
      *number = ctr;
      fclose(f);
    }
  }else{
    return -1;
  }
  // printf("%d percent\n\n",percent);
  return 0;
}

int readfileCountBulkParticles(const char* name, ar* a, int *number){
  FILE* f = fopen(name, "r");
  int i = 0;
  char line[256];
  int ctr = 0;
  char *err;
  if(f){
    err = fgets(line, sizeof(line), f); // omit header
    if(err){
      while(fgets(line, sizeof(line), f) != NULL){
          sscanf(line, "%d, %d, %*d, %lf, %lf, %lf, %lf, %lf, %*f, %lf, %lf, %lf, %*f\n",
                &a->traj[i], &a->step[i],
                &a->x[i], &a->y[i], &a->z[i],
                &a->ix[i], &a->iy[i],
                &a->vx[i], &a->vy[i], &a->vz[i]);
          a->z[i] -= zSurface;
          if(a->z[i] >= Boundary){
            ctr++;
          }
          i++;
      }
      printf("lines: %d\n", i);
      *number = ctr;
      fclose(f);
    }
  }else{
    return -1;
  }
  // printf("%d percent\n\n",percent);
  return 0;
}

/**
* Reads data from the whole array
* to a smaller local array, which conatins only the information
* about the given trajectory
* and particle regime (i.e. bound particles)
*/
//a is source, a2 is destination
void transferData(ar *a, ar *a2, int traj){
  int i;
  int n = a->n;
  int j = 0;
  for(i = 0; i < n; i++){
    // omitted bound particle condition in z coordinate
    // should be checked already in readfile function
    if (a->traj[i] == traj && a->z[i] < Boundary-0.1){
      // printf("hallo\n");
      a2->traj[j] = a->traj[i];
      a2->step[j] = a->step[i];
      a2->x[j] = a->x[i];
      a2->y[j] = a->y[i];
      a2->z[j] = a->z[i];
      a2->ix[j] = a->ix[i];
      a2->iy[j] = a->iy[i];
      a2->vx[j] = a->vx[i];
      a2->vy[j] = a->vy[i];
      a2->vz[j] = a->vz[i];
      j++;
    } else if(a->traj[i] > traj) break;
  }
}

/**
* Reads data from the whole array
* to a smaller local array, which conatins only the information
* about the given trajectory
* Copies all particles in that trajectory
*/
//a is source, a2 is destination
void transferDataAll(ar *a, ar *a2, int traj){
  int i;
  int n = a->n;
  int j = 0;
  for(i = 0; i < n; i++){
    // omitted bound particle condition in z coordinate
    // should be checked already in readfile function
    if (a->traj[i] == traj){
      // printf("hallo\n");
      a2->traj[j] = a->traj[i];
      a2->step[j] = a->step[i];
      a2->x[j] = a->x[i];
      a2->y[j] = a->y[i];
      a2->z[j] = a->z[i];
      a2->ix[j] = a->ix[i];
      a2->iy[j] = a->iy[i];
      a2->vx[j] = a->vx[i];
      a2->vy[j] = a->vy[i];
      a2->vz[j] = a->vz[i];
      j++;
    } else if(a->traj[i] > traj) break;
  }
}

/**
* Reads data from the whole array
* to a smaller local array, which conatins only the information
* about the given trajectory
* and particle regime (i.e. bulk particles)
*/
//a is source, a2 is destination
void transferDataBulk(ar *a, ar *a2, int traj){
  int i;
  int n = a->n;
  int j = 0;
  for(i = 0; i < n; i++){
    // omitted bound particle condition in z coordinate
    // should be checked already in readfile function
    if (a->traj[i] == traj && a->z[i] > Boundary){
      // printf("hallo\n");
      a2->traj[j] = a->traj[i];
      a2->step[j] = a->step[i];
      a2->x[j] = a->x[i];
      a2->y[j] = a->y[i];
      a2->z[j] = a->z[i];
      a2->ix[j] = a->ix[i];
      a2->iy[j] = a->iy[i];
      a2->vx[j] = a->vx[i];
      a2->vy[j] = a->vy[i];
      a2->vz[j] = a->vz[i];
      j++;
    } else if(a->traj[i] > traj) break;
  }
}

int readfileSingleTraj(const char* name, ar* a, int trajectory, int *index){
  FILE* f = fopen(name, "r");
  int i;
  int n = a->n;
  char line[256];
  int ctr = 0;
  int trajtest;
  char *err;
  if(f){
    err = fgets(line, sizeof(line), f); // omit header
    if(err){
      for(i = 0; i < n; i++){

        err = fgets(line, sizeof(line), f);
        if(err){
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
      }
      a->n = ctr;
      *index = ctr;
      fclose(f);
    }
  }else{
    return -1;
  }
  return 0;
}

void getMaxTime(ar *a, int *max){
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
    fprintf(f, "%d, %d, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n",
                a->traj[i], a->step[i],
                a->x[i], a->y[i], a->z[i],
                a->ix[i], a->iy[i],
                a->vx[i], a->vy[i], a->vz[i]);
  }
  fclose(f);
}

void printdata_i(ar *a, int i){
    printf("%d, %d, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n",
                a->traj[i], a->step[i],
                a->x[i], a->y[i], a->z[i],
                a->ix[i], a->iy[i],
                a->vx[i], a->vy[i], a->vz[i]);
}
void printdataAu_i(Au *a, int i){
    printf("%lf, %lf, %lf\n",
                a->x[i], a->y[i], a->z[i]);

}

void setParams(
  double *angle, int *pressure, int *temp_S, int *temp_P,
  int arg_c, char **arg_v){
  // read input parameters. for angle and pressure we need to do a conversion.
  // in main program we continue using the lower case variables for filenames
  int Angle;
  double Pressure;
  if (arg_c == 5){
    Angle = atoi(arg_v[1]);
    *temp_S = atoi(arg_v[2]);
    *temp_P = atoi(arg_v[3]);
    Pressure = atof(arg_v[4]);
    *pressure = (int) (Pressure * 10);
    printf("pressure %lf\n", Pressure);
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
