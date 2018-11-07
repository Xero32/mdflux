#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <immintrin.h>
#include <pwd.h>
#include "md_struct.h"
#include "gofr.h"

//step width
#define STW 1000
#define NumOfTraj 20

extern double Boundary;


void GofR(ar* a, gofr *g, int trajnum, int maxtime, int range){
  // make sure z coordinate is shifted with surface as 0
  printf("Caclulate radial pair distribution function\n");
  int i, j, k, t, r;
  int n = a->n;
  int l = g->l;
  double particles = 0.0;
  double dr = g->dr;
  g->t1 = maxtime-range;
  g->t2 = maxtime;
  double x,y,z, dx, dy, dz, R_norm, R;
  // caution! hard coded. take 5.0 AA as volume height
  double volume = x_max*y_max*5.0*1.0e-30;
  // iterate over trajectories
  for(k = 0; k < trajnum; k++){
    printf("\t%dth trajectory\n", k);
    // iterate over time
    // #pragma omp parallel for num_threads(4)
    for(t = maxtime-range; t < maxtime; t += STW*5){
      printf("\t\tTimestep %d\n", t);
      // scan grid
      for(r = 0; r < l; r++){
        R = g->r[r];
        // iterate over particles
        {
          for(i = 0; i < n; i++){
            // assure to analyze the correct step + traj
            // omitted boundary condition, as we should only have bound particles in ar struct
            if(a->traj[i] == k && a->step[i] == t){
              particles += 1.0;
              x = a->x[i] + a->ix[i] * x_max;
              y = a->y[i] + a->iy[i] * y_max;
              z = a->z[i];

              for(j = 0; j < i; j++){
                dx = x - (a->x[j] + a->ix[j] * x_max);
                dy = y - (a->y[j] + a->iy[j] * y_max);
                dz = z - (a->z[j]);
                R_norm = sqrt(dx*dx + dy*dy + dz*dz);

                if (R < R_norm && R_norm <= R+dr){
                  g->g[r] += 1.0;
                }
              }
              for(j = i+1; j < n; j++){
                dx = x - (a->x[j] + a->ix[j] * x_max);
                dy = y - (a->y[j] + a->iy[j] * y_max);
                dz = z - (a->z[j]);
                R_norm = sqrt(dx*dx + dy*dy + dz*dz);

                if (R < R_norm && R_norm <= R+dr){
                  g->g[r] += 1.0;
                }
              }
            }
          }
        }
      }
    }
  }
  // average particle count over all trajectories and time iterations
  // normalize g(r) with particle number and volume
  particles = particles / (trajnum * range * STW);
  for(r = 0; r < l; r++){
    g->r[r] = g->r[r] * volume / particles;
  }
}

void writeGofR(gofr *g, const char* gname){
  FILE* f = fopen(gname, "w");
  int i, l;
  printf("get l\n");
  l = g->l;
  if(f){
    printf("Opened file %s\n", gname);
    for(i = 0; i < l; i++){
      printf("printing line\n");
      fprintf(f, "%lf, %le\n", g->r[i], g->g[i]);
    }
    fclose(f);
  }else{
  printf("Could not open file %s\n", gname);
  }
}


int main(int argc, char** argv){
  double angle;
  int pressure, temp_S, temp_P;
  setParams(&angle, &pressure, &temp_S, &temp_P, argc, argv);
  double Pressure = pressure / 10.0; // returns pressure in atm
  char *home = getenv("HOME");
  x_max =lattice_const * 9.0 * sqrt(3.0) / sqrt(2.0);
  y_max = lattice_const * 12. / sqrt(2.0);
  // read traj data
  char dest[256] = "";

  int length = 120;
  double start = 0.0;
  double end = 12.0;
  printf("init a\n");
  ar *a0 = new_ar(NO_OF_ENTRIES/10);

  printf("init g\n");
  gofr *g0 = new_gofr(length);


  printf("init grid\n");
  int maximumtime = 0;
  int timerange = 10000; // time range corresponds to 200 ps

  // init gofr and its grid
  make_ggrid(g0, start, end);

  // printf("Read Trajectory %d\n", i);
  char dir[] = "/lammps/flux/";
  char fname[64];
  snprintf(fname, sizeof(fname), "A%03d_TS%dK_TP%dK_p%02ddatm.csv",
      (int)(angle*100), temp_S, temp_P, pressure);
  strcat(dest, home);
  strcat(dest, dir);
  strcat(dest, fname);
  if (readfileBoundParticles(dest, a0) == 0){
    getMaxTime(a0, STW, &maximumtime);
  }else{
    printf("Unable to open file %s\n", dest);
    return -1;
  }
  memset(dest,0,strlen(dest));
  memset(dir,0,strlen(dir));

  printf("Max Time: %d, Time Range: %d\n", maximumtime, timerange);

  // gofr *g = new_gofr(length);
  // make_ggrid(g, start, end);

  // actual calculation of g(r)

  GofR(a0, g0, NumOfTraj, maximumtime, timerange); del_ar(a0);

  printf("check\n");

  strcat(dir, "/lammps/");
  char gname[100];
  strcat(dest, home);
  strcat(dest, dir);
  snprintf(gname, sizeof(gname), "gofrA%03d_TS%dK_TP%dK_p%02ddatm.csv",
      (int)(angle*100), temp_S, temp_P, pressure);
  strcat(dest, gname);
  // writeGofR(g0, dest);
  writeGofR(g0, "/home/becker/lammps/test.dat");
  printf("check2\n");
  printf("so far so good\n");
  del_gofr(g0);

  return 0;
}
