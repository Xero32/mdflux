#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <immintrin.h>
#include <pwd.h>
#include <time.h>
#include "md_struct.h"
#include "gofr.h"

//step width
#define STW 1000
#define NumOfTraj 20

extern double Boundary;


void GofR(ar* a, gofr *g, int trajnum, int maxtime, int range, int number){
  // make sure z coordinate is shifted with surface as 0
  printf("Caclulate radial pair distribution function\n");
  int i, j, k, t;

  // int n = number;
  int n = a->n;
  int l = g->l;
  clock_t tstart, tend;
  double particles = 0.0;
  double dr = g->dr;
  double rmax = g->rmax;
  // use coefficient of linear mapping from r space to array index
  double coeff = 1.0 / dr; // in example this should
  g->t1 = maxtime-range;
  g->t2 = maxtime;
  double x,y,z, dx, dy, dz, R_norm;
  // caution! hard coded. take 5.0 AA as volume height
  // double volume = x_max*y_max*5.0*1.0e-30;
  // iterate over trajectories
  k = trajnum;
  printf("\t%dth trajectory\n", k);
  tstart = clock();
  // iterate over time
  // #pragma omp parallel for num_threads(4)
  for(t = maxtime-range; t < maxtime; t += STW){
    for(i = 0; i < n; i++){
      // assure to analyze the correct step + traj
      // omitted boundary condition, as we should only have bound particles in ar struct
      // omitted trajectory condition, which is implemented in fct 'transferData'
      if(a->step[i] == t){
        particles += 1.0;
        x = a->x[i];
        y = a->y[i];
        z = a->z[i];

        for(j = i; j < n; j++){
          if(a->step[j] == t && i != j){
            dx = x - (a->x[j]);
            dy = y - (a->y[j]);
            dz = z - (a->z[j]);
            R_norm = sqrt(dx*dx + dy*dy + dz*dz);

            if (R_norm < rmax){
              g->g[(int) (R_norm * coeff)] += 1.0;
            }
          }
        }
      }
    }
  }
  tend = clock();
  printf("Time elapsed: %f\n", (double)(tend-tstart)/CLOCKS_PER_SEC);
  // average particle count over all trajectories and time iterations
  // normalize g(r) with particle number and volume
  particles = particles;
  for(i = 0; i < l; i++){
    g->g[i] *= 1.0 / particles;// * volume / particles;
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
  // set up Pressure in atm for potential later use.
  // and just do something with it, so the compiler won't complain
  double Pressure = pressure / 10.0; // returns pressure in atm
  if(Pressure == Pressure){
    Pressure += 0.0;
  }
  char *home = getenv("HOME");
  x_max =lattice_const * 9.0 * sqrt(3.0) / sqrt(2.0);
  y_max = lattice_const * 12. / sqrt(2.0);
  // read traj data
  char dest[256] = "";
  int numberOfParticles;
  int length = 240;
  double start = 0.0;
  double end = 24.0;
  int trajnum = 20;
  printf("init a\n");
  ar *a0 = new_ar(NO_OF_ENTRIES);


  printf("init g\n");
  gofr *g0 = new_gofr(length);


  printf("init grid\n");
  int maximumtime = 0;
  int timerange = 40000; // time range corresponds to 100 ps

  // init gofr and its grid
  make_ggrid(g0, start, end);
  // printf("%d\n",g0->l);
  // for(int j = 0; j < length; j++){
  //   printf("%lf, %le\n", g0->r[j], g0->g[j]);
  // }

  // printf("Read Trajectory %d\n", i);
  char dir[] = "/lammps/flux/";
  char fname[64];
  snprintf(fname, sizeof(fname), "A%03d_TS%dK_TP%dK_p%02ddatm.csv",
      (int)(angle*100), temp_S, temp_P, pressure);
  strcat(dest, home);
  strcat(dest, dir);
  strcat(dest, fname);
  int traj;
  if (0 == readfileCountBoundParticles(dest, a0, &numberOfParticles)){
    printf("Num of Particles %d\n", numberOfParticles);
    getMaxTime(a0, STW, &maximumtime);
  }else{
    printf("Unable to open file %s\n", dest);
    return -1;
  }
  // use new a struct to save relevant data, i.e. bound particles in one single traj
  ar *a = new_ar(numberOfParticles);



  printf("Max Time: %d, Time Range: %d\n", maximumtime, timerange);
  // actual calculation of g(r)
  for(traj = 0; traj < trajnum; traj++){
    transferData(a0, a, traj);
    GofR(a, g0, traj, maximumtime, timerange, numberOfParticles);
  }
  del_ar(a0);
  del_ar(a);

  for(int i = 0; i < length; i++){
    printf("%lf, %le\n", g0->r[i], g0->g[i] * trajnum);
  }

  memset(dest,0,strlen(dest));
  memset(dir,0,strlen(dir));
  strcat(dir, "/lammps/");
  char gname[100];
  strcat(dest, home);
  strcat(dest, dir);
  snprintf(gname, sizeof(gname), "gofrA%03d_TS%dK_TP%dK_p%02ddatm.csv",
      (int)(angle*100), temp_S, temp_P, pressure);
  strcat(dest, gname);
  // writeGofR(g0, dest);
  // writeGofR(g0, "/home/becker/lammps/test.dat");
  printf("check2\n");
  printf("so far so good\n");
  del_gofr(g0);
  printf("lalalal\n");

  return 0;
}
