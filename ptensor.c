#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <immintrin.h>
#include <pwd.h>
#include "pressure.h"
#include "md_struct.h"

// constants
extern double kB;
extern double e0;
extern double pi;
extern double au;
extern double atm;
extern double NA;
// lattice properties
extern const double area;
extern const double lattice_const;
extern double zSurface;
extern double x_max; // lattice_const * 9.0 * sqrt(3.0) / sqrt(2.0);
extern double y_max; // lattice_const * 12. / sqrt(2.0);
extern double BoxHeight; // zSurface + 60


// Lennard-Jones parameter for argon
double epsilon;
double sigma;

int NoOfEntries;
int max_traj;
double max_traj_inv;
double m;
double area_inv;



int main(int argc, char** argv){

  x_max = lattice_const * 9.0 * sqrt(3.0) / sqrt(2.0);
  y_max = lattice_const * 12. / sqrt(2.0);
  max_traj = 20;
  max_traj_inv = 1.0 / max_traj;
  // Lennard-Jones parameter for argon
  epsilon = 1.67e-21 / e0;
  sigma = 3.4;
  area_inv = 1.0 / area;
  m = 40.0 * au;
  double u = 0.0;
  double rho = 0.0;

  double angle;
  int pressure, temp_S, temp_P;
  setParams(&angle, &pressure, &temp_S, &temp_P, argc, argv);
  double Pressure = pressure / 10.0; // returns pressure in atm

  char *home = getenv("HOME");
  char dest[100];


  printf("Initializing Particle struct\n");
  ar *a;
  a = new_ar(NO_OF_ENTRIES);

  printf("Initializing grid\n");
  int l = 150;
  pzz* p;
  p = new_pzz(l);
  make_pgrid(p, 0.0, 60.0);

  char dir[] = "/lammps/flux/";
  char fname[100];
  snprintf(fname, sizeof(fname), "A%03d_TS%dK_TP%dK_p%02ddatm.csv",
      (int)(angle*100), temp_S, temp_P, pressure);
  strcat(dest, home);
  strcat(dest, dir);
  strcat(dest, fname);
  printf("Reading file %s\n\n", dest);
  if (readfile(dest, a) == -1){
    printf("Unable to open file %s\n", dest);
    return -1;
  }
  memset(dest,0,strlen(dest));
  memset(dir,0,strlen(dir));

  printf("Calculating pressure tensor\n");
  int time = 2000000; // equals 500 ps
  calc_pressure(a, p, time, u, rho, epsilon, sigma, m);
  del_ar(a);


  strcat(dir, "/lammps/");
  char pname[100];
  strcat(dest, home);
  strcat(dest, dir);
  snprintf(pname, sizeof(pname), "ptensor%03d_%d_%d_%02d.csv",
      (int)(angle*100), temp_S, temp_P, pressure);
  strcat(dest, pname);
  printf("Writing pressure data to file %s\n", dest);
  write_pressure(p, dest);


  del_pzz(p);
  return 0;

}
