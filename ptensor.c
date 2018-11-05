#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <immintrin.h>
#include "pressure.h"
#include "p_struct.h"
// Constants
double kB;
double e0;
double pi;
double au;
double atm;
double NA;
double area;
double lattice_const; // for gold
double zSurface;
double x_max;
double y_max;
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

  printf("Initializing Particle struct\n");
  NoOfEntries = 10000000;
  ar *a;
  a = new_ar(NoOfEntries);

  printf("Initializing grid\n");
  int l = 150;
  pzz* p;
  p = new_pzz(l);
  make_grid(p, 0.0, 60.0);

  // char fname[] = "/home/becker/lammps/flux/A052_TS300K_TP300K_p10datm.csv";
  char fname[] = "/home/marvin/mdflux-master/A052_TS300K_TP300K_p10datm.csv";
  printf("Reading file %s\n\n", fname);
  if (readfile(fname, a) == -1){
    printf("Unable to open file %s\n", fname);
    return -1;
  }
  // printdata("/home/becker/lammps/testc.txt", a, 10);
  // printf("%d\n", a->step[1000]);
  printf("Calculating pressure tensor\n");
  int time = 2000000; // equal 500 ps
  calc_pressure(a, p, time, u, rho, epsilon, sigma);
  del_ar(a);

  // char pname[] = "/home/becker/lammps/pressuredata.csv";
  char pname[] = "/home/marvin/mdflux-master/pressuredata.csv";
  printf("Writing pressure data to file %s\n", pname);
  write_pressure(p, pname);


  del_pzz(p);
  return 0;

}
