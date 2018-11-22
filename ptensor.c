#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <immintrin.h>
#include <pwd.h>
#include "pressure.h"
#include "md_struct.h"

extern unsigned int DELTA_T;
// constants
extern const double kB;
extern const double e0;
extern const double pi;
extern const double au;
extern const double atm;
extern const double NA;
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
extern double max_traj_inv;
extern double area_inv;



int main(int argc, char** argv){
  printf("\n********************************************************************\n");

  x_max = lattice_const * 9.0 * sqrt(3.0) / sqrt(2.0);
  y_max = lattice_const * 12. / sqrt(2.0);
  max_traj = 20;
  max_traj_inv = 1.0 / max_traj;
  // Lennard-Jones parameter for argon
  epsilon = 1.67e-21 / e0;
  sigma = 3.4;
  area_inv = 1.0 / area;
  double m = 40.0 * au;
  double u = 0.0;
  double rho = 0.0;
  // coefficients for Born-Mayer potential used in the Argon-Gold interaction
  // according to lammps implementation
  double params[] = {3592.5, 0.34916, 0, 44.99, -2481.30};

  double angle;
  int pressure, temp_S, temp_P;
  setParams(&angle, &pressure, &temp_S, &temp_P, argc, argv);
  double Pressure = pressure / 10.0; // returns pressure in atm
  printf("Evaluate presure tensor for angle %f temp_S %d temp_P %d pressure %f", angle, temp_S, temp_P, Pressure);
  printf("\n********************************************************************\n\n");

// use pressure in datm (i.e. int type) for better comparison
// and set the necessary delta_t accordingly
  if (pressure == 10){
    DELTA_T = 100;
  } else if (pressure == 20){
    DELTA_T = 80;
  } else if (pressure == 40){
    DELTA_T = 50;
  } else if (pressure == 80){
    DELTA_T = 35;
  } else if (pressure > 80){
    DELTA_T = 20;
  } else if(pressure == 5){
    DELTA_T = 150;
  }

  char *home = getenv("HOME");
  char dest[100];


  printf("Initializing Particle struct\n");
  ar *a;
  a = new_ar(NO_OF_ENTRIES);

  printf("Initializing grid\n");
  int l = 180;
  pzz* p;
  p = new_pzz(l);
  make_pgrid(p, 0.0, 60.0);
  int i;
  for(i = 0; i < l; i++){
    p->pz_u[i] = 0.0;
    p->pz_s[i] = 0.0;
    p->pz_k[i] = 0.0;
  }

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
  // for(int i = 0; i < 100; i++){
  //   printdata_i(a, i);
  // }
  memset(dest,0,strlen(dest));
  memset(dir,0,strlen(dir));

  printf("Calculating pressure tensor\n");
  int time = 2000000; // equals 500 ps

  calc_pressure(a, p, time, u, rho, epsilon, sigma, params, m);
  del_ar(a);


  strcat(dir, "/lammps/");
  char pname[100];
  strcat(dest, home);
  strcat(dest, dir);
  snprintf(pname, sizeof(pname), "ptensorA%03d_TS%dK_TP%dK_p%02ddatm.csv",
      (int)(angle*100), temp_S, temp_P, pressure);
  strcat(dest, pname);
  printf("Writing pressure data to file %s\n\n", dest);


  // values for 300 300 1.0
  // obtained from analysis in python script
  double density = 7.768e25;
  double temp_P_obt = 321.314;
  double gasParams[] = {density, (double)temp_P_obt};
  write_pressure(p, dest, gasParams);


  del_pzz(p);
  return 0;

}
