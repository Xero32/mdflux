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


extern double Boundary;
extern double x_max; // lattice_const * 9.0 * sqrt(3.0) / sqrt(2.0);
extern double y_max; // lattice_const * 12. / sqrt(2.0);
const int max_traj = 20;

int GofR_Surface(ar* a, int trajnum, int maxtime, int range, int number, double* glist, double dr, double rmax){
  // make sure z coordinate is shifted with surface as 0
  int i, j, k, t;
  int n = number;
  // printf("length of glist: %d\n", l);

  int particles = 0;
  // use coefficient of linear mapping from r space to array index
  double coeff = 1.0 / dr; // in example this should
  // int l = rmax * coeff;
  int shell_index;
  double shell_vol;
  double x,y,z, dx, dy, dz, R_norm;
  // iterate over trajectories
  k = trajnum;
  printf("Surface, \t%dth trajectory\n", k);

  // iterate over time
  // #pragma omp parallel for num_threads(4)
  for(t = maxtime-range; t < maxtime; t += STW){
    for(i = 0; i < n; i++){
      // assure to analyze the correct step + traj
      // omitted boundary condition, as we should only have bound particles in ar struct
      // omitted trajectory condition, which is implemented in fct 'transferData'
      if(a->step[i] == t && a->traj[i] == k){
        particles += 1;
        x = a->x[i]; // + a->ix[i] * x_max;
        y = a->y[i]; // + a->iy[i] * y_max;
        z = a->z[i];
        // calc mirror images only for bulk contribution
        // ignore ix, iy for bound particles

        for(j = i; j < n; j++){
          if(a->step[j] == t && i != j && a->traj[j] == k){
            dx = x - (a->x[j]); // + a->ix[j] * x_max);
            dy = y - (a->y[j]); // + a->iy[j] * y_max);
            dz = z - (a->z[j]);
            R_norm = sqrt(dx*dx + dy*dy + dz*dz);

            if (R_norm < rmax){
              // index and absolute value of r are correlated by conversion coefficient
              // since we have linear r space
              shell_index = (int) (R_norm * coeff);
              shell_vol = 2.0 * pi * (shell_index * dr) * dr * 1e-20;
              glist[shell_index] += 1.0 / shell_vol;
            }
          }  else if (a->traj[j] > k) break;
        }
      }
    }
  }
  // average particle count over all trajectories and time iterations
  // normalize g(r) with particle number and volume
  return particles;
}

int GofR_Bulk(ar* a, int trajnum, int maxtime, int range, int number, double* glist, double dr, double rmax){
  // make sure z coordinate is shifted with surface as 0
  int i, j, k, t;
  int n = number;
  // printf("length of glist: %d\n", l);

  int particles = 0;
  // use coefficient of linear mapping from r space to array index
  double coeff = 1.0 / dr; // in example this should
  // int l = rmax * coeff;
  int shell_index;
  double shell_vol;
  double x,y,z, dx, dy, dz, R_norm;
  // iterate over trajectories
  k = trajnum;
  printf("Bulk, \t\t%dth trajectory\n", k);

  // iterate over time
  // #pragma omp parallel for num_threads(4)
  for(t = maxtime-range; t < maxtime; t += STW){
    for(i = 0; i < n; i++){
      // assure to analyze the correct step + traj
      // omitted boundary condition, as we should only have bound particles in ar struct
      // omitted trajectory condition, which is implemented in fct 'transferData'

      // in the bulk phase we are interested in the long range behavior
      // therefore we need the mirror images as well
      if(a->step[i] == t && a->traj[i] == k){
        particles += 1;
        x = a->x[i] + a->ix[i] * x_max;
        y = a->y[i] + a->iy[i] * y_max;
        z = a->z[i];
        // calc mirror images only for bulk contribution
        // ignore ix, iy for bound particles (see function above)

        for(j = i; j < n; j++){
          if(a->step[j] == t && i != j && a->traj[j] == k){
            dx = x - (a->x[j] + a->ix[j] * x_max);
            dy = y - (a->y[j] + a->iy[j] * y_max);
            dz = z - (a->z[j]);
            R_norm = sqrt(dx*dx + dy*dy + dz*dz);

            if (R_norm < rmax){
              shell_index = (int) (R_norm * coeff);
              // convert r and dr from angström to meters
              shell_vol = 4.0 * pi * (shell_index * dr) * (shell_index * dr) * dr * 1e-30;
              glist[shell_index] += 1.0 / shell_vol;
            }
          }  else if (a->traj[j] > k) break;
        }
      }
    }
  }
  // average particle count over all trajectories and time iterations
  // normalize g(r) with particle number and volume
  return particles;
}

void writeGofR(gofr *g, const char* gname){
  //deprecated
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
  double tstart, tend;

  char *home = getenv("HOME");
  x_max = lattice_const * 9.0 * sqrt(3.0) / sqrt(2.0);
  y_max = lattice_const * 12. / sqrt(2.0);
  double area = x_max * y_max * 1.0e-20;
  // read traj data
  char dest[256] = "";
  int numberOfParticles;
  int length = 400;
  const double start = 0.0;
  double end = 100.0;
  double dr = (end - start) / length;
  int trajnum = NumOfTraj;
  printf("init a\n");
  ar *a0 = new_ar(NO_OF_ENTRIES);
  double *glist, *rlist, *glist_bulk;
  glist = (double*) _mm_malloc(length * sizeof(double), 64);
  glist_bulk = (double*) _mm_malloc(length * sizeof(double), 64);
  rlist = (double*) _mm_malloc(length * sizeof(double), 64);
  for (int i = 0; i < length; i++){
    rlist[i] = i * dr;
    glist[i] = 0.0;
    glist_bulk[i] = 0.0;
  }

  printf("init grid\n");
  int maximumtime = 0;
  int timerange = 1600000; // time range corresponds to 100 ps
  if(Pressure > 2.0){
    timerange = 800000;
  }
  double range = timerange / STW;

  // printf("Read Trajectory %d\n", i);
  char dir[] = "/lammps/flux/";
  char fname[64];
  snprintf(fname, sizeof(fname), "A%03d_TS%dK_TP%dK_p%02ddatm.csv",
      (int)(angle*100), temp_S, temp_P, pressure);
  strcat(dest, home);
  strcat(dest, dir);
  strcat(dest, fname);

  // Here we open the tracetory data file and write all the data to one struct
  // later this data will be seperated into surface and bulk states based on
  // the particles' height above the surface.
  // the different particles are later evaluated by seperate radial dirstribution functions
  if (0 == readfileCountBoundParticles(dest, a0, &numberOfParticles)){
    printf("Reading file %s\n", dest);
    printf("Num of Particles %d\n", numberOfParticles);
    getMaxTime(a0, &maximumtime);
  }else{
    printf("Unable to open file %s\n", dest);
    return -1;
  }
  // use new a struct to save relevant data, i.e. bound particles in one single traj
  // ar *a = new_ar(numberOfParticles);


  int npart = 0;
  printf("Max Time: %d, Time Range: %d\n", maximumtime, timerange);
  // actual calculation of g(r)
  tstart = omp_get_wtime();

  #pragma omp parallel
  {
    int i;
    int l = length;
    int npartSurf = 0;
    int npartBulk = 0;
    double *glist_loc;
    glist_loc = (double*) _mm_malloc(length * sizeof(double), 64);
    double *glist_loc_bulk;
    glist_loc_bulk = (double*) _mm_malloc(length * sizeof(double), 64);

    // make sure we start with a zeroed array
    for(i = 0; i < l; i++){
      glist_loc[i] = 0.0;
      glist_loc_bulk[i] = 0.0;
    }

    ar *aBulk = new_ar(NO_OF_ENTRIES);
    ar *aSurf = new_ar(NO_OF_ENTRIES);

    // i is essentially the trajectory index
    #pragma omp for
    for(i = 0; i < max_traj; i++){
      // pick only the relevant trajectory data for the given trajectory
      transferData(a0, aSurf, i);
      transferDataBulk(a0, aBulk, i);
      // here we calculate different pair distr functions
      // the surface particles behave as a 2D arrangement of particles due to the strong binding to the surface
      // whereas the bulk particles behave as free gas particles (at least for low densities)
      npartSurf += GofR_Surface(aSurf, i, maximumtime, timerange, numberOfParticles, glist_loc, dr, end);
      npartBulk += GofR_Bulk(aBulk, i, maximumtime, timerange, numberOfParticles, glist_loc_bulk, dr, end);
    }
    #pragma omp critical
    {
      for(i = 0; i < length; i++){
        glist[i] += glist_loc[i];
        glist_bulk[i] += glist_loc_bulk[i];
      }
      npart += npartSurf;
      npart += npartBulk;
    }
    free(glist_loc);
    free(glist_loc_bulk);
    del_ar(aSurf);
    del_ar(aBulk);
  }
  tend = omp_get_wtime();
  del_ar(a0);
  printf("Time elapsed: %f\n", (double)(tend-tstart));

  printf("New number of Particles: %d\n", npart);
  // double volume = x_max*y_max*9.0;

  double partnum = (double) npart / (trajnum * range);
  printf("Number of Particles per trajectory per timestep: %lf\n", partnum);
  // double density = partnum / volume;
  double density = (Pressure * atm) / (kB * temp_P); // in m^-3
  // double areaDensity = density * (Boundary * 1e-10);
  double areaDensity;
  // area density should be around 5 particles per 15.57 nm^2 at p_ext=1 atm
  double areaParticles;
  // set expected particles per area according to simulation data of average bound particles
  if(pressure == 10){
    areaParticles = 5.5;
  } else if(pressure == 20){
    areaParticles = 11.3;
  } else if(pressure == 40){
    areaParticles = 21.0;
  } else if(pressure == 80){
    areaParticles = 36.8;
  } else{
    areaParticles = 121.5;
    // default value, corresponds to maximum number of adsorption sites
  }
  areaDensity = areaParticles / area;


  memset(dest,0,strlen(dest));
  memset(dir,0,strlen(dir));
  strcat(dir, "/lammps/");
  char gname[100];
  strcat(dest, home);
  strcat(dest, dir);
  snprintf(gname, sizeof(gname), "gofrA%03d_TS%dK_TP%dK_p%02ddatm.csv",
      (int)(angle*100), temp_S, temp_P, pressure);
  strcat(dest, gname);
  FILE* fp = fopen(dest, "w");
  if(fp){
    printf("Writing pressure data to file %s\n", dest);
    for(int i = 0; i < length; i++){
      fprintf(fp, "%lf, %le, %le\n",
        rlist[i], glist[i]/(areaDensity*trajnum*range), glist_bulk[i]/(density*trajnum*range));
    }
    fclose(fp);
  }else{
    printf("Unable to open file %s\n", dest);
  }
  free(glist);
  free(glist_bulk);

  return 0;
}
